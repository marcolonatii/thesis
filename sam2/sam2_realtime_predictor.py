# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import warnings
from collections import OrderedDict
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_frame
import time

class SAM2RealtimePredictor(SAM2Base):
    """The predictor class to handle user interactions and manage inference states."""
    
    def __init__(
        self,
        fill_hole_area=0,
        non_overlap_masks=False,
        clear_non_cond_mem_around_input=False,
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

    @torch.inference_mode()
    def init_state(self, img_cv, offload_video_to_cpu=False, offload_state_to_cpu=False, async_loading_frames=False):
        """
        기존 init_state와 유사하지만, propagation 없이 각 프레임을 독립 처리하기 위한 상태만 관리.
        """
        compute_device = self.device  # device of the model
        img, img_height, img_width = load_frame(img_cv, offload_video_to_cpu=offload_video_to_cpu)
        inference_state = {}
        inference_state["images"] = [img]
        inference_state["num_frames"] = 1
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["video_height"] = img_height
        inference_state["video_width"] = img_width
        inference_state["device"] = compute_device
        inference_state["storage_device"] = torch.device("cpu") if offload_state_to_cpu else compute_device

        # 객체별 프롬프트 및 결과를 저장하는 딕셔너리 (propagation 관련 정보는 사용하지 않음)
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}

        # 단일 프레임(0번 프레임)에 대해 이미지 특성 계산 (배치 사이즈 1)
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    @torch.inference_mode()
    def add_frame(self, inference_state, img_cv, offload_video_to_cpu=False):
        """새로운 프레임을 추가하고 이미지 특성을 계산 (배치 사이즈 1)."""
        img, _, _ = load_frame(img_cv, offload_video_to_cpu=offload_video_to_cpu)
        inference_state["images"].append(img)
        inference_state["num_frames"] += 1
        st = time.time()
        self._get_image_feature(inference_state, frame_idx=inference_state["num_frames"]-1, batch_size=1)
        et = time.time()
        print(f"Time taken to load and process frame {inference_state['num_frames']-1}: {et-st:.2f} seconds")
        return inference_state

    def _obj_id_to_idx(self, inference_state, obj_id):
        """객체 id를 내부 인덱스로 변환 (없으면 새로 추가)."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx
        obj_idx = len(inference_state["obj_id_to_idx"])
        inference_state["obj_id_to_idx"][obj_id] = obj_idx
        inference_state["obj_idx_to_id"][obj_idx] = obj_id
        inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
        inference_state["point_inputs_per_obj"][obj_idx] = {}
        inference_state["mask_inputs_per_obj"][obj_idx] = {}
        inference_state["output_dict_per_obj"][obj_idx] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        inference_state["temp_output_dict_per_obj"][obj_idx] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        inference_state["frames_tracked_per_obj"][obj_idx] = {}
        return obj_idx

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """내부 인덱스를 객체 id로 변환."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """추가된 객체 수 반환."""
        return len(inference_state["obj_idx_to_id"])


    def get_mask(
        self,
        inference_state,
        frame_idx
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)
        obj_ids = inference_state["obj_ids"]
        batch_size = self._get_obj_num(inference_state)
        reverse = False

        pred_masks_per_obj = [None] * batch_size
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via _run_single_frame_inference because the
            # number of clicks on each object might be different.
            if frame_idx in obj_output_dict["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = obj_output_dict[storage_key][frame_idx]
                device = inference_state["device"]
                pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                if self.clear_non_cond_mem_around_input:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_obj_non_cond_mem_around_input(
                        inference_state, frame_idx, obj_idx
                    )
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=obj_output_dict,
                    frame_idx=frame_idx,
                    batch_size=1,  # run on the slice of a single object
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                obj_output_dict[storage_key][frame_idx] = current_out
            inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                "reverse": reverse
            }
            pred_masks_per_obj[obj_idx] = pred_masks
        # Resize the output mask to the original video resolution (we directly use
        # the mask scores on GPU for output to avoid any CPU conversion in between)
        if len(pred_masks_per_obj) > 1:
            all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
        else:
            all_pred_masks = pred_masks_per_obj[0]
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, all_pred_masks
        )
        if frame_idx >= 10:
            self.clear_old_frames(inference_state, frame_idx-9)

        return frame_idx, obj_ids, video_res_masks, inference_state
    
    @torch.inference_mode()
    def clear_old_frames(self, inference_state, min_valid_frame_idx):
        # 1) cached_features
        old_frame_keys = []
        for frame_idx in inference_state["cached_features"].keys():
            if frame_idx < min_valid_frame_idx:
                old_frame_keys.append(frame_idx)
        for old_f in old_frame_keys:
            inference_state["cached_features"].pop(old_f, None)

        # 2) remove old images from inference_state["images"]
        num_images = len(inference_state["images"])
        for old_idx in range(min_valid_frame_idx):
            if old_idx < num_images:
                inference_state["images"][old_idx] = None  # or pop(0), but careful with index shifts

        # 3) point_inputs_per_obj, mask_inputs_per_obj
        for obj_idx in list(inference_state["point_inputs_per_obj"].keys()):
            point_dict = inference_state["point_inputs_per_obj"][obj_idx]
            old_keys = [k for k in point_dict if k < min_valid_frame_idx]
            for k in old_keys:
                point_dict.pop(k, None)

            mask_dict = inference_state["mask_inputs_per_obj"][obj_idx]
            old_keys = [k for k in mask_dict if k < min_valid_frame_idx]
            for k in old_keys:
                mask_dict.pop(k, None)

        # 4) output_dict_per_obj (cond_frame_outputs, non_cond_frame_outputs)
        for obj_idx in list(inference_state["output_dict_per_obj"].keys()):
            out_dict = inference_state["output_dict_per_obj"][obj_idx]
            for mode in ["non_cond_frame_outputs"]:
                frame_out = out_dict[mode]
                old_keys = [k for k in frame_out if k < min_valid_frame_idx]
                for k in old_keys:
                    frame_out.pop(k, None)

        # 5) temp_output_dict_per_obj (cond_frame_outputs, non_cond_frame_outputs)
        for obj_idx in list(inference_state["temp_output_dict_per_obj"].keys()):
            out_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            for mode in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                frame_out = out_dict[mode]
                old_keys = [k for k in frame_out if k < min_valid_frame_idx]
                for k in old_keys:
                    frame_out.pop(k, None)

        # 6) frames_tracked_per_obj
        for obj_idx in list(inference_state["frames_tracked_per_obj"].keys()):
            track_dict = inference_state["frames_tracked_per_obj"][obj_idx]
            old_keys = [k for k in track_dict if k < min_valid_frame_idx]
            for k in old_keys:
                track_dict.pop(k, None)

    @torch.inference_mode()
    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, points=None, labels=None, clear_old_points=True, normalize_coords=True, box=None):
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)

        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)

        if points.dim() == 2:
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        if box is not None:
            if not clear_old_points:
                raise ValueError("cannot add box without clearing old points; use clear_old_points=True")
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device).reshape(1,2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)

        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)

        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        reverse = False if is_init_cond_frame else obj_frames_tracked[frame_idx]["reverse"]

        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].to(inference_state["device"], non_blocking=True)
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(inference_state, frame_idx, is_cond=is_cond, consolidate_at_video_res=True)
        _, video_res_masks = self._get_orig_video_res_output(inference_state, consolidated_out["pred_masks_video_res"])
        return frame_idx, obj_ids, video_res_masks, inference_state

    @torch.inference_mode()
    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask.unsqueeze(0).unsqueeze(0).float().to(inference_state["device"])
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = F.interpolate(mask_inputs_orig, size=(self.image_size, self.image_size), align_corners=False, mode="bilinear", antialias=True)
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)

        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        reverse = False if is_init_cond_frame else obj_frames_tracked[frame_idx]["reverse"]

        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(inference_state, frame_idx, is_cond=is_cond, consolidate_at_video_res=True)
        _, video_res_masks = self._get_orig_video_res_output(inference_state, consolidated_out["pred_masks_video_res"])
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = F.interpolate(any_res_masks, size=(video_H, video_W), mode="bilinear", align_corners=False)
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(self, inference_state, frame_idx, is_cond, consolidate_at_video_res=False):
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        if consolidate_at_video_res:
            H, W = inference_state["video_height"], inference_state["video_width"]
            mask_key = "pred_masks_video_res"
        else:
            H = W = self.image_size // 4
            mask_key = "pred_masks"
        consolidated_out = {
            mask_key: torch.full(
                size=(batch_size, 1, H, W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"]
            )
        }
        for obj_idx in range(batch_size):
            temp_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            out = temp_dict[storage_key].get(frame_idx, None)
            if out is None:
                out = inference_state["output_dict_per_obj"][obj_idx][storage_key].get(frame_idx, None)
            if out is None:
                continue
            obj_mask = out["pred_masks"]
            cons = consolidated_out[mask_key]
            if obj_mask.shape[-2:] == cons.shape[-2:]:
                cons[obj_idx:obj_idx+1] = obj_mask
            else:
                resized = F.interpolate(obj_mask, size=cons.shape[-2:], mode="bilinear", align_corners=False)
                cons[obj_idx:obj_idx+1] = resized
        return consolidated_out

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError("No input points or masks are provided; please add inputs first.")
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            temp_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            for is_cond in [False, True]:
                key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                for frame_idx, out in temp_dict[key].items():
                    if out["maskmem_features"] is None:
                        high_res = F.interpolate(
                            out["pred_masks"].to(inference_state["device"]),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False
                        )
                        feats, pos = self._run_memory_encoder(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            batch_size=1,
                            high_res_masks=high_res,
                            object_score_logits=out["object_score_logits"],
                            is_mask_from_pts=True,
                        )
                        out["maskmem_features"] = feats
                        out["maskmem_pos_enc"] = pos
                    obj_output_dict[key][frame_idx] = out
                    if self.clear_non_cond_mem_around_input:
                        self._clear_non_cond_mem_around_input(inference_state, frame_idx, obj_idx)
            if not inference_state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"]:
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)
                raise RuntimeError(f"No input points or masks for object id {obj_id}")

    @torch.inference_mode()
    def propagate_in_video(self, inference_state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
        self.propagate_in_video_preflight(inference_state)
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        if start_frame_idx is None:
            start_frame_idx = min(t for d in inference_state["output_dict_per_obj"].values() for t in d["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end = max(start_frame_idx - max_frame_num_to_track, 0)
            order = range(start_frame_idx, end-1, -1) if start_frame_idx>0 else []
        else:
            end = min(start_frame_idx + max_frame_num_to_track, num_frames-1)
            order = range(start_frame_idx, end+1)
        for frame_idx in tqdm(order, desc="propagate in video"):
            preds = []
            for obj_idx in range(batch_size):
                od = inference_state["output_dict_per_obj"][obj_idx]
                if frame_idx in od["cond_frame_outputs"]:
                    out = od["cond_frame_outputs"][frame_idx]
                    pm = out["pred_masks"].to(inference_state["device"], non_blocking=True)
                    if self.clear_non_cond_mem_around_input:
                        self._clear_non_cond_mem_around_input(inference_state, frame_idx, obj_idx)
                else:
                    out, pm = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=od,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    od["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
                preds.append(pm)
            all_pm = torch.cat(preds, dim=0) if len(preds)>1 else preds[0]
            _, vid_masks = self._get_orig_video_res_output(inference_state, all_pm)
            yield frame_idx, obj_ids, vid_masks

    @torch.inference_mode()
    def vectorized_track_frame(self, inference_state, frame_idx, reverse=False):
        B = self._get_obj_num(inference_state)
        *_, feats = self._get_image_feature(inference_state, frame_idx, B)
        max_p = max(
            (inference_state["point_inputs_per_obj"][i].get(frame_idx, {}).get("point_coords", torch.empty(0,2)).shape[0]
             for i in range(B)), default=0
        )
        coords = torch.zeros((B, max_p, 2), device=self.device)
        labels = -torch.ones((B, max_p), dtype=torch.int32, device=self.device)
        for i in range(B):
            p = inference_state["point_inputs_per_obj"][i].get(frame_idx)
            if p is not None:
                n = p["point_coords"].shape[0]
                coords[i,:n] = p["point_coords"]
                labels[i,:n] = p["point_labels"]
        batched_pts = {"point_coords": coords, "point_labels": labels}

        compact_out, pred_masks = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict={},
            frame_idx=frame_idx,
            batch_size=B,
            is_init_cond_frame=False,
            point_inputs=batched_pts,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=True,
        )

        for i in range(B):
            out_slice = {k: (None if compact_out[k] is None else compact_out[k][i:i+1]) for k in compact_out}
            key = "cond_frame_outputs" if frame_idx in inference_state["frames_tracked_per_obj"][i] else "non_cond_frame_outputs"
            inference_state["output_dict_per_obj"][i][key][frame_idx] = out_slice

        _, video_masks = self._get_orig_video_res_output(inference_state, pred_masks)
        return video_masks

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        old_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        if old_idx is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(f"Cannot remove object id {obj_id}")
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames

        inputs = set(inference_state["point_inputs_per_obj"][old_idx]) | set(inference_state["mask_inputs_per_obj"][old_idx])
        for f in inputs:
            self.clear_all_prompts_in_frame(inference_state, f, obj_id, need_output=False)

        old_ids = inference_state["obj_ids"]
        old_inds = list(range(len(old_ids)))
        remain = [i for i in old_inds if i != old_idx]
        new_ids = [old_ids[i] for i in remain]
        new_inds = list(range(len(new_ids)))
        idx_map = dict(zip(remain, new_inds))

        inference_state["obj_id_to_idx"] = dict(zip(new_ids, new_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_inds, new_ids))
        inference_state["obj_ids"] = new_ids

        def remap(d):
            items = list(d.items())
            d.clear()
            for k,v in items:
                if k in idx_map:
                    d[idx_map[k]] = v

        for dname in ["point_inputs_per_obj","mask_inputs_per_obj","output_dict_per_obj","temp_output_dict_per_obj","frames_tracked_per_obj"]:
            remap(inference_state[dname])

        if need_output:
            for f in inputs:
                is_cond = any(f in inference_state["temp_output_dict_per_obj"][i]["cond_frame_outputs"] for i in range(len(new_ids)))
                cons = self._consolidate_temp_output_across_obj(inference_state, f, is_cond, consolidate_at_video_res=True)
                _, vm = self._get_orig_video_res_output(inference_state, cons["pred_masks_video_res"])
                updated_frames.append((f, vm))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx, obj_idx):
        r = self.memory_temporal_stride_for_eval
        ns = self.num_maskmem
        begin = frame_idx - r*ns
        end = frame_idx + r*ns
        for t in range(begin, end+1):
            inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].pop(t, None)

    @torch.inference_mode()
    def clear_all_prompts_in_frame(self, inference_state, frame_idx, obj_id, need_output=True):
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        pi = inference_state["point_inputs_per_obj"][obj_idx]
        mi = inference_state["mask_inputs_per_obj"][obj_idx]
        pi.pop(frame_idx, None)
        mi.pop(frame_idx, None)

        tmp = inference_state["temp_output_dict_per_obj"][obj_idx]
        tmp["cond_frame_outputs"].pop(frame_idx, None)
        tmp["non_cond_frame_outputs"].pop(frame_idx, None)

        out = inference_state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        if out is not None:
            inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"][frame_idx] = out
            inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None)

        if not need_output:
            return

        is_cond = any(frame_idx in inference_state["temp_output_dict_per_obj"][i]["cond_frame_outputs"]
                      for i in range(len(inference_state["obj_ids"])))
        cons = self._consolidate_temp_output_across_obj(inference_state, frame_idx, is_cond, consolidate_at_video_res=True)
        _, vm = self._get_orig_video_res_output(inference_state, cons["pred_masks_video_res"])
        return frame_idx, inference_state["obj_ids"], vm

    @torch.inference_mode()
    def reset_state(self, inference_state):
        self._reset_tracking_results(inference_state)
        for k in ["obj_id_to_idx","obj_idx_to_id","obj_ids",
                  "point_inputs_per_obj","mask_inputs_per_obj",
                  "output_dict_per_obj","temp_output_dict_per_obj",
                  "frames_tracked_per_obj"]:
            if inference_state[k] is not None:
                inference_state[k].clear()

    def _reset_tracking_results(self, inference_state):
        for d in [inference_state["point_inputs_per_obj"],
                  inference_state["mask_inputs_per_obj"],
                  inference_state["output_dict_per_obj"],
                  inference_state["temp_output_dict_per_obj"],
                  inference_state["frames_tracked_per_obj"]]:
            for v in d.values():
                v.clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            inference_state["cached_features"][frame_idx] = (image, backbone_out)

        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": [f.expand(batch_size, -1, -1, -1) for f in backbone_out["backbone_fpn"]],
            "vision_pos_enc": [p.expand(batch_size, -1, -1, -1) for p in backbone_out["vision_pos_enc"]],
        }
        features = self._prepare_backbone_features(expanded_backbone_out)
        return (expanded_image, *features)

    def _run_single_frame_inference(
        self, inference_state, output_dict, frame_idx, batch_size,
        is_init_cond_frame, point_inputs, mask_inputs, reverse,
        run_mem_encoder, prev_sam_mask_logits=None
    ):
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, self.fill_hole_area)
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(self, inference_state, frame_idx, batch_size, high_res_masks, object_score_logits, is_mask_from_pts):
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(inference_state, frame_idx, batch_size)
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, {"maskmem_pos_enc": maskmem_pos_enc})
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        model_constants = inference_state["constants"]
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            maskmem_features = model_constants["maskmem_pos_enc"]
            batch_size = out_maskmem_pos_enc[0].size(0)
            return [x.expand(batch_size, -1, -1, -1) for x in maskmem_features]
        else:
            return None

class SAM2RealtimePredictorVOS(SAM2RealtimePredictor):
    """Optimized for VOS."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compile_all_components()

    def _compile_all_components(self):
        print("Compiling all components for VOS setting. First time may be very slow.")
        self.memory_encoder.forward = torch.compile(
            self.memory_encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        self.memory_attention.forward = torch.compile(
            self.memory_attention.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=True,
        )
        self.sam_prompt_encoder.forward = torch.compile(
            self.sam_prompt_encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        self.sam_mask_decoder.forward = torch.compile(
            self.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

    def forward_image(self, img_batch: torch.Tensor):
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        for i in range(len(backbone_out["backbone_fpn"])):
            backbone_out["backbone_fpn"][i] = backbone_out["backbone_fpn"][i].clone()
            backbone_out["vision_pos_enc"][i] = backbone_out["vision_pos_enc"][i].clone()
        return backbone_out

    def _forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
        B = backbone_features.size(0)
        device = backbone_features.device
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        if mask_inputs is not None:
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(mask_inputs.float(), size=self.sam_prompt_encoder.mask_input_size, align_corners=False, mode="bilinear", antialias=True)
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt
        )
        sparse_embeddings = sparse_embeddings.clone()
        dense_embeddings = dense_embeddings.clone()
        image_pe = self.sam_prompt_encoder.get_dense_pe().clone()

        (low_res_multimasks, ious, sam_output_tokens, object_score_logits) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        low_res_multimasks = low_res_multimasks.clone()
        ious = ious.clone()
        sam_output_tokens = sam_output_tokens.clone()
        object_score_logits = object_score_logits.clone()

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        sam_output_token = sam_output_tokens[:, 0]

        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            is_obj_appearing = (object_score_logits > 0).float()
            if self.soft_no_obj_ptr:
                lambda_obj = object_score_logits.sigmoid()
            else:
                lambda_obj = is_obj_appearing
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_obj * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_obj) * self.no_obj_ptr

        return low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def _encode_new_memory(self, current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)

        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)

        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"].clone()
        maskmem_pos_enc = [m.clone() for m in maskmem_out["vision_pos_enc"]]

        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (1 - is_obj_appearing[..., None, None]) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)

        return maskmem_features, maskmem_pos_enc