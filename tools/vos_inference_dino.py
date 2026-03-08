# -*- coding: utf-8 -*-
"""
VOS (Video Object Segmentation) inference with SAM2 + Grounding DINO.

Pipeline:
  1. Load a video (.mp4) or a folder of JPEG frames.
  2. Run Grounding DINO on the anchor frame (default: frame 0) to get an
     XYXY bounding box for the target object(s).
  3. Register the box prompt with SAM2VideoPredictor.
  4. Propagate the mask forward through the video (and optionally backward
     from the anchor frame to frame 0).
  5. Save per-frame binary masks and optionally overlay visualisations.

Supported input formats:
  - Directory of JPEG/JPG images named as integers (e.g. 00000.jpg, …)
    →  matches what SAM2's load_video_frames_from_jpg_images expects
  - .mp4 video file  (requires `decord`)

Usage examples:
  # From a JPEG folder
  python vos_inference.py \
      -video path/to/frames_dir \
      -out_dir path/to/output \
      -sam2_cfg sam2_hiera_b+.yaml \
      -checkpoint ../thesis/checkpoints/sam2_hiera_base_plus.pt

  # From an mp4, custom anchor frame and text prompt
  python vos_inference.py \
      -video clip.mp4 \
      -out_dir output/ \
      -anchor_frame 5 \
      -text_prompt "camouflaged frog"

  # Multi-object: separate prompts with '|'
  python vos_inference.py \
      -video frames/ \
      -out_dir output/ \
      -text_prompt "camouflaged frog|camouflaged fish"
"""

import os
import re
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# SAM2
# Add the repository root (parent of `tools/`) to PYTHONPATH so `sam2` can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sam2.build_sam import build_sam2_video_predictor

# Grounding DINO (HuggingFace)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

torch.manual_seed(2024)
torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def caption_from_path(img_path: str) -> str:
    """Derive 'camouflaged <name>' from a file basename (same as train_gdino.py)."""
    base = os.path.splitext(os.path.basename(img_path))[0]
    name = re.sub(r'[_\-]+', ' ', base)
    name = re.sub(r'\d+', '', name).strip()
    name = re.sub(r'\s+', ' ', name).strip()
    if name == '':
        name = 'animal'
    return f'camouflaged {name.lower()}'


@torch.no_grad()
def get_gdino_boxes(
    pil_image: Image.Image,
    text_prompts: list,
    gdino_model,
    gdino_processor,
    device,
    box_threshold: float = 0.3,
    text_threshold: float = 0.3,
    fallback_full_image: bool = False,
) -> list:
    """
    Run Grounding DINO for each text prompt and return the best XYXY box
    per prompt (numpy float32, shape (4,)).

    If no detection is found for a prompt:
      - fallback_full_image=True  → return full-image box as fallback.
      - fallback_full_image=False → return None for that prompt (Strategy A: caller skips it).
    """
    H, W = pil_image.size[1], pil_image.size[0]
    boxes_out = []
    for prompt in text_prompts:
        gdino_text = prompt + '.'
        inputs = gdino_processor(
            images=pil_image,
            text=gdino_text,
            return_tensors='pt'
        ).to(device)
        outputs = gdino_model(**inputs)
        results = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(H, W)],
        )
        boxes  = results[0]['boxes']   # (N, 4) XYXY px
        scores = results[0]['scores']  # (N,)

        if len(boxes) == 0:
            if fallback_full_image:
                box_xyxy = np.array([0.0, 0.0, float(W), float(H)], dtype=np.float32)
            else:
                box_xyxy = None   # Strategy A: caller will skip this frame/prompt
        else:
            best = scores.argmax().item()
            box_xyxy = boxes[best].cpu().numpy().astype(np.float32)

        boxes_out.append(box_xyxy)
    return boxes_out


def sorted_frame_paths(video_path: str) -> list:
    """
    Return sorted list of JPEG/PNG frame paths inside a directory.
    Falls back to empty list for video files (handled by SAM2 internally).
    """
    if not os.path.isdir(video_path):
        return []
    ext = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    paths = [
        os.path.join(video_path, f)
        for f in os.listdir(video_path)
        if os.path.splitext(f)[1] in ext
    ]
    paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return paths


def save_mask(mask: np.ndarray, path: str) -> None:
    """Save a boolean/binary mask as a PNG (0 / 255)."""
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def blend_mask_on_frame(frame_rgb: np.ndarray, mask: np.ndarray,
                        color=(0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    """Blend a binary mask over an RGB frame."""
    overlay = frame_rgb.copy()
    overlay[mask.astype(bool)] = (
        (1 - alpha) * overlay[mask.astype(bool)] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='SAM2 VOS inference with Grounding DINO')
parser.add_argument('-video',         type=str, required=True,
                    help='Path to a JPEG folder or an .mp4 video file')
parser.add_argument('-out_dir',       type=str, default='vos_output',
                    help='Root output directory')
parser.add_argument('-sam2_cfg',      type=str, default='sam2_hiera_b+.yaml',
                    help='SAM2 Hydra config filename')
parser.add_argument('-checkpoint',    type=str,
                    default='../thesis/checkpoints/sam2_hiera_base_plus.pt',
                    help='Path to SAM2 checkpoint (.pt)')
parser.add_argument('-gdino_model',   type=str,
                    default='IDEA-Research/grounding-dino-base',
                    help='HuggingFace model ID or local path for Grounding DINO')
parser.add_argument('-text_prompt',   type=str, default=None,
                    help='Text prompt(s), separated by "|" for multi-object. '
                         'If omitted, derived from folder/file name.')
parser.add_argument('-anchor_frame',  type=int, default=0,
                    help='Frame index on which Grounding DINO is run (0-based)')
parser.add_argument('-propagate_backward', action='store_true', default=False,
                    help='Also propagate backward from anchor_frame to frame 0')
parser.add_argument('-box_threshold', type=float, default=0.3)
parser.add_argument('-text_threshold',type=float, default=0.3)
parser.add_argument('-save_overlay',  action='store_true', default=False,
                    help='Save colour-blended overlays alongside the binary masks')
parser.add_argument('-offload_video', action='store_true', default=False,
                    help='Offload video frames to CPU memory (saves GPU VRAM)')
parser.add_argument('--device',       type=str, default='cuda:0')
args = parser.parse_args()

device = torch.device(args.device)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
masks_dir   = os.path.join(args.out_dir, 'masks')
overlay_dir = os.path.join(args.out_dir, 'overlay')
os.makedirs(masks_dir, exist_ok=True)
if args.save_overlay:
    os.makedirs(overlay_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Build SAM2 video predictor once
    print(f'Loading SAM2 ({args.sam2_cfg}) …')
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.checkpoint,
        device=device,
        mode='eval',
        apply_postprocessing=True,
    )
    predictor.eval()
    print('SAM2 loaded.')

    # Build Grounding DINO once
    print(f'Loading Grounding DINO ({args.gdino_model}) …')
    gdino_processor = AutoProcessor.from_pretrained(args.gdino_model)
    gdino_model_hf = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.gdino_model
    ).to(device)
    gdino_model_hf.eval()
    print('Grounding DINO loaded.')

    def process_single_video(video_path: str, video_name: str | None = None):
        """Process one video given a directory of frames or an mp4 path.
        If `video_name` is provided, it's used for caption generation when no
        explicit text prompt is given.
        """
        # Resolve text prompts for this video
        if args.text_prompt is not None:
            text_prompts = [p.strip() for p in args.text_prompt.split('|') if p.strip()]
        else:
            text_prompts = [caption_from_path(video_name or video_path)]
        n_objects = len(text_prompts)
        print(f'\nProcessing "{video_name or video_path}" — tracking {n_objects} object(s): {text_prompts}')

        # Collect frame paths and anchor
        frame_paths = sorted_frame_paths(video_path)
        is_image_folder = len(frame_paths) > 0
        if is_image_folder:
            n_frames = len(frame_paths)
            anchor_idx = min(args.anchor_frame, n_frames - 1)
            anchor_path = frame_paths[anchor_idx]
            anchor_pil = Image.open(anchor_path).convert('RGB')
        else:
            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.anchor_frame)
            ret, bgr = cap.read()
            cap.release()
            if not ret:
                print(f'  [WARN] Could not read anchor frame {args.anchor_frame} from {video_path}; skipping')
                return
            anchor_idx = args.anchor_frame
            anchor_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        print(f'  Video length: {n_frames} frames | anchor={anchor_idx}')

        # Run Grounding DINO on every frame.
        # Policy per object:
        #   - Real detection found  → always register the box.
        #   - No detection, object not yet seen (cold-start) → register full-image box so
        #     SAM2 has at least one prompt to start from.
        #   - No detection, object already seen before → skip (Strategy A: let SAM2
        #     propagate from its own memory rather than injecting a noisy full-frame box).
        print(f'  Running Grounding DINO on all {n_frames} frames …')
        per_frame_detections: dict = {}   # frame_idx -> list of (obj_id, box_xyxy)
        n_detected = 0
        n_coldstart = 0
        n_skipped = 0
        ever_seen: set = set()  # obj_ids that have had at least one real detection

        for frame_idx in tqdm(range(n_frames), desc='  GDINO', leave=False):
            if is_image_folder:
                frame_pil = Image.open(frame_paths[frame_idx]).convert('RGB')
            else:
                _cap = cv2.VideoCapture(video_path)
                _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                _ret, _bgr = _cap.read()
                _cap.release()
                if not _ret:
                    continue
                frame_pil = Image.fromarray(cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB))

            W_f, H_f = frame_pil.size
            full_box = np.array([0.0, 0.0, float(W_f), float(H_f)], dtype=np.float32)

            box_list = get_gdino_boxes(
                frame_pil, text_prompts, gdino_model_hf, gdino_processor,
                device, args.box_threshold, args.text_threshold,
                fallback_full_image=False,   # returns None on miss
            )
            for obj_id, box_xyxy in enumerate(box_list, start=1):
                if box_xyxy is not None:
                    # Real detection — register and mark as seen
                    per_frame_detections.setdefault(frame_idx, []).append((obj_id, box_xyxy))
                    ever_seen.add(obj_id)
                    n_detected += 1
                elif obj_id not in ever_seen:
                    # Cold-start: first time we see this object, GDINO missed →
                    # use full-image box so SAM2 has something to work with
                    per_frame_detections.setdefault(frame_idx, []).append((obj_id, full_box))
                    n_coldstart += 1
                else:
                    # Already seen before, GDINO misses now → skip (Strategy A)
                    n_skipped += 1

        n_hit = len(per_frame_detections)
        print(f'  GDINO: {n_detected} real detection(s) + {n_coldstart} cold-start full-image box(es) '
              f'across {n_hit}/{n_frames} frames ({n_skipped} miss(es) skipped, '
              f'SAM2 memory carries forward).')

        if n_hit == 0:
            print(f'  [WARN] No prompts registered for any frame; skipping video.')
            return

        # Register all detections with SAM2 before propagation.
        print('  Initialising SAM2 video state and registering box prompts …')
        with torch.inference_mode():
            inference_state = predictor.init_state(
                video_path=video_path,
                offload_video_to_cpu=args.offload_video,
            )

            for fidx, obj_box_list in sorted(per_frame_detections.items()):
                for obj_id, box_xyxy in obj_box_list:
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=fidx,
                        obj_id=obj_id,
                        box=box_xyxy,
                        normalize_coords=True,
                    )
            print(f'  Registered {n_detected} box prompt(s) across {n_hit} frame(s).')

        # helper: propagate and save (re-uses outer `masks_dir`, `overlay_dir`)
        def run_propagation(reverse: bool = False):
            direction = 'backward' if reverse else 'forward'
            # With per-frame GDINO prompts registered on all frames,
            # always propagate from frame 0 forward (or from the last frame backward).
            start_idx = 0
            print(f'    Propagating {direction} from frame {start_idx} …')

            with torch.inference_mode():
                for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=start_idx,
                    reverse=reverse,
                ):
                    # Aggregate a combined per-frame mask (one file per frame per video)
                    combined_mask = None
                    for i, obj_id in enumerate(obj_ids):
                        mask_logit = video_res_masks[i]
                        # Squeeze to 2D HxW; SAM2 may return (1, H, W) or (1, 1, H, W)
                        binary_mask = (mask_logit > 0.0).cpu().numpy().squeeze().astype(np.uint8)

                        fname = f'{frame_idx:05d}_obj{obj_id}.png'
                        obj_mask_dir = os.path.join(masks_dir, f'obj{obj_id:02d}')
                        os.makedirs(obj_mask_dir, exist_ok=True)
                        save_mask(binary_mask, os.path.join(obj_mask_dir, fname))

                        # build combined mask
                        if combined_mask is None:
                            combined_mask = np.zeros_like(binary_mask, dtype=np.uint8)
                        combined_mask[binary_mask > 0] = 255

                        if args.save_overlay:
                            obj_overlay_dir = os.path.join(overlay_dir, f'obj{obj_id:02d}')
                            os.makedirs(obj_overlay_dir, exist_ok=True)
                            if is_image_folder:
                                frame_pil = Image.open(frame_paths[frame_idx]).convert('RGB')
                                frame_rgb = np.array(frame_pil)
                            else:
                                _cap = cv2.VideoCapture(video_path)
                                _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                _, _bgr = _cap.read()
                                _cap.release()
                                frame_rgb = cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)

                            colors = [(0, 255, 0), (255, 80, 0), (0, 80, 255), (255, 255, 0), (255, 0, 255)]
                            color = colors[(obj_id - 1) % len(colors)]
                            blended = blend_mask_on_frame(frame_rgb, binary_mask, color=color)
                            Image.fromarray(blended).save(os.path.join(obj_overlay_dir, fname))

                    # Save the combined binary mask into a per-video folder expected by generate_videos.py
                    per_video_name = video_name or os.path.basename(video_path.rstrip(os.sep))
                    per_video_dir = os.path.join(masks_dir, per_video_name)
                    os.makedirs(per_video_dir, exist_ok=True)
                    if combined_mask is None:
                        # no objects found for this frame; create empty mask
                        combined_mask = np.zeros((1, 1), dtype=np.uint8)
                    combined_fname = f'{frame_idx:05d}.png'
                    Image.fromarray(combined_mask).save(os.path.join(per_video_dir, combined_fname))

        # Forward
        run_propagation(reverse=False)
        # Backward if requested (propagates from last frame back to 0)
        if args.propagate_backward:
            run_propagation(reverse=True)

        # Merge per-object masks into combined masks if multiple objects
        if n_objects > 1:
            print('  Merging per-object masks into combined masks …')
            combined_dir = os.path.join(masks_dir, 'combined')
            os.makedirs(combined_dir, exist_ok=True)
            obj_dirs = [os.path.join(masks_dir, f'obj{i+1:02d}') for i in range(n_objects)]
            if len(obj_dirs) == 0 or not os.path.exists(obj_dirs[0]):
                return
            ref_files = sorted(os.listdir(obj_dirs[0]))
            for fname in tqdm(ref_files, desc='    combining'):
                combined = np.zeros_like(np.array(Image.open(os.path.join(obj_dirs[0], fname)).convert('L')), dtype=np.uint8)
                for oi, odir in enumerate(obj_dirs):
                    fpath = os.path.join(odir, fname)
                    if os.path.exists(fpath):
                        m = np.array(Image.open(fpath).convert('L')) > 0
                        combined[m] = (oi + 1) * (255 // n_objects)
                Image.fromarray(combined).save(os.path.join(combined_dir, fname))

        print(f'  Done. Masks for "{video_name or video_path}" saved to: {masks_dir}')

    # If the provided `-video` is a directory that contains subdirectories, treat
    # each subdirectory as a separate video (frames/<video_name>/...)
    if os.path.isdir(args.video):
        subdirs = [p for p in os.listdir(args.video) if os.path.isdir(os.path.join(args.video, p))]
        if len(subdirs) > 0:
            subdirs.sort()
            print(f'Found {len(subdirs)} subfolders in {args.video}; processing each as a video.')
            for n, vname in enumerate(subdirs, start=1):
                print(f'\n[{n}/{len(subdirs)}] Processing video: {vname}')
                process_single_video(os.path.join(args.video, vname), vname)
            print(f'Completed processing {len(subdirs)} videos; outputs under {masks_dir}')
            return

    # Otherwise treat `-video` as a single video (folder of frames or mp4)
    process_single_video(args.video, None)


if __name__ == '__main__':
    main()
