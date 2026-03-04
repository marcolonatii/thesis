# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# VLSAM-VOS: SAM2 video-object-segmentation with BLIP+Mamba-derived prompts.
#
# Pipeline
# --------
# Phase 1 – VLSAM frame scoring
#   For every frame the BLIP vision model + Mamba text model produce
#   sparse/dense embeddings that are fed to the SAM2 mask-decoder →
#   a per-frame binary mask is predicted.
#
# Phase 2 – SAM2 temporal propagation
#   The SAM2 video-predictor is seeded with the VLSAM masks via
#   `add_new_mask` (one call per frame) and then `propagate_in_video`
#   produces the final temporally-consistent masks.

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from typing import Any, Optional, Tuple

from transformers import (
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
    MambaModel,
)
from sam2.build_sam import build_sam2, build_sam2_video_predictor


# ---------------------------------------------------------------------------
# DAVIS palette
# ---------------------------------------------------------------------------
DAVIS_PALETTE = (
    b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80"
    b"\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80"
    b"\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00"
    b"\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0"
    b"\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80"
    b"@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0"
    b"\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0"
    b"@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0"
    b"@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80"
    b"\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`"
    b"\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00"
    b" @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@"
    b"\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0"
    b"\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00"
    b"\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0"
    b"\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 "
    b"\x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0"
    b"\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80"
    b"\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80"
    b"\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0"
    b"\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0"
    b"@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@"
    b"\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0"
    b"@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0"
    b"\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0"
    b"\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00"
    b" `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``"
    b"\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 "
    b"\xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0"
    b"\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`"
    b"\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"
)

# SAM normalisation constants
_SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
_SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
_SAM_INPUT_SIZE = 1024


# ---------------------------------------------------------------------------
# Positional encoding (matches original VLSAM training code)
# ---------------------------------------------------------------------------
class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


# ---------------------------------------------------------------------------
# VLSAM model (SAM2 backbone + BLIP/Mamba-derived prompts)
# ---------------------------------------------------------------------------
class VLSAM(nn.Module):
    """SAM2 image encoder + mask decoder driven by BLIP/Mamba embeddings."""

    def __init__(self, image_encoder, mask_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.pe_layer = PositionEmbeddingRandom(256 // 2)
        self.pseudo_mask_embed = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor, text_embeddings: torch.Tensor,
                image_features: torch.Tensor) -> torch.Tensor:
        # Drop BLIP CLS token → (1, 576, 1024) = 24×24 spatial tokens
        image_features = image_features[:, 1:, :]

        # SAM2 image encoder
        out = self.image_encoder(image)
        if isinstance(out, dict):
            image_embedding = out["vision_features"]   # (B, 256, 64, 64)
            bfpn = out["backbone_fpn"]
            feat_s0 = self.mask_decoder.conv_s0(bfpn[0])
            feat_s1 = self.mask_decoder.conv_s1(bfpn[1])
            high_res_features = (feat_s0, feat_s1)
        else:
            image_embedding = out
            high_res_features = None

        # Sparse prompts: concat Mamba text tokens + BLIP image tokens
        mamba_text = text_embeddings.view(1, -1, 256)
        blip_img = image_features.view(1, -1, 256)
        sparse_embeddings = torch.cat((mamba_text, blip_img), dim=1)

        # Dense prompt: learned projection of image embedding
        dense_embeddings = self.pseudo_mask_embed(image_embedding)

        low_res_masks, _iou, _tok, _obj = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.pe_layer((64, 64)).unsqueeze(0),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # Upsample to original image size
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks  # (1, 1, H, W)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _preprocess_image_for_sam(pil_image: Image.Image,
                               device: torch.device) -> torch.Tensor:
    """Return (1, 3, 1024, 1024) SAM-normalised tensor."""
    img = pil_image.resize((_SAM_INPUT_SIZE, _SAM_INPUT_SIZE), Image.BILINEAR)
    img_t = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)
    img_t = (img_t - _SAM_PIXEL_MEAN) / _SAM_PIXEL_STD
    return img_t.unsqueeze(0).to(device)


def _get_vlm_features(pil_image, vlm_model, processor, mamba_model, tokenizer,
                       device):
    """Run BLIP + Mamba on a single PIL image; return (text_emb, img_feat)."""
    blip_inputs = processor(pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_out = vlm_model.vision_model(
            pixel_values=blip_inputs["pixel_values"], return_dict=True
        )
        image_features = vision_out.last_hidden_state       # (1, 577, 1024)

        caption_ids = vlm_model.generate(**blip_inputs, max_new_tokens=30)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)

        mamba_inputs = tokenizer(caption, return_tensors="pt").to(device)
        mamba_out = mamba_model(**mamba_inputs)
        text_embeddings = mamba_out.last_hidden_state        # (1, seq_len, D)

    return text_embeddings, image_features


def _save_ann_png(path: str, mask: np.ndarray, palette: bytes) -> None:
    assert mask.dtype == np.uint8 and mask.ndim == 2
    img = Image.fromarray(mask)
    img.putpalette(palette)
    img.save(path)


def _sorted_frame_names(video_dir: str):
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in exts
    ]
    names.sort(key=lambda n: int(n))
    return names


# ---------------------------------------------------------------------------
# Per-video VLSAM-VOS inference
# ---------------------------------------------------------------------------
def vlsam_vos_inference(
    vlsam_model,
    vos_predictor,
    vlm_model,
    processor,
    mamba_model,
    tokenizer,
    base_video_dir: str,
    output_mask_dir: str,
    video_name: str,
    score_thresh: float = 0.0,
    seed_strategy: str = "all",   # "all" | "first"
    device=None,
):
    """
    Two-phase inference for one video:

    Phase 1 – run VLSAM on every frame → binary masks
    Phase 2 – seed SAM2 video predictor with those masks → propagate
    """
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = _sorted_frame_names(video_dir)
    out_dir = os.path.join(output_mask_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Phase 1: per-frame VLSAM masks ────────────────────────────────────
    print(f"  Phase 1 – VLSAM scoring {len(frame_names)} frames …")
    vlsam_masks = {}   # frame_idx → np.bool_ array (H, W) at original resolution

    vlsam_model.eval()
    for frame_idx, frame_name in enumerate(tqdm(frame_names, desc="VLSAM", leave=False)):
        frame_path = os.path.join(video_dir, frame_name + ".jpg")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(video_dir, frame_name + ".jpeg")

        pil_image = Image.open(frame_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        text_emb, img_feat = _get_vlm_features(
            pil_image, vlm_model, processor, mamba_model, tokenizer, device
        )
        image_tensor = _preprocess_image_for_sam(pil_image, device)

        with torch.no_grad():
            pred = vlsam_model(image_tensor, text_emb, img_feat)  # (1,1,1024,1024)

        # Resize logit mask back to original frame resolution
        pred_orig = F.interpolate(
            pred,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        binary = (pred_orig[0, 0] > score_thresh).cpu().numpy()   # bool (H, W)
        vlsam_masks[frame_idx] = binary

    # ── Phase 2: SAM2 VOS propagation ─────────────────────────────────────
    print(f"  Phase 2 – SAM2 VOS propagation …")

    inference_state = vos_predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    # Determine which frames to use as seeds
    if seed_strategy == "first":
        seed_frame_idxs = [0]
    else:  # "all" – seed every frame with the VLSAM prediction
        seed_frame_idxs = list(range(len(frame_names)))

    obj_id = 1  # single foreground object
    for frame_idx in seed_frame_idxs:
        mask = vlsam_masks[frame_idx]
        # Resize to video predictor's internal resolution if necessary
        if mask.shape != (height, width):
            mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=(height, width), mode="nearest")
            mask = mask_t[0, 0].numpy().astype(bool)

        vos_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask,
        )

    # Propagate and collect per-frame results
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in vos_predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            oid: (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, oid in enumerate(out_obj_ids)
        }

    # Write output masks as palette PNGs
    for out_frame_idx, per_obj in video_segments.items():
        # Combine per-object masks into a single label map (object=1, background=0)
        combined = np.zeros((height, width), dtype=np.uint8)
        for oid in sorted(per_obj, reverse=True):
            combined[per_obj[oid].reshape(height, width)] = oid

        frame_name = frame_names[out_frame_idx]
        _save_ann_png(
            os.path.join(out_dir, f"{frame_name}.png"),
            combined,
            DAVIS_PALETTE,
        )

    print(f"  [OK] {len(video_segments)} frames → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "VLSAM-VOS: SAM2 video-object segmentation with BLIP+Mamba prompts.\n"
            "Phase 1 – VLSAM scores every frame (BLIP + Mamba → mask).\n"
            "Phase 2 – SAM2 video predictor propagates those masks temporally."
        )
    )
    # SAM2 model
    parser.add_argument("--sam2_cfg", type=str, required=True,
                        help="SAM2 config file (e.g. configs/sam2.1/sam2.1_hiera_l.yaml)")
    parser.add_argument("--sam2_checkpoint", type=str, required=True,
                        help="SAM2 checkpoint (e.g. sam2.1_hiera_large.pt)")
    # Optional VLSAM fine-tuned weights
    parser.add_argument("--vlsam_checkpoint", type=str, default=None,
                        help="Optional VLSAM checkpoint (.pth with 'model' key)")
    # Data
    parser.add_argument("--base_video_dir", type=str, required=True,
                        help="Root dir with per-video JPEG frame folders")
    parser.add_argument("--output_mask_dir", type=str, required=True,
                        help="Root dir where per-video PNG masks are saved")
    parser.add_argument("--video_list_file", type=str, default=None,
                        help="Text file listing video names (one per line); "
                             "defaults to all subdirs in base_video_dir")
    # Inference options
    parser.add_argument("--score_thresh", type=float, default=0.0,
                        help="Logit threshold for mask binarisation (default: 0.0)")
    parser.add_argument("--seed_strategy", type=str, default="all",
                        choices=["all", "first"],
                        help="'all'  – seed SAM2 with VLSAM mask at every frame "
                             "(dense conditioning, default); "
                             "'first' – seed only frame 0 then propagate freely")
    parser.add_argument("--apply_postprocessing", action="store_true",
                        help="Apply SAM2 post-processing (hole filling etc.)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (default: cuda)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build VLSAM (SAM2 backbone + BLIP/Mamba prompts) ──────────────────
    print("Loading SAM2 model for VLSAM …")
    sam2_model = build_sam2(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=device,
    )
    vlsam_model = VLSAM(
        image_encoder=sam2_model.image_encoder,
        mask_decoder=sam2_model.sam_mask_decoder,
    ).to(device)
    if args.vlsam_checkpoint is not None:
        ckpt = torch.load(args.vlsam_checkpoint, map_location=device, weights_only=False)
        res = vlsam_model.load_state_dict(ckpt["model"], strict=False)
        if res.missing_keys:
            print(f"  [VLSAM ckpt] missing keys (will keep current init): {res.missing_keys}")
        if res.unexpected_keys:
            print(f"  [VLSAM ckpt] unexpected keys (ignored): {res.unexpected_keys}")
        print(f"Loaded VLSAM checkpoint: {args.vlsam_checkpoint}")
    vlsam_model.eval()

    # ── Build SAM2 video predictor (for VOS propagation) ──────────────────
    print("Loading SAM2 video predictor …")
    vos_predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        device=device,
    )

    # Sync weights from the VLSAM model (which has the VLSAM checkpoint applied)
    # into the video predictor, so both models use identical trained weights.
    #
    # image_encoder: carries backbone + adapter weights from VLSAM checkpoint.
    # sam_mask_decoder: may have been fine-tuned with --train_mask_decoder;
    #   the predictor only loaded the SAM2 checkpoint so it would otherwise
    #   have stale weights.
    if args.vlsam_checkpoint is not None:
        try:
            vos_predictor.image_encoder.load_state_dict(
                vlsam_model.image_encoder.state_dict(), strict=False
            )
            print("Synchronized image_encoder weights to SAM2 video predictor")
        except Exception as e:
            print(f"Warning: could not sync image_encoder to video predictor: {e}")

        try:
            vos_predictor.sam_mask_decoder.load_state_dict(
                vlsam_model.mask_decoder.state_dict(), strict=False
            )
            print("Synchronized sam_mask_decoder weights to SAM2 video predictor")
        except Exception as e:
            print(f"Warning: could not sync sam_mask_decoder to video predictor: {e}")

    # ── Vision–language models ─────────────────────────────────────────────
    print("Loading BLIP …")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    vlm_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device).eval()

    print("Loading Mamba …")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    mamba_model = MambaModel.from_pretrained("state-spaces/mamba-130m-hf").to(device).eval()

    # ── Video list ─────────────────────────────────────────────────────────
    if args.video_list_file is not None:
        with open(args.video_list_file) as f:
            video_names = [v.strip() for v in f if v.strip()]
    else:
        video_names = sorted(
            p for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        )
    print(f"\nRunning VLSAM-VOS on {len(video_names)} video(s): {video_names}")
    print(f"Seed strategy: '{args.seed_strategy}'\n")

    # ── Per-video inference ────────────────────────────────────────────────
    for n, video_name in enumerate(video_names):
        print(f"[{n + 1}/{len(video_names)}] {video_name}")
        vlsam_vos_inference(
            vlsam_model=vlsam_model,
            vos_predictor=vos_predictor,
            vlm_model=vlm_model,
            processor=processor,
            mamba_model=mamba_model,
            tokenizer=tokenizer,
            base_video_dir=args.base_video_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            score_thresh=args.score_thresh,
            seed_strategy=args.seed_strategy,
            device=device,
        )

    print(f"\nDone. Masks saved to: {args.output_mask_dir}")


if __name__ == "__main__":
    main()
