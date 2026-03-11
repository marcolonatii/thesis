#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_bridge_batch.py
=========================
For each video folder under a frames root, run the trained DINOv3->SaliencyBridge
on the first N frames (default 5) and save side-by-side visualisations:

  [ original | GT overlay (green) | predicted overlay (red) ]

Usage:
  python tools/visualize_bridge_batch.py \
      -frames_root /Experiments/marcol01/frames \
      -masks_root  /Experiments/marcol01/masks \
      -out_dir     viz_bridge_batch \
      -bridge_ckpt VLSAM_fine-tuning/bridge_ckpts/bridge_best.pt \
      --device     cuda:0

Options:
  -frames_per_video : number of frames to process per video (default 5)
  -max_videos       : limit number of videos processed
  -threshold        : logit threshold for binarising bridge output (default 0.0)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "VLSAM_fine-tuning"))

from dinov3_sam2_bridge import DINOv3SAM2Bridge

_IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
_MASK_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def sorted_frame_paths(folder: Path) -> List[Path]:
    paths = [f for f in folder.iterdir() if f.suffix in _IMG_EXTS and f.is_file()]
    paths.sort(key=lambda f: int(f.stem) if f.stem.isdigit() else f.stem)
    return paths


def find_mask(masks_dir: Path, stem: str) -> Path | None:
    for ext in _MASK_EXTS:
        cand = masks_dir / (stem + ext)
        if cand.exists():
            return cand
    return None


def overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray,
                 color: tuple, alpha: float = 0.5) -> np.ndarray:
    out = frame_rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def add_label(img: np.ndarray, text: str, font_scale: float = 0.8,
              color=(255, 255, 255), bg=(0, 0, 0)) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((4, 4), text, font=font)
    draw.rectangle(bbox, fill=bg)
    draw.text((4, 4), text, fill=color, font=font)
    return np.array(pil)


def make_panel(
    frame_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    frame_name: str,
) -> np.ndarray:
    frame_panel  = add_label(frame_rgb.copy(),               f"{frame_name}\noriginal")
    gt_panel     = add_label(overlay_mask(frame_rgb, gt_mask,   (0, 220,  0)),  "GT mask")
    pred_panel   = add_label(overlay_mask(frame_rgb, pred_mask, (220,  0,  0)), "predicted")
    return np.concatenate([frame_panel, gt_panel, pred_panel], axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch visualise bridge predictions (first N frames per video)")
    p.add_argument("-frames_root", required=True, help="Root folder containing video subfolders of frames")
    p.add_argument("-masks_root",  required=True, help="Root folder containing mask subfolders matching video names")
    p.add_argument("-out_dir",     default="viz_bridge_batch", help="Output root folder for visualisations")
    p.add_argument("-bridge_ckpt", default=None, help="Path to bridge checkpoint (optional)")
    p.add_argument("-frames_per_video", type=int, default=5, help="Number of frames to process per video")
    p.add_argument("-max_videos", type=int, default=None, help="Process at most this many videos")
    p.add_argument("-threshold",   type=float, default=0.0, help="Logit threshold for binarising the bridge output")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    frames_root = Path(args.frames_root)
    masks_root  = Path(args.masks_root)
    out_root    = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[init] Loading DINOv3SAM2Bridge …")
    bridge = DINOv3SAM2Bridge(freeze_backbone=True, device=device)
    bridge.eval()

    if args.bridge_ckpt:
        ckpt = Path(args.bridge_ckpt)
        if ckpt.exists():
            state = torch.load(str(ckpt), map_location=device)
            bridge.bridge.load_state_dict(state["bridge_state_dict"])
            print(f"[init] Loaded weights from {ckpt}")
        else:
            print(f"[warn] Checkpoint not found: {ckpt}. Using random weights.")
    else:
        print("[warn] No -bridge_ckpt; using random weights.")

    # Collect video subfolders
    video_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
    video_dirs.sort(key=lambda p: p.name)
    if args.max_videos:
        video_dirs = video_dirs[: args.max_videos]

    print(f"[info] Found {len(video_dirs)} video folders; processing first {args.frames_per_video} frames each.\n")

    for video_dir in tqdm(video_dirs, desc="videos"):
        video_name = video_dir.name
        masks_dir = masks_root / video_name
        out_dir = out_root / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted_frame_paths(video_dir)[: args.frames_per_video]
        if len(frame_paths) == 0:
            print(f"[skip] No frames in {video_dir}")
            continue

        pairs = []
        for fp in frame_paths:
            mp = find_mask(masks_dir, fp.stem)
            if mp is not None:
                pairs.append((fp, mp))
            else:
                print(f"[skip] No mask for {video_name}/{fp.name}")

        if len(pairs) == 0:
            print(f"[warn] No valid frame/mask pairs for {video_name}, skipping.")
            continue

        for frame_path, mask_path in pairs:
            frame_pil = Image.open(frame_path).convert("RGB")
            H, W = frame_pil.height, frame_pil.width
            frame_rgb = np.array(frame_pil)

            gt_pil = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
            gt_mask = (np.array(gt_pil) > 127)

            with torch.no_grad():
                pixel_values = bridge.extractor.preprocess(frame_pil, device=device)
                logits = bridge(pixel_values, target_size=(H, W))
            pred_mask = (logits[0, 0].cpu().numpy() > args.threshold)

            panel = make_panel(frame_rgb, gt_mask, pred_mask, frame_path.stem)
            out_path = out_dir / f"{frame_path.stem}.jpg"
            Image.fromarray(panel).save(str(out_path), quality=92)

    print(f"\nDone. Visualisations saved to: {out_root}")


if __name__ == "__main__":
    main()