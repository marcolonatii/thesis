#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_bridge.py
===================
Run the trained DINOv3->SaliencyBridge on every frame of each video and
save binary prediction masks (0/255 PNG) organised as:

  <out_dir>/
    <video_name>/
      00000.png
      00001.png
      ...

Usage
-----
  python tools/inference_bridge.py \\
      -frames_root /Experiments/marcol01/frames \\
      -out_dir     /home/marcol01/pred_masks_bridge \\
      -bridge_ckpt VLSAM_fine-tuning/checkpoints_cod10k/bridge_epoch013.pt \\
      --device     cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "VLSAM_fine-tuning"))

from bridge_2 import DINOv3SAM2Bridge

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def sorted_frame_paths(folder: Path) -> List[Path]:
    paths = [f for f in folder.iterdir() if f.suffix in _IMG_EXTS and f.is_file()]
    paths.sort(key=lambda f: int(f.stem) if f.stem.isdigit() else f.stem)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DINOv3+SaliencyBridge on all video frames and save binary masks."
    )
    p.add_argument("-frames_root", required=True,
                   help="Root folder containing per-video subfolders of frames")
    p.add_argument("-out_dir",     required=True,
                   help="Root output directory for predicted masks")
    p.add_argument("-bridge_ckpt", default=None,
                   help="Path to trained SaliencyBridge checkpoint")
    p.add_argument("-threshold",   type=float, default=0.0,
                   help="Logit threshold for binarising bridge output (default 0.0)")
    p.add_argument("-max_videos",  type=int,   default=None,
                   help="Process at most this many videos (default: all)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def run_bridge(bridge: DINOv3SAM2Bridge, frame_path: Path,
               device: torch.device, threshold: float) -> np.ndarray:
    """Return a binary (H, W) uint8 mask (0 or 255) for one frame."""
    pil_img = Image.open(frame_path).convert("RGB")
    H, W = pil_img.height, pil_img.width
    pixel_values = bridge.extractor.preprocess(pil_img, device=device)
    logits = bridge(pixel_values, target_size=(H, W))   # (1,1,H,W)
    binary = (logits[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
    return binary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    frames_root = Path(args.frames_root)
    out_root    = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not frames_root.is_dir():
        raise FileNotFoundError(f"frames_root not found: {frames_root}")

    # ── Load bridge ────────────────────────────────────────────────────────
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
        print("[warn] No -bridge_ckpt provided; using random weights.")

    # ── Collect videos ─────────────────────────────────────────────────────
    video_dirs = sorted(d for d in frames_root.iterdir() if d.is_dir())
    if args.max_videos:
        video_dirs = video_dirs[: args.max_videos]

    if not video_dirs:
        raise FileNotFoundError(f"No subdirectories found in {frames_root}")

    print(f"[info] {len(video_dirs)} video(s) to process.\n")

    # ── Process ────────────────────────────────────────────────────────────
    for video_dir in tqdm(video_dirs, desc="videos"):
        frame_paths = sorted_frame_paths(video_dir)
        if not frame_paths:
            print(f"[skip] No frames in {video_dir.name}")
            continue

        out_dir = out_root / video_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for fp in tqdm(frame_paths, desc=video_dir.name, leave=False):
            mask = run_bridge(bridge, fp, device, args.threshold)
            Image.fromarray(mask).save(str(out_dir / f"{fp.stem}.png"))

        print(f"[OK] {video_dir.name}: {len(frame_paths)} frames -> {out_dir}")

    print(f"\nDone. Masks saved under {out_root}")


if __name__ == "__main__":
    main()
