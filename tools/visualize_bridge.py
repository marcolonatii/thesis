#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_bridge.py
===================
For each (frame, GT mask) pair, run the trained DINOv3→SaliencyBridge,
then save a side-by-side image:

  [  original frame  |  GT mask (green overlay)  |  predicted mask (red overlay)  ]

Usage
-----
  python tools/visualize_bridge.py \
      -frames   /Experiments/marcol01/frames/myvideo \
      -masks    /Experiments/marcol01/masks/myvideo \
      -out_dir  viz_bridge/myvideo \
      -bridge_ckpt VLSAM_fine-tuning/bridge_ckpts/bridge_best.pt \
      --device cuda:0

  # Limit to first 20 frames
  python tools/visualize_bridge.py ... -max_frames 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
import torch.nn.functional as F

# ── repo root so VLSAM_fine-tuning is importable ─────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "VLSAM_fine-tuning"))

from dinov3_sam2_bridge import DINOv3SAM2Bridge


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
_MASK_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def sorted_frame_paths(folder: Path) -> list[Path]:
    paths = [f for f in folder.iterdir() if f.suffix in _IMG_EXTS]
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
    """Blend a binary (bool/uint8) mask over an RGB frame."""
    out = frame_rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def add_label(img: np.ndarray, text: str, font_scale: float = 0.8,
              color=(255, 255, 255), bg=(0, 0, 0)) -> np.ndarray:
    """Burn a text label into the top-left corner of an image (numpy HxWx3)."""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    # background rectangle
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
    """
    Build a three-panel tile:
      [  original frame  |  GT overlay (green)  |  pred overlay (red)  ]
    """
    frame_panel  = add_label(frame_rgb.copy(),               f"{frame_name}\noriginal")
    gt_panel     = add_label(overlay_mask(frame_rgb, gt_mask,   (0, 220,  0)),  "GT mask")
    pred_panel   = add_label(overlay_mask(frame_rgb, pred_mask, (220,  0,  0)), "predicted")

    # Ensure all panels have the same height (they should, same frame source)
    return np.concatenate([frame_panel, gt_panel, pred_panel], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise DINOv3 bridge predictions vs GT masks"
    )
    p.add_argument("-frames",      required=True,
                   help="Folder of video frames (JPEG/PNG)")
    p.add_argument("-masks",       required=True,
                   help="Folder of ground-truth binary masks (matching stems)")
    p.add_argument("-out_dir",     default="viz_bridge",
                   help="Output folder for the visualisation images")
    p.add_argument("-bridge_ckpt", default=None,
                   help="Path to bridge_best.pt checkpoint. "
                        "If omitted, runs with random weights.")
    p.add_argument("-max_frames",  type=int, default=None,
                   help="Process at most this many frame/mask pairs")
    p.add_argument("-threshold",   type=float, default=0.0,
                   help="Logit threshold for binarising the bridge output (default 0.0)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    frames_dir = Path(args.frames)
    masks_dir  = Path(args.masks)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load bridge ───────────────────────────────────────────────────────
    print("[init] Loading DINOv3SAM2Bridge …")
    bridge = DINOv3SAM2Bridge(freeze_backbone=True, device=device)
    bridge.eval()

    if args.bridge_ckpt is not None:
        ckpt = Path(args.bridge_ckpt)
        if not ckpt.exists():
            print(f"[warn] Checkpoint not found: {ckpt}. Using random weights.")
        else:
            state = torch.load(str(ckpt), map_location=device)
            bridge.bridge.load_state_dict(state["bridge_state_dict"])
            print(f"[init] Loaded weights from {ckpt}")
    else:
        print("[warn] No -bridge_ckpt; using random weights.")

    # ── Collect frame / mask pairs ────────────────────────────────────────
    frame_paths = sorted_frame_paths(frames_dir)
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]

    pairs: list[tuple[Path, Path]] = []
    for fp in frame_paths:
        mp = find_mask(masks_dir, fp.stem)
        if mp is not None:
            pairs.append((fp, mp))
        else:
            print(f"[skip] No mask for {fp.name}")

    print(f"[info] Processing {len(pairs)} frame/mask pairs …\n")

    # ── Process each pair ─────────────────────────────────────────────────
    for frame_path, mask_path in tqdm(pairs, desc="visualising"):
        # Load frame
        frame_pil = Image.open(frame_path).convert("RGB")
        H, W      = frame_pil.height, frame_pil.width
        frame_rgb = np.array(frame_pil)

        # Load GT mask → binary (H, W) bool
        gt_pil  = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
        gt_mask = (np.array(gt_pil) > 127)

        # Run bridge → predicted mask
        with torch.no_grad():
            pixel_values = bridge.extractor.preprocess(frame_pil, device=device)
            logits       = bridge(pixel_values, target_size=(H, W))  # (1,1,H,W)
        pred_mask = (logits[0, 0].cpu().numpy() > args.threshold)

        # Build and save panel
        panel = make_panel(frame_rgb, gt_mask, pred_mask, frame_path.stem)
        out_path = out_dir / f"{frame_path.stem}.jpg"
        Image.fromarray(panel).save(str(out_path), quality=92)

    print(f"\nDone. Visualisations saved to: {out_dir}")


if __name__ == "__main__":
    main()
