#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vos_inference_bridge.py
=======================
VOS inference using DINOv3 + SaliencyBridge as a dense SAM2 prompt.

Pipeline
--------
1. SALIENCY PROMPT
   Run the trained DINOv3→SaliencyBridge on frame 0; register the
   resulting (H, W) saliency logit map with SAM2 via `add_new_mask`.

2. PROPAGATION
   Call `predictor.propagate_in_video` forward.
   Monitor each frame for failure signals (mask area drop, IoU drop).

3. RE-DETECT ON FAILURE
   On a detected failure, run the bridge again on that frame, re-register
   with SAM2, and resume propagation from there.

Usage
-----
  python tools/vos_inference_bridge.py \\
      -video /Experiments/marcol01/frames/myvideo \\
      -out_dir vos_bridge_output \\
      -bridge_ckpt VLSAM_fine-tuning/bridge_ckpts/bridge_best.pt \\
      -sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \\
      -checkpoint sam2.1_hiera_large.pt \\
      --device cuda:1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

# ── repo root on PYTHONPATH so `sam2` and `VLSAM_fine-tuning` are importable ─
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "VLSAM_fine-tuning"))

from sam2.build_sam import build_sam2_video_predictor
from bridge_2 import DINOv3SAM2Bridge, add_saliency_to_sam2


# ─────────────────────────────────────────────────────────────────────────────
# Frame I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def sorted_frame_paths(video_path: str | Path) -> list[Path]:
    """Return frame paths sorted numerically by stem."""
    p = Path(video_path)
    if not p.is_dir():
        raise FileNotFoundError(f"Not a directory: {p}")
    paths = [f for f in p.iterdir() if f.suffix in _IMG_EXTS]
    paths.sort(key=lambda f: int(f.stem) if f.stem.isdigit() else f.stem)
    return paths


def load_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(str(path))


def blend_overlay(frame_rgb: np.ndarray, mask: np.ndarray,
                  color=(0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    out = frame_rgb.copy()
    m = mask.astype(bool)
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Bridge inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_saliency(
    bridge: DINOv3SAM2Bridge,
    frame_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Run DINOv3 + SaliencyBridge on one frame.
    Returns (saliency (1,1,H,W), (H,W)).
    """
    pil_img = Image.open(frame_path).convert("RGB")
    H, W = pil_img.height, pil_img.width
    pixel_values = bridge.extractor.preprocess(pil_img, device=device)
    saliency = bridge(pixel_values, target_size=(H, W))   # (1,1,H,W)
    return saliency, (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Saliency foreground check
# ─────────────────────────────────────────────────────────────────────────────

def saliency_fg_count(saliency: torch.Tensor, threshold: float = 0.5) -> int:
    """Return number of foreground pixels (sigmoid(logits) > threshold)."""
    return int((torch.sigmoid(saliency) > threshold).sum().item())


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Core per-video pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    frame_paths: list[Path],
    predictor,
    bridge: DINOv3SAM2Bridge,
    out_dir: Path,
    device: torch.device,
    redetect_every: int | None = None,  # None → no periodic re-detection
    obj_id: int = 1,
    save_overlay: bool = False,
) -> dict[int, np.ndarray]:
    """
    Full pipeline for a single video.  Returns {frame_idx: bool mask (H,W)}.
    """
    if not frame_paths:
        print("[warn] No frames found.")
        return {}

    N = len(frame_paths)
    results: dict[int, np.ndarray] = {}

    # ── Step 1: scan frames until a saliency map with fg pixels is found ──
    bridge.eval()
    seed_frame_idx: int | None = None
    seed_saliency: torch.Tensor | None = None

    print("[bridge] Scanning frames for first non-empty saliency map …")
    for fi in range(N):
        sal, (H, W) = compute_saliency(bridge, frame_paths[fi], device)
        fg = saliency_fg_count(sal)
        print(f"  frame {fi:05d}: {fg} foreground pixels")
        if fg > 0:
            seed_frame_idx = fi
            seed_saliency  = sal
            break

    if seed_frame_idx is None:
        print("[warn] Bridge produced 0 foreground pixels on every frame. ")
        print("       Saving empty masks and skipping propagation.")
        out_dir.mkdir(parents=True, exist_ok=True)
        empty = np.zeros((H, W), dtype=bool)
        for fi in range(N):
            save_mask(empty, out_dir / f"{fi:05d}.png")
        return results

    print(f"[bridge] Seed frame: {seed_frame_idx} ")

    # ── Step 2: init SAM2 and register seed saliency ──────────────────────
    video_dir = str(frame_paths[0].parent)
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    add_saliency_to_sam2(
        predictor, inference_state, seed_frame_idx, obj_id, seed_saliency
    )

    # ── Step 3: backward propagation (seed → 0), only if seed > 0 ─────────
    if seed_frame_idx > 0:
        print(f"[propagate] Backward from frame {seed_frame_idx} to 0 …")
        for fidx, _oids, logits in tqdm(
            predictor.propagate_in_video(
                inference_state, start_frame_idx=seed_frame_idx, reverse=True
            ),
            total=seed_frame_idx + 1, desc="backward", leave=True,
        ):
            results[fidx] = (logits[0, 0].cpu().numpy() > 0.0)

    # ── Step 4: forward propagation (seed → end), optionally chunked ──────
    print(f"[propagate] Forward from frame {seed_frame_idx} to {N - 1} …")

    if redetect_every is None:
        # ── single pass, no re-detection ──────────────────────────────────
        for fidx, _oids, logits in tqdm(
            predictor.propagate_in_video(
                inference_state, start_frame_idx=seed_frame_idx, reverse=False
            ),
            total=N - seed_frame_idx, desc="forward", leave=True,
        ):
            results[fidx] = (logits[0, 0].cpu().numpy() > 0.0)
    else:
        # ── chunked pass: stop at each N-frame boundary, maybe re-prompt ──
        current_start = seed_frame_idx
        n_redetect = 0

        while current_start < N:
            next_stop = min(current_start + redetect_every, N - 1)
            chunk_len = next_stop - current_start + 1

            for fidx, _oids, logits in tqdm(
                predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=current_start,
                    reverse=False,
                ),
                total=chunk_len, desc=f"forward [{current_start}–{next_stop}]",
                leave=True,
            ):
                results[fidx] = (logits[0, 0].cpu().numpy() > 0.0)
                if fidx >= next_stop:
                    break  # stop at boundary; safe to break generators in Python

            current_start = next_stop + 1
            if current_start >= N:
                break

            # try re-detection at the boundary frame
            sal, _ = compute_saliency(bridge, frame_paths[next_stop], device)
            fg = saliency_fg_count(sal)
            if fg > 0:
                n_redetect += 1
                print(f"\n[redetect #{n_redetect}] frame {next_stop}: {fg} fg pixels → re-prompting SAM2")
                add_saliency_to_sam2(
                    predictor, inference_state, next_stop, obj_id, sal
                )
            else:
                print(f"\n[redetect] frame {next_stop}: 0 fg pixels → keeping SAM2 state")

    # ── Step 5: save outputs ───────────────────────────────────────────────
    print(f"[save] Writing {len(results)} masks to {out_dir} …")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fidx in sorted(results.keys()):
        mask = results[fidx]
        save_mask(mask, out_dir / f"{fidx:05d}.png")

        if save_overlay:
            frame_rgb = load_rgb(frame_paths[fidx])
            overlay   = blend_overlay(frame_rgb, mask)
            Image.fromarray(overlay).save(str(out_dir / f"{fidx:05d}_overlay.jpg"))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VOS inference: DINOv3 bridge → SAM2 dense prompt + propagation"
    )
    # I/O
    p.add_argument("-video",        required=True,
                   help="Path to a frame folder or a parent folder of videos")
    p.add_argument("-out_dir",      default="vos_bridge_output",
                   help="Root output directory")

    # SAM2
    p.add_argument("-sam2_cfg",     default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("-checkpoint",   required=True,
                   help="Path to SAM2 .pt checkpoint")

    # Bridge
    p.add_argument("-bridge_ckpt",  default=None,
                   help="Path to trained SaliencyBridge checkpoint.")

    # Periodic re-detection
    p.add_argument("-redetect_every", type=int, default=None,
                   help="Every N frames re-run the bridge; if > 0 fg pixels found, "
                        "re-prompt SAM2.  Omit to disable re-detection entirely.")

    # Misc
    p.add_argument("-save_overlay", action="store_true", default=False,
                   help="Save colour-blended overlay images")
    p.add_argument("--device",      default="cuda:0")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # ── Load SAM2 ──────────────────────────────────────────────────────────
    print("[init] Loading SAM2 …")
    predictor = build_sam2_video_predictor(
        args.sam2_cfg,
        args.checkpoint,
        device=device,
        mode="eval",
        apply_postprocessing=True,
    )

    # ── Load Bridge ────────────────────────────────────────────────────────
    print("[init] Loading DINOv3SAM2Bridge …")
    bridge = DINOv3SAM2Bridge(freeze_backbone=True, device=device)
    bridge.eval()

    if args.bridge_ckpt is not None:
        ckpt_path = Path(args.bridge_ckpt)
        if not ckpt_path.exists():
            print(f"[warn] Bridge checkpoint not found: {ckpt_path}. "
                  "Running with random weights.")
        else:
            state = torch.load(str(ckpt_path), map_location=device)
            bridge.bridge.load_state_dict(state["bridge_state_dict"])
            print(f"[init] Loaded bridge weights from {ckpt_path}")
    else:
        print("[warn] No -bridge_ckpt provided; using un-trained bridge weights.")

    # ── Determine input structure ──────────────────────────────────────────
    video_path = Path(args.video)
    out_root   = Path(args.out_dir)

    # Case A: video_path contains JPEG frames directly → single video
    # Case B: video_path is a parent of per-video subdirectories → batch
    frame_paths = sorted_frame_paths(video_path) if video_path.is_dir() else []
    is_single_video = len(frame_paths) > 0

    if is_single_video:
        videos = [(video_path.name, frame_paths)]
    else:
        # treat each sub-directory as a separate video
        subdirs = sorted(p for p in video_path.iterdir() if p.is_dir())
        if not subdirs:
            raise FileNotFoundError(
                f"No frames and no subdirectories found in {video_path}"
            )
        videos = [(s.name, sorted_frame_paths(s)) for s in subdirs]

    print(f"[init] Found {len(videos)} video(s) to process.\n")

    # ── Process each video ─────────────────────────────────────────────────
    for vid_name, fpaths in videos:
        if not fpaths:
            print(f"[skip] {vid_name}: no frames found.")
            continue
        print(f"{'='*60}")
        print(f"[video] {vid_name}  ({len(fpaths)} frames)")
        out_dir = out_root / vid_name

        try:
            process_video(
                frame_paths=fpaths,
                predictor=predictor,
                bridge=bridge,
                out_dir=out_dir,
                device=device,
                redetect_every=args.redetect_every,
                save_overlay=args.save_overlay,
            )
        except Exception as exc:
            print(f"[error] {vid_name}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\nDone. Results saved under {out_root}")


if __name__ == "__main__":
    main()
