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
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

import torch

# ── repo root on PYTHONPATH so `sam2` and `VLSAM_fine-tuning` are importable ─
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "VLSAM_fine-tuning"))

from sam2.build_sam import build_sam2_video_predictor
from bridge_2 import DINOv3SAM2Bridge, add_saliency_to_sam2


# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def _saliency_to_binary(saliency: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert (1,1,H,W) logit saliency map to a boolean (H,W) numpy mask."""
    return (torch.sigmoid(saliency)[0, 0].cpu().numpy() > threshold)


def _mask_points(
    binary_mask: np.ndarray,
    n: int = 1,
) -> np.ndarray | None:
    """
    Return (n, 2) array of (x, y) foreground points sampled from binary_mask.

    Strategy
    --------
    1. Compute the centroid of all foreground pixels.
    2. If the centroid itself is NOT a foreground pixel (common for concave /
       curved shapes like snakes), snap it to the nearest foreground pixel.
    3. If n > 1, additionally sample (n-1) evenly-spaced foreground pixels
       across the sorted list of all fg pixels (good for elongated objects).

    Returns None if the mask is empty.
    """
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return None

    fg_pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (M, 2)

    # --- centroid, snapped to nearest fg pixel if not inside mask ---
    cx, cy = float(xs.mean()), float(ys.mean())
    if not binary_mask[int(round(cy)), int(round(cx))]:
        # find closest foreground pixel to the centroid
        dists = np.sqrt((fg_pts[:, 0] - cx) ** 2 + (fg_pts[:, 1] - cy) ** 2)
        nearest = fg_pts[int(np.argmin(dists))]
        cx, cy = float(nearest[0]), float(nearest[1])

    if n == 1:
        return np.array([[cx, cy]], dtype=np.float32)

    # --- sample n-1 additional interior points using distance transform ---
    # distance_transform_edt gives each fg pixel its distance to the nearest
    # background pixel; sampling from high-distance pixels keeps points well
    # inside the object and away from edges.
    dist = distance_transform_edt(binary_mask)       # (H, W) float
    dist_vals = dist[ys, xs]                         # distance for each fg pixel
    # sort by distance descending (most-interior first), then pick evenly-spaced
    order = np.argsort(dist_vals)[::-1]              # indices into fg_pts
    interior_pts = fg_pts[order]                     # most-interior first
    # cap pool to the top half so linspace never reaches edge pixels
    pool_size = max(n - 1, len(interior_pts) // 2)
    pool = interior_pts[:pool_size]
    indices = np.linspace(0, len(pool) - 1, n, dtype=int)[:-1]
    extra = pool[indices]                            # (n-1, 2)
    return np.vstack([[[cx, cy]], extra]).astype(np.float32)  # (n, 2)


def prompt_sam2(
    predictor,
    inference_state,
    frame_idx: int,
    obj_id: int,
    saliency: torch.Tensor,
    mode: str = "both",          # "mask" | "centroid" | "both"
    sal_threshold: float = 0.5,
    num_points: int = 1,
) -> None:
    """
    Register a saliency prompt with SAM2 using the chosen prompt mode.

    Modes
    -----
    mask      – dense binary mask via add_new_mask
    centroid  – foreground point(s) via add_new_points_or_box
                (centroid snapped to nearest fg pixel; n extra evenly-spaced
                 points added if num_points > 1 — useful for elongated objects)
    both      – centroid point(s) first, then dense mask (recommended)
    """
    binary = _saliency_to_binary(saliency, sal_threshold)

    if mode in ("centroid", "both"):
        pts = _mask_points(binary, n=num_points)
        if pts is not None:
            labels = np.ones(len(pts), dtype=np.int32)   # all foreground
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=pts,
                labels=labels,
            )

    if mode in ("mask", "both"):
        add_saliency_to_sam2(
            predictor, inference_state, frame_idx, obj_id, saliency
        )


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
    redetect_every: int | None = None,
    obj_id: int = 1,
    save_overlay: bool = False,
    prompt_mode: str = "both",
    num_points: int = 1,
    min_fg_pixels: int = 1,
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

    print(f"[bridge] Scanning frames for seed (min_fg_pixels={min_fg_pixels}) …")
    for fi in range(N):
        sal, (H, W) = compute_saliency(bridge, frame_paths[fi], device)
        fg = saliency_fg_count(sal)
        print(f"  frame {fi:05d}: {fg} foreground pixels")
        if fg >= min_fg_pixels:
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
                prompt_sam2(
                    predictor, inference_state, next_stop, obj_id, sal,
                    mode=prompt_mode, num_points=num_points,
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

    # Prompt mode
    p.add_argument("-prompt_mode", default="both",
                   choices=["mask", "centroid", "both"],
                   help="How to prompt SAM2: "
                        "'mask' = dense binary mask only; "
                        "'centroid' = foreground point(s) only; "
                        "'both' = point(s) + dense mask (recommended).")
    p.add_argument("-num_points", type=int, default=1,
                   help="Number of foreground points to pass to SAM2 when "
                        "prompt_mode is 'centroid' or 'both'. "
                        "1 = snapped centroid only; >1 also adds evenly-spaced "
                        "fg pixels (useful for elongated objects like snakes).")

    # Misc
    p.add_argument("-save_overlay", action="store_true", default=False,
                   help="Save colour-blended overlay images")
    p.add_argument("-min_fg_pixels", type=int, default=1,
                   help="Minimum foreground pixel count for a frame to be "
                        "accepted as the seed (default 1; increase to skip "
                        "small/noisy detections, e.g. 500).")
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
                prompt_mode=args.prompt_mode,
                num_points=args.num_points,
                min_fg_pixels=args.min_fg_pixels,
            )
        except Exception as exc:
            print(f"[error] {vid_name}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\nDone. Results saved under {out_root}")


if __name__ == "__main__":
    main()
