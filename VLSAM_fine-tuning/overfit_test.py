"""
overfit_test.py
===============
Sanity-check script: pick N samples from the training data, overfit on them,
and save visualisations every few epochs so you can see whether the model
is learning anything at all.

If it cannot memorise 5 frames, something is structurally wrong.

Usage
-----
  python VLSAM_fine-tuning/overfit_test.py \
      --data_root /Experiments/marcol01 \
      --image_size 736 1280 \
      --n_samples 5 \
      --epochs 200 \
      --lr 1e-3 \
      --device cuda:0 \
      --out_dir overfit_viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dinov3_sam2_bridge import DINOv3SAM2Bridge


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  required=True)
    p.add_argument("--image_size", type=int, nargs=2, default=[736, 1280], metavar=("H", "W"))
    p.add_argument("--n_samples",  type=int, default=5,   help="How many samples to overfit")
    p.add_argument("--epochs",     type=int, default=200)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--save_every", type=int, default=20,  help="Save visualisation every N epochs")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",    default="overfit_viz")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTENSIONS  = {".png", ".jpg", ".jpeg"}


def collect_pairs(data_root: Path) -> list[tuple[Path, Path]]:
    frames_root = data_root / "frames_train"
    masks_root  = data_root / "masks_train"
    pairs = []
    for video_dir in sorted(frames_root.iterdir()):
        if not video_dir.is_dir():
            continue
        mask_dir = masks_root / video_dir.name
        if not mask_dir.exists():
            continue
        for frame_path in sorted(video_dir.iterdir()):
            if frame_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            for ext in MASK_EXTENSIONS:
                cand = mask_dir / (frame_path.stem + ext)
                if cand.exists():
                    pairs.append((frame_path, cand))
                    break
    return pairs


def load_sample(
    frame_path: Path,
    mask_path: Path,
    bridge: DINOv3SAM2Bridge,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """Returns (pixel_values (1,3,H,W), mask (1,H,W), orig PIL frame)."""
    pil_frame = Image.open(frame_path).convert("RGB")
    pil_mask  = Image.open(mask_path).convert("L")

    pixel_values = bridge.extractor.preprocess(pil_frame, device=device, size=image_size)

    pil_mask = pil_mask.resize((image_size[1], image_size[0]), Image.NEAREST)
    mask_np  = np.array(pil_mask, dtype=np.float32)
    mask_t   = torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0).to(device)  # (1,H,W)

    return pixel_values, mask_t, pil_frame


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def make_overlay(pil_img: Image.Image, mask_np: np.ndarray, colour: tuple) -> Image.Image:
    """Blend a binary mask as a semi-transparent colour overlay on the image."""
    img_resized = pil_img.resize((mask_np.shape[1], mask_np.shape[0]), Image.BILINEAR)
    out = img_resized.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    pixels = overlay.load()
    for y in range(mask_np.shape[0]):
        for x in range(mask_np.shape[1]):
            if mask_np[y, x]:
                pixels[x, y] = (*colour, 140)
    return Image.alpha_composite(out, overlay).convert("RGB")


def save_visualisation(
    samples: list[tuple[torch.Tensor, torch.Tensor, Image.Image]],
    bridge: DINOv3SAM2Bridge,
    image_size: tuple[int, int],
    out_path: Path,
    epoch: int,
    threshold: float = 0.5,
) -> None:
    bridge.eval()
    panels = []
    with torch.no_grad():
        for pixel_values, gt_mask, pil_orig in samples:
            logits = bridge(pixel_values, target_size=image_size)  # (1,1,H,W)
            pred_mask = (torch.sigmoid(logits[0, 0]) > threshold).cpu().numpy().astype(np.uint8)
            gt_np     = gt_mask[0].cpu().numpy().astype(np.uint8)

            orig_small = pil_orig.resize((image_size[1] // 2, image_size[0] // 2), Image.BILINEAR)
            gt_vis     = make_overlay(pil_orig, gt_np,   (0, 255, 0))
            gt_vis     = gt_vis.resize((image_size[1] // 2, image_size[0] // 2), Image.BILINEAR)
            pred_vis   = make_overlay(pil_orig, pred_mask, (255, 60, 60))
            pred_vis   = pred_vis.resize((image_size[1] // 2, image_size[0] // 2), Image.BILINEAR)

            # Stitch [orig | GT | pred] horizontally
            row = Image.new("RGB", (orig_small.width * 3, orig_small.height))
            row.paste(orig_small, (0, 0))
            row.paste(gt_vis,     (orig_small.width, 0))
            row.paste(pred_vis,   (orig_small.width * 2, 0))
            panels.append(row)

    # Stack samples vertically
    total_h = sum(p.height for p in panels)
    canvas  = Image.new("RGB", (panels[0].width, total_h), (30, 30, 30))
    y_off = 0
    for p in panels:
        canvas.paste(p, (0, y_off))
        y_off += p.height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"  Saved viz → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

bce_fn = nn.BCEWithLogitsLoss()

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs   = torch.sigmoid(logits).reshape(logits.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    inter   = (probs * targets).sum(dim=1)
    union   = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()

def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return bce_fn(logits, targets) + dice_loss(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = get_args()
    device     = torch.device(args.device)
    image_size = tuple(args.image_size)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading DINOv3SAM2Bridge …")
    bridge = DINOv3SAM2Bridge(freeze_backbone=True, device=device)

    patch = bridge.extractor.patch_size
    if image_size[0] % patch != 0 or image_size[1] % patch != 0:
        raise ValueError(
            f"image_size {image_size} must be multiples of {patch}. "
            f"Try {(round(image_size[0]/patch)*patch, round(image_size[1]/patch)*patch)}."
        )

    n_params = sum(p.numel() for p in bridge.bridge.parameters())
    print(f"Trainable params: {n_params:,}")

    # ── Collect N samples ─────────────────────────────────────────────────
    print(f"Collecting {args.n_samples} samples from {args.data_root} …")
    all_pairs = collect_pairs(Path(args.data_root))
    if len(all_pairs) < args.n_samples:
        raise RuntimeError(f"Only {len(all_pairs)} pairs found, need {args.n_samples}.")

    # Space them across the dataset so we get variety (not just the first video)
    step = max(1, len(all_pairs) // args.n_samples)
    chosen = [all_pairs[i * step] for i in range(args.n_samples)]

    samples = []
    for fp, mp in chosen:
        pv, mk, pil = load_sample(fp, mp, bridge, image_size, device)
        samples.append((pv, mk, pil))
        print(f"  {fp.name}  mask_fg={mk.mean():.3f}  pixel_values={tuple(pv.shape)}")

    # ── Overfit loop ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(bridge.bridge.parameters(), lr=args.lr, weight_decay=0.0)

    print(f"\nOverfitting {args.n_samples} samples for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        bridge.train()
        total_loss = 0.0
        for pixel_values, gt_mask, _ in samples:
            logits = bridge(pixel_values, target_size=image_size)  # (1,1,H,W)
            loss   = criterion(logits.squeeze(1), gt_mask)         # (1,H,W)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(samples)
        print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}")

        if epoch % args.save_every == 0 or epoch == 1 or epoch == args.epochs:
            save_visualisation(
                samples, bridge, image_size,
                out_dir / f"epoch_{epoch:04d}.png",
                epoch=epoch,
            )

    print(f"\nDone. Check {out_dir}/ for visualisations.")
    print("If loss went to ~0 and predictions match GT, the pipeline is correct.")
    print("If loss stalls or predictions are random, something structural is still wrong.")


if __name__ == "__main__":
    main()
