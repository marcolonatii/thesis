"""
train_bridge_2.py
=================
Trains the bridge_2 architecture (DINOv3 + RGB encoder SaliencyBridge) to
predict binary foreground masks from DINOv3 patch features fused with raw RGB.

Identical to train_bridge.py except it imports DINOv3SAM2Bridge from bridge_2
instead of dinov3_sam2_bridge, and exposes the --dropout hyper-parameter.

Dataset layout expected
-----------------------
  <data_root>/
    frames_train/
      <video>/
        00000.jpg  ...
    masks_train/
      <video>/
        00000.png  ...   (white = foreground)

Usage
-----
  python VLSAM_fine-tuning/train_bridge_2.py \
      --data_root /Experiments/marcol01 \
      --image_size 736 1280 \
      --epochs 40 \
      --batch_size 4 \
      --lr 1e-4 \
      --dropout 0.1 \
      --device cuda:0 \
      --checkpoint_dir VLSAM_fine-tuning/bridge2_ckpts
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ── make sure the package is importable from any cwd ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_2 import DINOv3SAM2Bridge          # ← only difference vs train_bridge.py


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FrameMaskDataset(Dataset):
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    MASK_EXTENSIONS  = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        data_root: str | Path,
        image_size: tuple[int, int] = (448, 448),
    ) -> None:
        self.image_size = image_size

        frames_root = Path(data_root) / "frames_train"
        masks_root  = Path(data_root) / "masks_train"

        if not frames_root.exists():
            raise FileNotFoundError(f"frames directory not found: {frames_root}")
        if not masks_root.exists():
            raise FileNotFoundError(f"masks directory not found: {masks_root}")

        self.pairs: list[tuple[Path, Path]] = []

        for video_dir in sorted(frames_root.iterdir()):
            if not video_dir.is_dir():
                continue
            mask_dir = masks_root / video_dir.name
            if not mask_dir.exists():
                continue
            for frame_path in sorted(video_dir.iterdir()):
                if frame_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue
                mask_path = None
                for ext in self.MASK_EXTENSIONS:
                    cand = mask_dir / (frame_path.stem + ext)
                    if cand.exists():
                        mask_path = cand
                        break
                if mask_path is None:
                    continue
                self.pairs.append((frame_path, mask_path))

        if not self.pairs:
            raise RuntimeError(
                f"No (frame, mask) pairs found under {data_root}. "
                "Check that frames_train/ and masks_train/ exist and are non-empty."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Image.Image, torch.Tensor]:
        frame_path, mask_path = self.pairs[idx]

        frame = Image.open(frame_path).convert("RGB")

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(
            (self.image_size[1], self.image_size[0]), Image.NEAREST
        )
        mask_np = np.array(mask, dtype=np.float32)
        mask_tensor = torch.from_numpy((mask_np > 127).astype(np.float32))

        return frame, mask_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AugmentWrapper(Dataset):
    def __init__(self, subset: Dataset, aug_fn) -> None:
        self.subset = subset
        self.aug_fn = aug_fn

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        frame, mask = self.subset[idx]
        return self.aug_fn(frame, mask)


def collate_fn(bridge: DINOv3SAM2Bridge, device: torch.device, image_size: tuple[int, int]):
    def _collate(batch: list[tuple[Image.Image, torch.Tensor]]):
        frames, masks = zip(*batch)
        pixel_values = bridge.extractor.preprocess(list(frames), device=device, size=image_size)
        masks = torch.stack(masks, dim=0).to(device)
        return pixel_values, masks
    return _collate


# ─────────────────────────────────────────────────────────────────────────────
# Train / validation
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    bridge: DINOv3SAM2Bridge,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    image_size: tuple[int, int],
    train: bool,
) -> tuple[float, float, float]:
    """Returns (avg_loss, iou, precision) for the epoch."""
    bridge.train(train)
    context = torch.enable_grad() if train else torch.no_grad()
    total_loss = 0.0
    n_batches  = 0
    total_tp   = 0
    total_fp   = 0
    total_fn   = 0

    with context:
        for pixel_values, gt_masks in tqdm(loader, desc="train" if train else "val ", leave=False):
            logits = bridge(pixel_values, target_size=image_size)  # (B, 1, H, W)
            logits = logits.squeeze(1)                              # (B, H, W)

            loss = criterion(logits, gt_masks)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            with torch.no_grad():
                pred = (logits.detach() > 0).long()
                gt   = gt_masks.long()
                total_tp += int((pred & gt).sum())
                total_fp += int((pred & ~gt.bool()).sum())
                total_fn += int((~pred.bool() & gt).sum())

    avg_loss  = total_loss / max(n_batches, 1)
    iou       = total_tp / max(total_tp + total_fp + total_fn, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    return avg_loss, iou, precision


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train bridge_2 (DINOv3 + RGB encoder) SaliencyBridge"
    )
    p.add_argument("--data_root",       required=True)
    p.add_argument("--image_size",      type=int, nargs=2, default=[448, 448],
                   metavar=("H", "W"))
    p.add_argument("--val_split",       type=float, default=0.1)
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=5e-5)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--pos_weight",      type=float, default=None)
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint_dir",  default="./bridge2_ckpts")
    p.add_argument("--resume",          default=None)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--patience",        type=int,   default=5,
                   help="Early stopping patience on val_iou (epochs without improvement)")
    return p.parse_args()


def main() -> None:
    args = get_args()
    device     = torch.device(args.device)
    image_size = tuple(args.image_size)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────
    print("Loading DINOv3SAM2Bridge (bridge_2) …")
    bridge = DINOv3SAM2Bridge(freeze_backbone=True, device=device)

    # ── Validate image_size ────────────────────────────────────────────────
    patch = bridge.extractor.patch_size
    if image_size[0] % patch != 0 or image_size[1] % patch != 0:
        raise ValueError(
            f"image_size {image_size} must be multiples of {patch}. "
            f"Try {(round(image_size[0]/patch)*patch, round(image_size[1]/patch)*patch)}."
        )

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        bridge.bridge.load_state_dict(state["bridge_state_dict"])
        print(f"Resumed from {args.resume}")

    trainable_params = list(bridge.bridge.parameters())
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_params:,}  (SaliencyBridge only)")

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"Loading dataset from {args.data_root} …")
    full_dataset = FrameMaskDataset(args.data_root, image_size=image_size)
    print(f"  Total pairs: {len(full_dataset)}")

    n_val   = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {n_train}   Val: {n_val}")

    # ── Augmentations ─────────────────────────────────────────────────────
    def augment_fn(frame: Image.Image, mask: torch.Tensor):
        if random.random() < 0.5:
            frame = TF.hflip(frame)
            mask  = TF.hflip(mask.unsqueeze(0)).squeeze(0)

        if random.random() < 0.5:
            frame = TF.vflip(frame)
            mask  = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            frame = TF.rotate(frame, angle,
                               interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask  = TF.rotate(mask.unsqueeze(0), angle,
                               interpolation=TF.InterpolationMode.NEAREST, fill=0).squeeze(0)

        if random.random() < 0.3:
            frame = TF.affine(
                frame,
                angle=0.0,
                translate=(random.randint(-12, 12), random.randint(-12, 12)),
                scale=random.uniform(0.9, 1.1),
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask.unsqueeze(0),
                angle=0.0,
                translate=(random.randint(-12, 12), random.randint(-12, 12)),
                scale=random.uniform(0.9, 1.1),
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.NEAREST,
                fill=0,
            ).squeeze(0)

        if random.random() < 0.8:
            frame = TF.adjust_brightness(frame, random.uniform(0.75, 1.25))
            frame = TF.adjust_contrast(frame,   random.uniform(0.80, 1.20))
            frame = TF.adjust_saturation(frame, random.uniform(0.80, 1.20))

        if random.random() < 0.2:
            frame = TF.adjust_hue(frame, random.uniform(-0.03, 0.03))

        if random.random() < 0.15:
            frame = TF.gaussian_blur(frame, kernel_size=3)

        return frame, mask

    train_ds = AugmentWrapper(train_ds, augment_fn)

    _collate = collate_fn(bridge, device, image_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate, pin_memory=False,
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], device=device)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def focal_loss(logits, targets, alpha=0.25, gamma=2.0, eps=1e-6):
        probs = torch.sigmoid(logits)
        bce   = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt      = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt + eps) ** gamma * bce).mean()

    def tversky_loss(logits, targets, alpha=0.2, beta=0.8, eps=1e-6):
        probs   = torch.sigmoid(logits).reshape(logits.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)
        return (1.0 - ((tp + eps) / (tp + alpha * fn + beta * fp + eps))).mean()

    def criterion(logits, targets):
        return focal_loss(logits, targets) + tversky_loss(logits, targets)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss      = float("inf")
    best_val_iou       = -float("inf")
    best_val_precision = -float("inf")
    epochs_no_improve  = 0
    print(f"\nStarting training for {args.epochs} epochs (patience={args.patience}) …\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou, train_prec = run_epoch(
            bridge, train_loader, criterion, optimizer, device, image_size, train=True,
        )
        val_loss, val_iou, val_prec = run_epoch(
            bridge, val_loader, criterion, None, device, image_size, train=False,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_iou={train_iou:.4f}  train_prec={train_prec:.4f}  "
            f"val_loss={val_loss:.4f}  val_iou={val_iou:.4f}  val_prec={val_prec:.4f}  "
            f"lr={lr_now:.2e}"
        )

        def _save(path: Path, extra: dict) -> None:
            torch.save(
                {
                    "epoch": epoch,
                    "bridge_state_dict": bridge.bridge.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "val_precision": val_prec,
                    **extra,
                },
                path,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save(ckpt_dir / "bridge_best_loss.pt", {"metric": "val_loss"})
            print(f"  ↳ Best val_loss={best_val_loss:.4f} → {ckpt_dir / 'bridge_best_loss.pt'}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            _save(ckpt_dir / "bridge_best_iou.pt", {"metric": "val_iou"})
            print(f"  ↳ Best val_iou={best_val_iou:.4f} → {ckpt_dir / 'bridge_best_iou.pt'}")
        else:
            epochs_no_improve += 1
            print(f"  (no val_iou improvement for {epochs_no_improve}/{args.patience} epochs)")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        if val_prec > best_val_precision:
            best_val_precision = val_prec
            _save(ckpt_dir / "bridge_best_precision.pt", {"metric": "val_precision"})
            print(f"  ↳ Best val_prec={best_val_precision:.4f} → {ckpt_dir / 'bridge_best_precision.pt'}")

        _save(ckpt_dir / f"bridge_epoch{epoch:03d}.pt", {})

    print(f"\nTraining complete.")
    print(f"  Best val_loss={best_val_loss:.4f}  → {ckpt_dir / 'bridge_best_loss.pt'}")
    print(f"  Best val_iou={best_val_iou:.4f}   → {ckpt_dir / 'bridge_best_iou.pt'}")
    print(f"  Best val_prec={best_val_precision:.4f} → {ckpt_dir / 'bridge_best_precision.pt'}")


if __name__ == "__main__":
    main()
