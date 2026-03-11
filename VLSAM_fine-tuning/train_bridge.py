"""
train_bridge.py
===============
Trains SaliencyBridge (the CNN head on top of frozen DINOv3) to predict
binary foreground masks from DINOv3 patch features.

Dataset layout expected
-----------------------
  <data_root>/
    frames/
      <video>/
        00000.jpg
        00001.jpg
        ...
    masks/
      <video>/
        00000.png    ← binary mask, white = foreground
        00001.png

Usage
-----
  python train_bridge.py \
      --data_root /Experiments/marcol01 \
      --epochs 20 \
      --batch_size 4 \
      --lr 1e-4 \
      --device cuda:0 \
      --checkpoint_dir ./bridge_ckpts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ── make sure the package is importable from any cwd ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dinov3_sam2_bridge import DINOv3SAM2Bridge


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FrameMaskDataset(Dataset):
    """
    Pairs every frame jpg with its corresponding mask png.

    Parameters
    ----------
    data_root : str | Path
        Root that contains `frames/<video>/` and `masks/<video>/`.
    image_size : tuple[int, int]
        (H, W) to resize frames and masks to before feeding the model.
        Must be a multiple of 14 (DINOv3 patch size).
    """

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
                # look for matching mask with any supported extension
                mask_path = None
                for ext in self.MASK_EXTENSIONS:
                    cand = mask_dir / (frame_path.stem + ext)
                    if cand.exists():
                        mask_path = cand
                        break
                if mask_path is None:
                    continue  # skip frames without a mask
                self.pairs.append((frame_path, mask_path))

        if not self.pairs:
            raise RuntimeError(
                f"No (frame, mask) pairs found under {data_root}. "
                "Check that frames/ and masks/ subdirectories exist and are non-empty."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_path, mask_path = self.pairs[idx]

        # ── frame ──────────────────────────────────────────────────────────
        # Do NOT resize here — preprocess() in collate_fn handles the resize
        # so DINO sees exactly the same geometry as the GT mask.
        frame = Image.open(frame_path).convert("RGB")

        # ── mask ───────────────────────────────────────────────────────────
        mask = Image.open(mask_path).convert("L")  # grayscale
        mask = mask.resize(
            (self.image_size[1], self.image_size[0]), Image.NEAREST
        )
        mask_np = np.array(mask, dtype=np.float32)
        mask_tensor = torch.from_numpy((mask_np > 127).astype(np.float32))  # (H, W) 0/1

        return frame, mask_tensor  # frame is PIL; collate handles preprocessing


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AugmentWrapper(Dataset):
    """
    Wraps a Subset and applies a joint augmentation function to each
    (PIL frame, mask tensor) pair.  Only used for the training split.

    The aug_fn receives (PIL Image, torch.Tensor mask H×W) and must
    return the same types.
    """

    def __init__(self, subset: Dataset, aug_fn) -> None:
        self.subset = subset
        self.aug_fn = aug_fn

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        frame, mask = self.subset[idx]   # PIL Image, float tensor (H, W)
        return self.aug_fn(frame, mask)


def collate_fn(bridge: DINOv3SAM2Bridge, device: torch.device, image_size: tuple[int, int]):
    """
    Returns a collate function that resizes + normalises PIL frames to
    exactly `image_size` using DINO's own mean/std — guaranteeing that
    the DINO token grid is aligned with the GT mask geometry.
    """
    def _collate(batch: list[tuple[Image.Image, torch.Tensor]]):
        frames, masks = zip(*batch)
        # size= forces an exact resize without any hidden processor crop/resize
        pixel_values = bridge.extractor.preprocess(list(frames), device=device, size=image_size)
        masks = torch.stack(masks, dim=0).to(device)  # (B, H, W)
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
    # Accumulate TP/FP/FN globally so metrics reflect the full epoch,
    # not an average of per-batch values.
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with context:
        for pixel_values, gt_masks in tqdm(loader, desc="train" if train else "val ", leave=False):
            # pixel_values : (B, 3, H, W)  already on device (via collate)
            # gt_masks     : (B, H, W)     float 0/1

            logits = bridge(pixel_values, target_size=image_size)  # (B, 1, H, W)
            logits = logits.squeeze(1)                              # (B, H, W)

            loss = criterion(logits, gt_masks)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            # ── Per-pixel binary metrics (threshold logits at 0) ──────────
            with torch.no_grad():
                pred = (logits.detach() > 0).long()   # (B, H, W)
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
    p = argparse.ArgumentParser(description="Train SaliencyBridge on top of frozen DINOv3")

    p.add_argument("--data_root",       required=True,
                   help="Root with frames/ and masks/ subdirs")
    p.add_argument("--image_size",      type=int, nargs=2, default=[448, 448],
                   metavar=("H", "W"),
                   help="Resize all frames/masks to this size (must be multiples of 14)")
    p.add_argument("--val_split",       type=float, default=0.05,
                   help="Fraction of data used for validation")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--pos_weight",      type=float, default=None,
                   help="Positive class weight for BCEWithLogitsLoss "
                        "(use >1 if foreground is rare, e.g. camouflaged objects)")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint_dir",  default="./bridge_ckpts",
                   help="Where to save model checkpoints")
    p.add_argument("--resume",          default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--num_workers",     type=int, default=4)

    return p.parse_args()


def main() -> None:
    args = get_args()
    device     = torch.device(args.device)
    image_size = tuple(args.image_size)  # (H, W)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────
    # Load bridge first so we can read the actual patch_size from the model.
    print("Loading DINOv3SAM2Bridge …")
    bridge = DINOv3SAM2Bridge(freeze_backbone=True, device=device)

    # ── Validate image_size ────────────────────────────────────────────────
    patch = bridge.extractor.patch_size  # read from the actual loaded model
    if image_size[0] % patch != 0 or image_size[1] % patch != 0:
        raise ValueError(
            f"image_size {image_size} must be multiples of {patch} (model patch size). "
            f"Try {(round(image_size[0]/patch)*patch, round(image_size[1]/patch)*patch)}."
        )

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        bridge.bridge.load_state_dict(state["bridge_state_dict"])
        print(f"Resumed from {args.resume}")

    # Only train the CNN head, not the frozen backbone
    trainable_params = list(bridge.bridge.parameters())
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_params:,}  (SaliencyBridge only)")

    # ── Dataset & Dataloaders ─────────────────────────────────────────────
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
    # Define joint transforms here.  Geometric ops must be applied to BOTH
    # frame and mask; colour ops (brightness, contrast, …) to frame only.
    # Add or remove transforms in the body of `augment_fn` as needed.
    # ──────────────────────────────────────────────────────────────────────
    def augment_fn(frame: Image.Image, mask: torch.Tensor):
        # ── Geometric (applied to both) ───────────────────────────────────
        # Random horizontal flip
        if random.random() < 0.5:
            frame = TF.hflip(frame)
            mask  = TF.hflip(mask.unsqueeze(0)).squeeze(0)

        # Random vertical flip
        if random.random() < 0.5:
            frame = TF.vflip(frame)
            mask  = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            frame = TF.rotate(frame, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask.unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST, fill=0).squeeze(0)

        if random.random() < 0.3:
            angle = 0.0
            scale = random.uniform(0.9, 1.1)
            translate = (random.randint(-12, 12), random.randint(-12, 12))
            shear = [0.0, 0.0]

            frame = TF.affine(
                frame,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )

            mask = TF.affine(
                mask.unsqueeze(0),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=TF.InterpolationMode.NEAREST,
                fill=0,
            ).squeeze(0)

        # ── Colour / photometric (frame only) ─────────────────────────────
        # Random brightness / contrast / saturation jitter
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
    # val_ds is intentionally left without augmentation.

    _collate = collate_fn(bridge, device, image_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate,
        pin_memory=False,
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], device=device)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Sigmoid focal loss (Lin et al. 2017).
        alpha : foreground prior weight (down-weights easy negatives).
        gamma : focusing exponent (larger → more focus on hard examples /
                small objects).
        """
        probs = torch.sigmoid(logits)
        # per-pixel binary cross-entropy (numerically stable)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)  # p_t
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        fl = alpha_t * (1.0 - pt + eps) ** gamma * bce
        return fl.mean()

    def dice_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Soft Dice loss on sigmoid probabilities."""
        probs   = torch.sigmoid(logits)
        probs   = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        intersection = (probs * targets).sum(dim=1)
        union        = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + eps) / (union + eps)
        return 1.0 - dice.mean()

    def tversky_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.2,
        beta: float = 0.8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Tversky loss (Salehi et al. 2017).
        alpha : weight on False Negatives  (>0.5 → penalise missed pixels)
        beta  : weight on False Positives  (>0.5 → penalise over-segmentation)
        Default alpha=0.2, beta=0.8 penalises FP more → reduces over-segmentation.
        """
        probs   = torch.sigmoid(logits)
        probs   = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)
        tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
        return 1.0 - tversky.mean()

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """BCE + Dice + Focal (small-object recall) + Tversky (over-segmentation control)."""
        return (
            #bce_fn(logits, targets)
            #+ dice_loss(logits, targets)
            focal_loss(logits, targets)
            + tversky_loss(logits, targets)
        )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss      = float("inf")
    best_val_iou       = -float("inf")
    best_val_precision = -float("inf")
    print(f"\nStarting training for {args.epochs} epochs …\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou, train_prec = run_epoch(
            bridge, train_loader, criterion, optimizer,
            device, image_size, train=True,
        )
        val_loss, val_iou, val_prec = run_epoch(
            bridge, val_loader, criterion, None,
            device, image_size, train=False,
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

        # ── Save best-val-loss checkpoint ─────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_dir / "bridge_best_loss.pt"
            _save(best_path, {"metric": "val_loss"})
            print(f"  ↳ Best val_loss={best_val_loss:.4f} → {best_path}")

        # ── Save best-val-IoU checkpoint ──────────────────────────────────
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            iou_path = ckpt_dir / "bridge_best_iou.pt"
            _save(iou_path, {"metric": "val_iou"})
            print(f"  ↳ Best val_iou={best_val_iou:.4f} → {iou_path}")

        # ── Save best-val-precision checkpoint ───────────────────────────
        if val_prec > best_val_precision:
            best_val_precision = val_prec
            prec_path = ckpt_dir / "bridge_best_precision.pt"
            _save(prec_path, {"metric": "val_precision"})
            print(f"  ↳ Best val_prec={best_val_precision:.4f} → {prec_path}")

        # ── Save periodic checkpoint every epoch ──────────────────────────
        if epoch % 1 == 0:
            latest_path = ckpt_dir / f"bridge_epoch{epoch:03d}.pt"
            _save(latest_path, {})

    print(f"\nTraining complete.")
    print(f"  Best val_loss={best_val_loss:.4f}  → {ckpt_dir / 'bridge_best_loss.pt'}")
    print(f"  Best val_iou={best_val_iou:.4f}   → {ckpt_dir / 'bridge_best_iou.pt'}")
    print(f"  Best val_prec={best_val_precision:.4f} → {ckpt_dir / 'bridge_best_precision.pt'}")


if __name__ == "__main__":
    main()
