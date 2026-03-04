"""
train_sam_adapter.py — Fine-tune SAM2 adapter modules using standard SAM prompts.

No VLM (BLIP / Mamba) involved.  Prompts are point or box clicks sampled from
the ground-truth masks, exactly as in the original SAM/SAM2 training regime.

What is trained (by default):
  • image_encoder.adapter           — FPN-neck bottleneck adapter
  • image_encoder.trunk.adapters    — per-Hiera-block bottleneck adapters

What is always frozen:
  • image_encoder backbone (everything except the adapters above)
  • sam_prompt_encoder

Optional:
  • --train_mask_decoder   also fine-tunes sam_mask_decoder

Dataset layout:
  <frames_dir>/<video_name>/<frame>.jpg   (RGB frames)
  <masks_dir>/<video_name>/<frame>.png    (binary masks, same stem)

Checkpoint saved to <work_dir>/adapter_best.pth and adapter_epochN.pth.

Usage:
  conda activate thesis
  PYTHONPATH=/home/marcol01/thesis:$PYTHONPATH \\
  python tools/train_sam_adapter.py \\
      --sam2_cfg   configs/sam2.1/sam2.1_hiera_l.yaml \\
      --sam2_ckpt  /home/marcol01/thesis/sam2.1_hiera_large.pt \\
      --frames_dir /path/to/frames \\
      --masks_dir  /path/to/masks \\
      --work_dir   ./work_dir/adapter
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_utils import get_next_point, sample_box_points

# ---------------------------------------------------------------------------
# SAM normalisation constants (same as SAM2ImagePredictor)
# ---------------------------------------------------------------------------
_SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
_SAM_PIXEL_STD  = torch.tensor([58.395,  57.12,  57.375]).view(3, 1, 1)
_SAM_SIZE       = 1024

# Spatial sizes of the three FPN feature levels (same as SAM2ImagePredictor)
_BB_FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FrameMaskDataset(Dataset):
    """Per-frame image + binary mask pairs from a video-folder layout."""

    def __init__(self, frames_dir: str, masks_dir: str, video_names=None):
        self.samples = []

        if video_names is None:
            video_names = sorted(
                p for p in os.listdir(frames_dir)
                if os.path.isdir(os.path.join(frames_dir, p))
            )

        img_exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
        for vid in video_names:
            vf = os.path.join(frames_dir, vid)
            vm = os.path.join(masks_dir,  vid)
            if not (os.path.isdir(vf) and os.path.isdir(vm)):
                continue
            for fname in sorted(os.listdir(vf)):
                stem, ext = os.path.splitext(fname)
                if ext not in img_exts:
                    continue
                mpath = os.path.join(vm, stem + ".png")
                if os.path.exists(mpath):
                    self.samples.append((os.path.join(vf, fname), mpath))

        print(f"[Dataset] {len(self.samples)} frame-mask pairs found.")

        self._mask_tf = T.Compose([
            T.Resize((_SAM_SIZE, _SAM_SIZE),
                     interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # --- Image: resize + SAM normalise → (3, 1024, 1024) float32 --------
        pil = Image.open(img_path).convert("RGB")
        pil = pil.resize((_SAM_SIZE, _SAM_SIZE), Image.BILINEAR)
        img_t = torch.as_tensor(np.array(pil), dtype=torch.float32).permute(2, 0, 1)
        img_t = (img_t - _SAM_PIXEL_MEAN) / _SAM_PIXEL_STD

        # --- Mask: resize + binarise → (1, 1024, 1024) float32 ---------------
        pil_m = Image.open(mask_path).convert("L")
        gt = self._mask_tf(pil_m)          # (1, 1024, 1024) in [0, 1]
        gt = (gt > 0.5).float()

        return img_t, gt


def collate_fn(batch):
    imgs, gts = zip(*batch)
    return torch.stack(imgs), torch.stack(gts)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0):
    prob = torch.sigmoid(logits).flatten(1)
    tgt  = targets.flatten(1)
    num  = 2.0 * (prob * tgt).sum(1) + eps
    den  = prob.sum(1) + tgt.sum(1)  + eps
    return (1.0 - num / den).mean()


def seg_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy + soft Dice."""
    bce  = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce + dice


# ---------------------------------------------------------------------------
# Feature extraction  (mirrors SAM2ImagePredictor.set_image, with grad)
# ---------------------------------------------------------------------------
def encode_images(model, imgs: torch.Tensor):
    """
    Run image_encoder and return (image_embed, high_res_feats).

    image_embed    : (B, C, 64, 64)  — top-level backbone feature
    high_res_feats : list of 2 tensors  [(B, C, 256, 256), (B, C, 128, 128)]
    """
    backbone_out = model.forward_image(imgs)          # runs image_encoder

    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)

    # Add no_mem_embed if configured (replaces video-memory conditioning on images)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    B = imgs.shape[0]
    # vision_feats are in (HW, B, C) format; reshape to (B, C, H, W)
    feats = [
        feat.permute(1, 2, 0).view(B, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], _BB_FEAT_SIZES[::-1])
    ][::-1]                                            # back to coarse→fine order

    image_embed    = feats[-1]           # (B, C, 64, 64)
    high_res_feats = feats[:-1]          # [(B, C, 256, 256), (B, C, 128, 128)]
    return image_embed, high_res_feats


# ---------------------------------------------------------------------------
# Point / box prompt sampling
# ---------------------------------------------------------------------------
def sample_prompts(gt: torch.Tensor, use_box: bool):
    """
    Sample one point (or a box) from the GT mask.

    gt : (B, 1, 1024, 1024)  float binary mask

    Returns point_inputs dict {"point_coords": (B,P,2), "point_labels": (B,P)}.
    (Box inputs are encoded as two corner points with labels 2 and 3, matching
    the standard SAM encoding used inside sam_prompt_encoder.)
    """
    # convert float masks to boolean using threshold 0.5
    gt_bool = (gt > 0.5)
    if use_box:
        coords, labels = sample_box_points(gt_bool)
    else:
        coords, labels = get_next_point(
            gt_masks=gt_bool,
            pred_masks=None,
            method="uniform",
        )
    return {"point_coords": coords, "point_labels": labels}


# ---------------------------------------------------------------------------
# Single forward pass → mask logits
# ---------------------------------------------------------------------------
def forward_pass(model, imgs: torch.Tensor, gt: torch.Tensor,
                 use_box: bool = False):
    """
    Returns high-res mask logits of shape (B, 1, 1024, 1024).
    """
    device = imgs.device

    # 1. Encode images (through adapted image_encoder)
    image_embed, high_res_feats = encode_images(model, imgs)   # grads flow here

    # 2. Sample prompts from GT (no grad needed)
    with torch.no_grad():
        point_inputs = sample_prompts(gt, use_box=use_box)
    point_coords = point_inputs["point_coords"].to(device)
    point_labels = point_inputs["point_labels"].to(device)

    # 3. Encode prompts (frozen prompt encoder)
    with torch.no_grad():
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )

    # 4. Decode masks
    low_res_masks, _ious, _tokens, _obj_scores = model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats,
    )

    # 5. Upsample to 1024×1024
    high_res_masks = F.interpolate(
        low_res_masks.float(),
        size=(_SAM_SIZE, _SAM_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return high_res_masks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, epoch: int,
                    prob_box: float = 0.5):
    model.train()
    total_loss, n = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    for imgs, gt in pbar:
        imgs = imgs.to(device)
        gt   = gt.to(device)

        use_box = (torch.rand(1).item() < prob_box)
        pred = forward_pass(model, imgs, gt, use_box=use_box)

        loss = seg_loss(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n          += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         prompt="box" if use_box else "point")

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, prob_box: float = 0.5):
    model.eval()
    all_dice = []

    for imgs, gt in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device)
        gt   = gt.to(device)

        use_box = (torch.rand(1).item() < prob_box)
        pred = forward_pass(model, imgs, gt, use_box=use_box)

        pred_bin = (pred > 0.0).float()
        num  = 2.0 * (pred_bin * gt).sum()
        den  = pred_bin.sum() + gt.sum()
        all_dice.append((num / (den + 1e-8)).item())

    return float(np.mean(all_dice)) if all_dice else 0.0


def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict()}, path)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Fine-tune SAM2 adapter modules with standard SAM prompts"
    )

    # Model
    p.add_argument("--sam2_cfg",  required=True,
                   help="SAM2 config (e.g. configs/sam2.1/sam2.1_hiera_l.yaml)")
    p.add_argument("--sam2_ckpt", required=True,
                   help="SAM2 base checkpoint (.pt)")
    p.add_argument("--adapter_ckpt", default=None,
                   help="Resume from an existing adapter checkpoint (optional)")

    # Data
    p.add_argument("--frames_dir", required=True,
                   help="Root dir with per-video JPEG frame sub-folders")
    p.add_argument("--masks_dir",  required=True,
                   help="Root dir with per-video PNG mask sub-folders")
    p.add_argument("--val_frames_dir", default=None)
    p.add_argument("--val_masks_dir",  default=None)
    p.add_argument("--video_list_file", default=None,
                   help="Text file: one video name per line (default: all sub-dirs)")

    # Training knobs
    p.add_argument("--epochs",            type=int,   default=30)
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--weight_decay",      type=float, default=1e-4)
    p.add_argument("--prob_box",          type=float, default=0.5,
                   help="Probability of using a box prompt instead of a point (default 0.5)")
    p.add_argument("--train_mask_decoder", action="store_true",
                   help="Also fine-tune sam_mask_decoder (default: frozen)")
    p.add_argument("--val_every",  type=int, default=5)
    p.add_argument("--save_every", type=int, default=10)

    # Output / hardware
    p.add_argument("--work_dir",    default="./work_dir/adapter")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--num_workers", type=int, default=4)

    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.work_dir, exist_ok=True)
    print(f"Device: {device}")

    # ── Build model ────────────────────────────────────────────────────────
    print("Loading SAM2 …")
    model = build_sam2(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_ckpt,
        device=device,
        mode="train",
    )

    if args.adapter_ckpt is not None:
        ckpt = torch.load(args.adapter_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"  Resumed from {args.adapter_ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # ── Freeze / unfreeze ─────────────────────────────────────────────────
    # Freeze everything first
    for p_ in model.parameters():
        p_.requires_grad_(False)

    # Unfreeze FPN-neck adapter
    found_adapter = False
    if hasattr(model.image_encoder, "adapter"):
        for p_ in model.image_encoder.adapter.parameters():
            p_.requires_grad_(True)
        print("  Training: image_encoder.adapter (FPN-neck)")
        found_adapter = True

    # Unfreeze per-block Hiera adapters
    if hasattr(model.image_encoder, "trunk") and \
            hasattr(model.image_encoder.trunk, "adapters"):
        for p_ in model.image_encoder.trunk.adapters.parameters():
            p_.requires_grad_(True)
        print("  Training: image_encoder.trunk.adapters (Hiera per-block)")
        found_adapter = True

    if not found_adapter:
        print("  WARNING: no adapter found in image_encoder or trunk. "
              "Make sure adapters are present in image_encoder.py / hieradet.py.")

    # Optionally unfreeze mask decoder
    if args.train_mask_decoder:
        for p_ in model.sam_mask_decoder.parameters():
            p_.requires_grad_(True)
        print("  Training: sam_mask_decoder")

    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    n_total = sum(p_.numel() for p_ in model.parameters())
    print(f"  Trainable: {n_train:,} / {n_total:,} params "
          f"({100 * n_train / n_total:.2f} %)")

    # ── Datasets ───────────────────────────────────────────────────────────
    video_names = None
    if args.video_list_file is not None:
        with open(args.video_list_file) as f:
            video_names = [v.strip() for v in f if v.strip()]

    train_ds = FrameMaskDataset(args.frames_dir, args.masks_dir, video_names)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if args.val_frames_dir and args.val_masks_dir:
        val_ds = FrameMaskDataset(args.val_frames_dir, args.val_masks_dir)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

    # ── Optimiser ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p_ for p_ in model.parameters() if p_.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.prob_box
        )
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loader is not None and epoch % args.val_every == 0:
            dice = validate(model, val_loader, device, args.prob_box)
            print(f"  Val Dice: {dice:.4f}")
            if dice > best_dice:
                best_dice = dice
                save_checkpoint(
                    model,
                    os.path.join(args.work_dir, "adapter_best.pth"),
                )

        if epoch % args.save_every == 0:
            save_checkpoint(
                model,
                os.path.join(args.work_dir, f"adapter_epoch{epoch}.pth"),
            )

    save_checkpoint(model, os.path.join(args.work_dir, "adapter_final.pth"))
    if val_loader is not None:
        print(f"\nDone. Best val Dice: {best_dice:.4f}")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
