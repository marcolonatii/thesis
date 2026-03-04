"""
train_vlsam.py — Fine-tune the VLSAM-specific modules on a segmentation dataset.

What is trained (by default):
  • pseudo_mask_embed  (Conv2d + GELU dense-prompt projection)
  • pe_layer           (PositionEmbeddingRandom)

What is always frozen:
  • SAM2 image_encoder
  • BLIP vision model + text decoder
  • Mamba model

Optional:
  • --train_mask_decoder  also fine-tunes sam_mask_decoder

Dataset structure expected under --data_root:
  <data_root>/
    Imgs/    *.jpg  (RGB images)
    GT/      *.png  (binary masks, same stem as Imgs)

Checkpoint saved under --work_dir as vlsam_best.pth and vlsam_epochN.pth.
Compatible with vlsam_vos_inference.py (checkpoint["model"] key).

Usage:
  conda activate thesis
  PYTHONPATH=/home/marcol01/thesis:$PYTHONPATH \\
  python tools/train_vlsam.py \\
      --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \\
      --sam2_checkpoint /home/marcol01/thesis/sam2.1_hiera_large.pt \\
      --data_root /path/to/dataset \\
      --work_dir ./work_dir/vlsam
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
from torchvision import transforms
from tqdm import tqdm

# -- local imports -----------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.build_sam import build_sam2
from transformers import (
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
    MambaModel,
)

# Re-use VLSAM class and helpers from the inference script
from tools.vlsam_vos_inference import (
    VLSAM,
    _get_vlm_features,
    _preprocess_image_for_sam,
    _SAM_INPUT_SIZE,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VLSAMDataset(Dataset):
    """Image + binary mask dataset for a video-folder layout.

    Expects:
        <frames_dir>/<video_name>/<frame>.jpg   (or .jpeg)
        <masks_dir>/<video_name>/<frame>.png    (binary / palette mask)

    Every (frame, mask) pair where both files exist is used as a sample.
    """

    def __init__(self, frames_dir: str, masks_dir: str, video_names=None):
        self.samples = []   # list of (img_path, mask_path)

        if video_names is None:
            video_names = sorted(
                p for p in os.listdir(frames_dir)
                if os.path.isdir(os.path.join(frames_dir, p))
            )

        img_exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
        for vid in video_names:
            vid_frame_dir = os.path.join(frames_dir, vid)
            vid_mask_dir = os.path.join(masks_dir, vid)
            if not os.path.isdir(vid_frame_dir) or not os.path.isdir(vid_mask_dir):
                continue
            for fname in sorted(os.listdir(vid_frame_dir)):
                stem, ext = os.path.splitext(fname)
                if ext not in img_exts:
                    continue
                mask_path = os.path.join(vid_mask_dir, stem + ".png")
                if os.path.exists(mask_path):
                    self.samples.append(
                        (os.path.join(vid_frame_dir, fname), mask_path)
                    )

        print(f"[Dataset] {len(self.samples)} frame-mask pairs found.")

        self.mask_transform = transforms.Compose([
            transforms.Resize((_SAM_INPUT_SIZE, _SAM_INPUT_SIZE),
                               interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        pil_img = Image.open(img_path).convert("RGB")
        pil_gt = Image.open(mask_path).convert("L")

        gt = self.mask_transform(pil_gt)   # (1, 1024, 1024) float [0,1]
        gt = (gt > 0.5).float()            # binarise

        return pil_img, gt                 # pil_img stays PIL for BLIP


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0):
    """Soft Dice loss on sigmoid-activated logits."""
    prob = torch.sigmoid(logits).flatten(1)
    tgt = targets.flatten(1)
    num = 2.0 * (prob * tgt).sum(1) + eps
    den = prob.sum(1) + tgt.sum(1) + eps
    return (1.0 - num / den).mean()


def seg_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce + dice


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_vlm_features_batched(pil_images, vlm_model, processor, mamba_model,
                              tokenizer, device):
    """Run BLIP + Mamba on a list of PIL images; return lists of tensors."""
    text_embs, img_feats = [], []
    for pil in pil_images:
        t, i = _get_vlm_features(pil, vlm_model, processor, mamba_model,
                                  tokenizer, device)
        text_embs.append(t)
        img_feats.append(i)
    return text_embs, img_feats


def collate_fn(batch):
    """Keep PIL images as a list; stack GT tensors."""
    pil_imgs, gts = zip(*batch)
    return list(pil_imgs), torch.stack(gts, dim=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(vlsam_model, loader, optimizer, vlm_model, processor,
                    mamba_model, tokenizer, device, epoch: int, args):
    vlsam_model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    for pil_imgs, gt in pbar:
        gt = gt.to(device)                                    # (B, 1, H, W)

        # --- VLM features (no grad, models are frozen) --------------------
        text_embs, img_feats = get_vlm_features_batched(
            pil_imgs, vlm_model, processor, mamba_model, tokenizer, device
        )

        # --- SAM-normalised image tensors ----------------------------------
        img_tensors = torch.cat([
            _preprocess_image_for_sam(p, device) for p in pil_imgs
        ], dim=0)                                             # (B, 3, 1024, 1024)

        # --- Forward one sample at a time (VLSAM currently processes B=1) -
        batch_loss = torch.tensor(0.0, device=device)
        for i in range(len(pil_imgs)):
            pred = vlsam_model(
                img_tensors[i:i+1],
                text_embs[i],
                img_feats[i],
            )                                                 # (1, 1, 1024, 1024)
            # resize GT to match prediction if needed
            gt_i = gt[i:i+1]                                 # (1, 1, 1024, 1024)
            if gt_i.shape[-2:] != pred.shape[-2:]:
                gt_i = F.interpolate(gt_i, size=pred.shape[-2:], mode="nearest")
            batch_loss = batch_loss + seg_loss(pred, gt_i)

        batch_loss = batch_loss / len(pil_imgs)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(vlsam_model, loader, vlm_model, processor, mamba_model, tokenizer,
             device, score_thresh: float = 0.0):
    vlsam_model.eval()
    all_dice = []

    for pil_imgs, gt in tqdm(loader, desc="Val", leave=False):
        gt = gt.to(device)
        text_embs, img_feats = get_vlm_features_batched(
            pil_imgs, vlm_model, processor, mamba_model, tokenizer, device
        )
        img_tensors = torch.cat([
            _preprocess_image_for_sam(p, device) for p in pil_imgs
        ], dim=0)

        for i in range(len(pil_imgs)):
            pred = vlsam_model(img_tensors[i:i+1], text_embs[i], img_feats[i])
            gt_i = gt[i:i+1]
            if gt_i.shape[-2:] != pred.shape[-2:]:
                gt_i = F.interpolate(gt_i, size=pred.shape[-2:], mode="nearest")

            pred_bin = (pred > score_thresh).float()
            gt_bin = gt_i.float()
            num = 2.0 * (pred_bin * gt_bin).sum()
            den = pred_bin.sum() + gt_bin.sum()
            all_dice.append((num / (den + 1e-8)).item())

    return float(np.mean(all_dice)) if all_dice else 0.0


def save_checkpoint(vlsam_model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": vlsam_model.state_dict()}, path)
    print(f"  Saved checkpoint → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune VLSAM modules")

    # Model
    parser.add_argument("--sam2_cfg", required=True,
                        help="SAM2 config (e.g. configs/sam2.1/sam2.1_hiera_l.yaml)")
    parser.add_argument("--sam2_checkpoint", required=True,
                        help="SAM2 checkpoint (.pt)")
    parser.add_argument("--vlsam_checkpoint", default=None,
                        help="Resume from existing VLSAM checkpoint (optional)")

    # Data
    parser.add_argument("--frames_dir", required=True,
                        help="Root dir with per-video JPEG frame folders "
                             "(e.g. /Experiments/marcol01/frames)")
    parser.add_argument("--masks_dir", required=True,
                        help="Root dir with per-video PNG mask folders "
                             "(e.g. /Experiments/marcol01/masks)")
    parser.add_argument("--val_frames_dir", default=None,
                        help="Optional separate validation frames dir")
    parser.add_argument("--val_masks_dir", default=None,
                        help="Optional separate validation masks dir")
    parser.add_argument("--video_list_file", default=None,
                        help="Text file listing video names to use (one per line); "
                             "defaults to all subdirs in frames_dir")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_mask_decoder", action="store_true",
                        help="Also fine-tune the SAM2 mask decoder "
                             "(default: frozen, only pseudo_mask_embed + pe_layer)")
    parser.add_argument("--train_adapter", action="store_true",
                        help="Also fine-tune the adapter inside ImageEncoder "
                             "(requires adapter to be uncommented in image_encoder.py)")
    parser.add_argument("--score_thresh", type=float, default=0.0,
                        help="Threshold for validation Dice (default: 0.0)")
    parser.add_argument("--val_every", type=int, default=5,
                        help="Run validation every N epochs (default: 5)")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save periodic checkpoint every N epochs (default: 10)")

    # Output
    parser.add_argument("--work_dir", default="./work_dir/vlsam",
                        help="Directory for checkpoints and logs")

    # Hardware
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.work_dir, exist_ok=True)
    print(f"Device: {device}")

    # ── SAM2 + VLSAM ──────────────────────────────────────────────────────
    print("Loading SAM2 …")
    sam2_model = build_sam2(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=device,
        mode="train",
    )

    vlsam_model = VLSAM(
        image_encoder=sam2_model.image_encoder,
        mask_decoder=sam2_model.sam_mask_decoder,
    ).to(device)

    if args.vlsam_checkpoint is not None:
        ckpt = torch.load(args.vlsam_checkpoint, map_location=device)
        vlsam_model.load_state_dict(ckpt["model"])
        print(f"  Resumed from {args.vlsam_checkpoint}")

    # ── Freeze / unfreeze ─────────────────────────────────────────────────
    # Always freeze SAM2 image encoder
    for p in vlsam_model.image_encoder.parameters():
        p.requires_grad_(False)

    # Optionally unfreeze adapters inside the image encoder
    if args.train_adapter:
        found_any = False
        # FPN-neck adapter (image_encoder.py)
        if hasattr(vlsam_model.image_encoder, "adapter"):
            for p in vlsam_model.image_encoder.adapter.parameters():
                p.requires_grad_(True)
            print("  FPN-neck adapter in ImageEncoder will be trained.")
            found_any = True
        # Per-block adapters in Hiera trunk (hieradet.py)
        if hasattr(vlsam_model.image_encoder, "trunk") and \
                hasattr(vlsam_model.image_encoder.trunk, "adapters"):
            for p in vlsam_model.image_encoder.trunk.adapters.parameters():
                p.requires_grad_(True)
            print("  Per-block adapters in Hiera trunk will be trained.")
            found_any = True
        if not found_any:
            print("  WARNING: --train_adapter set but no adapter found in image_encoder "
                  "or trunk. Make sure adapters are uncommented in image_encoder.py / hieradet.py.")

    if not args.train_mask_decoder:
        for p in vlsam_model.mask_decoder.parameters():
            p.requires_grad_(False)

    trainable = [n for n, p in vlsam_model.named_parameters() if p.requires_grad]
    print(f"  Trainable modules: {set(n.split('.')[0] for n in trainable)}")
    print(f"  Trainable params: {sum(p.numel() for p in vlsam_model.parameters() if p.requires_grad):,}")

    # ── VLM models (frozen) ────────────────────────────────────────────────
    print("Loading BLIP …")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    vlm_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device).eval()
    for p in vlm_model.parameters():
        p.requires_grad_(False)

    print("Loading Mamba …")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    mamba_model = MambaModel.from_pretrained("state-spaces/mamba-130m-hf").to(device).eval()
    for p in mamba_model.parameters():
        p.requires_grad_(False)

    # ── Dataset & dataloader ───────────────────────────────────────────────
    video_names = None
    if args.video_list_file is not None:
        with open(args.video_list_file) as f:
            video_names = [v.strip() for v in f if v.strip()]

    train_ds = VLSAMDataset(args.frames_dir, args.masks_dir, video_names)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if args.val_frames_dir is not None and args.val_masks_dir is not None:
        val_ds = VLSAMDataset(args.val_frames_dir, args.val_masks_dir)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in vlsam_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            vlsam_model, train_loader, optimizer,
            vlm_model, processor, mamba_model, tokenizer,
            device, epoch, args,
        )
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} — train loss: {train_loss:.4f}  lr: {scheduler.get_last_lr()[0]:.2e}")

        # Validation
        if val_loader is not None and epoch % args.val_every == 0:
            dice = validate(
                vlsam_model, val_loader,
                vlm_model, processor, mamba_model, tokenizer,
                device, args.score_thresh,
            )
            print(f"  Val Dice: {dice:.4f}")
            if dice > best_dice:
                best_dice = dice
                save_checkpoint(vlsam_model, os.path.join(args.work_dir, "vlsam_best.pth"))

        # Periodic save
        if epoch % args.save_every == 0:
            save_checkpoint(vlsam_model, os.path.join(args.work_dir, f"vlsam_epoch{epoch}.pth"))

    # Save final checkpoint
    save_checkpoint(vlsam_model, os.path.join(args.work_dir, "vlsam_final.pth"))
    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}" if val_loader else "\nTraining complete.")


if __name__ == "__main__":
    main()
