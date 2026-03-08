# -*- coding: utf-8 -*-
"""
Training script: Grounding DINO box prompts + SAM2 mask decoder.
Extends train_gdino.py to the SAM2 architecture.

Pipeline per image:
  1. Derive text prompt from filename: "camouflaged <animal name>"
  2. Run Grounding DINO (frozen) to obtain the best XYXY bounding box
  3. Encode the box with SAM2 prompt encoder (frozen)
  4. Fine-tune SAM2 mask decoder with Dice + BCE loss
"""

# %% environment
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
import shutil

join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import argparse
from datetime import datetime
import monai
from PIL import Image
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# SAM2  (tools/ -> .. -> thesis/)
sys.path.insert(0, os.path.abspath(join(os.path.dirname(__file__), '..')))
# utils_downstream lives in Vision-Language-SAM/
sys.path.insert(0, os.path.abspath(join(os.path.dirname(__file__), '..', '..', 'Vision-Language-SAM')))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.transforms import SAM2Transforms

# Grounding DINO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Metrics
from utils_downstream.saliency_metric import (
    cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, cal_acc
)

torch.manual_seed(2024)
torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helper: derive text prompt from image file path
# ---------------------------------------------------------------------------
def caption_from_path(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    name = re.sub(r'[_\-]+', ' ', base)
    name = re.sub(r'\d+', '', name).strip()
    name = re.sub(r'\s+', ' ', name).strip()
    if name == '':
        name = 'animal'
    return f'camouflaged {name.lower()}'


# ---------------------------------------------------------------------------
# Helper: run Grounding DINO and return best XYXY box (numpy float32)
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_gdino_box(pil_image, text_prompt, gdino_model, gdino_processor,
                  device, box_threshold=0.3, text_threshold=0.3):
    H, W = pil_image.size[1], pil_image.size[0]
    gdino_text = text_prompt + '.'
    inputs = gdino_processor(
        images=pil_image,
        text=gdino_text,
        return_tensors='pt'
    ).to(device)
    outputs = gdino_model(**inputs)
    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(H, W)]
    )
    boxes  = results[0]['boxes']   # (N, 4) XYXY pixel coords
    scores = results[0]['scores']  # (N,)
    if len(boxes) == 0:
        # Fall back to full-image box
        return np.array([0.0, 0.0, float(W), float(H)], dtype=np.float32)
    best = scores.argmax().item()
    return boxes[best].cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# SAM2 wrapper: trainable mask decoder, frozen image encoder & prompt encoder
# ---------------------------------------------------------------------------
class SAM2WithBox(nn.Module):
    """
    Thin wrapper around a SAM2Base model that exposes a differentiable
    forward pass through the mask decoder only.

    Image encoder and prompt encoder are always run under torch.no_grad().
    Only sam_mask_decoder receives gradients.
    """

    # Backbone feature map sizes (coarse -> fine), matching SAM2ImagePredictor
    _BB_FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]

    def __init__(self, sam2_model):
        super().__init__()
        self.model = sam2_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )

    # ------------------------------------------------------------------
    def _encode_image(self, orig_array: np.ndarray):
        """
        Encode an HxWx3 uint8 numpy image.
        Returns image_embed and high_res_feats (no grad).
        """
        input_image = self._transforms(orig_array)            # 3xHxW float
        input_image = input_image[None, ...].to(self.device)  # 1x3xHxW

        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._BB_FEAT_SIZES[::-1])
        ][::-1]

        image_embed     = feats[-1]           # 1 x C x 64 x 64
        high_res_feats  = feats[:-1]          # list of coarser feature maps
        return image_embed, high_res_feats

    # ------------------------------------------------------------------
    def _encode_box(self, box_xyxy: np.ndarray, orig_hw):
        """
        Encode a (4,) XYXY pixel box as two corner points with labels [2, 3],
        matching the SAM2ImagePredictor internal convention.
        Returns sparse_embeddings, dense_embeddings (no grad).
        """
        box_t = torch.as_tensor(box_xyxy, dtype=torch.float, device=self.device)
        # Transform box coordinates to the model input frame
        unnorm_box = self._transforms.transform_boxes(
            box_t, normalize=True, orig_hw=orig_hw
        )  # shape (1, 2, 2)

        # Represent as corner points with special labels 2 (top-left) & 3 (bottom-right)
        box_coords  = unnorm_box.reshape(1, 2, 2)   # B=1, 2 pts, (x,y)
        box_labels  = torch.tensor([[2, 3]], dtype=torch.int, device=self.device)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(box_coords, box_labels),
            boxes=None,
            masks=None,
        )
        return sparse_embeddings, dense_embeddings

    # ------------------------------------------------------------------
    def forward(self, orig_array: np.ndarray, box_xyxy: np.ndarray):
        """
        Args:
            orig_array : HxWx3 uint8 numpy image
            box_xyxy   : (4,) float32 XYXY box in pixel coords

        Returns:
            logit_mask : (1, 1, H_orig, W_orig) float32 tensor (logits, no sigmoid)
        """
        H_orig, W_orig = orig_array.shape[:2]

        # ---- frozen: image encoder + prompt encoder ----
        with torch.no_grad():
            image_embed, high_res_feats = self._encode_image(orig_array)
            sparse_embeddings, dense_embeddings = self._encode_box(
                box_xyxy, (H_orig, W_orig)
            )

        # ---- trained: mask decoder ----
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        # ---- upscale to original resolution ----
        ori_masks = self._transforms.postprocess_masks(low_res_masks, (H_orig, W_orig))
        return ori_masks  # (1, 1, H_orig, W_orig) — logits

    @property
    def device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NpyDataset(Dataset):
    """
    Loads paired images and GT masks from two directory trees. Supports per-video
    subdirectories: it recursively finds image files under `imgs_dir` and
    matches each image to a mask in `masks_dir` by relative path (replacing
    extension with .png). Only pairs where both image and mask exist are used.
    """
    IMG_EXTS = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    MASK_EXTS = {'.png', '.PNG'}

    def __init__(self, imgs_dir: str, masks_dir: str):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        # Recursively collect image files
        img_files = []
        for root, _, files in os.walk(self.imgs_dir):
            for f in files:
                if os.path.splitext(f)[1] in self.IMG_EXTS:
                    img_files.append(join(root, f))
        img_files = sorted(img_files)

        paired_imgs = []
        paired_masks = []
        missing_masks = []

        for img_path in img_files:
            rel = os.path.relpath(img_path, self.imgs_dir)
            # Candidate mask path mirrors the relative path under masks_dir
            mask_candidate = join(self.masks_dir, os.path.splitext(rel)[0] + '.png')
            if os.path.exists(mask_candidate):
                paired_imgs.append(img_path)
                paired_masks.append(mask_candidate)
            else:
                # Try to find any mask with same stem in corresponding dir
                mask_dir_for_rel = os.path.dirname(join(self.masks_dir, rel))
                stem = os.path.splitext(os.path.basename(rel))[0]
                found = False
                if os.path.isdir(mask_dir_for_rel):
                    for me in self.MASK_EXTS:
                        p = join(mask_dir_for_rel, stem + me)
                        if os.path.exists(p):
                            paired_imgs.append(img_path)
                            paired_masks.append(p)
                            found = True
                            break
                if not found:
                    missing_masks.append(img_path)

        if len(paired_imgs) == 0:
            print(f'No image/mask pairs found in {imgs_dir} and {masks_dir}.')
        else:
            print(f'Number of image/mask pairs: {len(paired_imgs)}')
            if missing_masks:
                print(f'Warning: {len(missing_masks)} images without matching masks were skipped.')

        self.img_path_files = paired_imgs
        self.gt_path_files = paired_masks

        self.mask_transform = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_path_files)

    def __getitem__(self, index):
        gt = Image.open(self.gt_path_files[index]).convert('L')
        gt = self.mask_transform(gt)
        # Actual RGB is loaded on-the-fly (needed by GDINO + SAM2 at native res)
        image_placeholder = torch.zeros(3, 1024, 1024)
        return (
            image_placeholder,
            torch.tensor(gt).long(),
            self.img_path_files[index],
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_epoch(loader, trainable_model, gdino_model, gdino_processor,
               device, box_threshold, text_threshold, out_dir=None):
    """Run evaluation with the SAM2WithBox model directly (for consistency)."""
    trainable_model.eval()

    mae_m, sm_m, em_m, wfm_m, dice_m, iou_m, ber_m, acc_m = (
        cal_mae(), cal_sm(), cal_em(), cal_wfm(),
        cal_dice(), cal_iou(), cal_ber(), cal_acc()
    )

    save_dir = out_dir or join('work_dir', 'pred_masks_gdino_sam2')
    os.makedirs(save_dir, exist_ok=True)

    pbar = tqdm(total=len(loader), leave=False, desc='eval')

    for _, gt2D, img_path in loader:
        img0 = img_path[0] if isinstance(img_path, (list, tuple)) else img_path
        pil_image  = Image.open(img0).convert('RGB')
        orig_array = np.array(pil_image)
        H_orig, W_orig = orig_array.shape[:2]

        text_prompt = caption_from_path(img0)
        box_xyxy = get_gdino_box(
            pil_image, text_prompt, gdino_model, gdino_processor,
            device, box_threshold, text_threshold
        )

        pred_logits = trainable_model(orig_array, box_xyxy)   # (1,1,H,W)
        pred = torch.sigmoid(pred_logits)
        pred_resized = F.interpolate(pred, size=(1024, 1024),
                                     mode='bilinear', align_corners=False)

        res = pred_resized.squeeze().cpu().numpy()
        gt  = gt2D.squeeze().cpu().numpy().astype(np.float32)

        mae_m.update(res, gt)
        sm_m.update(res, gt)
        em_m.update(res, gt)
        wfm_m.update(res, gt)
        dice_m.update(res, gt)
        iou_m.update(res, gt)
        ber_m.update(res, gt)

        # Save predicted mask
        try:
            seg_mask = (res > 0.5).astype('uint8') * 255
            fname = os.path.splitext(os.path.basename(img0))[0] + '.png'
            Image.fromarray(seg_mask).save(join(save_dir, fname))
        except Exception:
            pass

        pbar.update(1)

    pbar.close()
    trainable_model.train()

    return sm_m.show(), em_m.show(), wfm_m.show(), mae_m.show()


# ---------------------------------------------------------------------------
# Main (CLI parsed here, matching train_vlsam.py conventions)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Fine-tune SAM2 mask decoder with Grounding DINO box prompts')

    # Model
    parser.add_argument('--sam2_cfg', required=True,
                        help='SAM2 config (e.g. configs/sam2.1/sam2.1_hiera_l.yaml)')
    parser.add_argument('--sam2_checkpoint', required=True,
                        help='SAM2 checkpoint (.pt)')
    parser.add_argument('--resume', default=None,
                        help='Resume from an existing gdino_sam2 checkpoint (optional)')
    parser.add_argument('--gdino_model', default='IDEA-Research/grounding-dino-base',
                        help='HuggingFace model ID or local path for Grounding DINO')

    # Data
    parser.add_argument('--frames_dir', required=True,
                        help='Directory of training RGB images (.jpg / .png)')
    parser.add_argument('--masks_dir', required=True,
                        help='Directory of training GT masks (.png), '
                             'paired by sorted order with --frames_dir')
    parser.add_argument('--val_frames_dir', default=None,
                        help='Directory of validation RGB images (optional)')
    parser.add_argument('--val_masks_dir', default=None,
                        help='Directory of validation GT masks (optional; '
                             'required when --val_frames_dir is set)')

    # Training
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--batch_size',    type=int,   default=1)
    parser.add_argument('--lr',            type=float, default=2e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-2)
    parser.add_argument('--train_mask_decoder', action='store_true',
                        help='Fine-tune the SAM2 mask decoder (default: True; '
                             'flag kept for compatibility — decoder is always trained)')
    parser.add_argument('--use_amp',       action='store_true', default=False,
                        help='Enable automatic mixed precision (fp16)')
    parser.add_argument('--val_every',     type=int,   default=1,
                        help='Run validation every N epochs (default: 1)')
    parser.add_argument('--save_every',    type=int,   default=5,
                        help='Save a periodic checkpoint every N epochs (default: 5)')

    # Grounding DINO thresholds
    parser.add_argument('--box_threshold',  type=float, default=0.3)
    parser.add_argument('--text_threshold', type=float, default=0.3)

    # Output
    parser.add_argument('--work_dir',  default='./work_dir/gdino_sam2',
                        help='Directory for checkpoints and logs')
    parser.add_argument('--out_dir',   default=None,
                        help='Directory where predicted masks are saved during eval '
                             '(defaults to <work_dir>/pred_masks)')

    # Hardware
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    run_id = datetime.now().strftime('%Y%m%d-%H%M')
    os.makedirs(args.work_dir, exist_ok=True)
    shutil.copyfile(__file__, join(args.work_dir, run_id + '_' + os.path.basename(__file__)))
    print(f'Device: {device}')

    # ---- Build SAM2 ----
    sam2_model = build_sam2(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=device,
        mode='train',
        apply_postprocessing=False,   # postprocessing handled in forward()
    )
    sam2_model.to(device)

    trainable_model = SAM2WithBox(sam2_model).to(device)

    # ---- Freeze image encoder + prompt encoder; train mask decoder only ----
    for param in trainable_model.model.image_encoder.parameters():
        param.requires_grad = False
    for param in trainable_model.model.sam_prompt_encoder.parameters():
        param.requires_grad = False
    for param in trainable_model.model.sam_mask_decoder.parameters():
        param.requires_grad = True

    # Freeze memory/video-specific modules (not used in image-level training)
    for name, param in trainable_model.model.named_parameters():
        if 'memory' in name or 'obj_ptr' in name or 'mask_downsample' in name:
            param.requires_grad = False

    n_total     = sum(p.numel() for p in trainable_model.parameters())
    n_trainable = sum(p.numel() for p in trainable_model.parameters() if p.requires_grad)
    print(f'Total parameters     : {n_total:,}')
    print(f'Trainable parameters : {n_trainable:,}')

    # ---- Grounding DINO (always frozen) ----
    gdino_processor = AutoProcessor.from_pretrained(args.gdino_model)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model).to(device)
    gdino_model.eval()
    for param in gdino_model.parameters():
        param.requires_grad = False
    print(f'Loaded Grounding DINO: {args.gdino_model}')

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainable_model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss  = nn.BCEWithLogitsLoss(reduction='mean')

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # ---- Datasets ----
    train_dataset = NpyDataset(args.frames_dir, args.masks_dir)
    print(f'Training samples : {len(train_dataset)}')

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=False
    )

    val_dataloader = None
    if args.val_frames_dir is not None:
        if args.val_masks_dir is None:
            raise ValueError('--val_masks_dir must be set when --val_frames_dir is provided')
        val_dataset = NpyDataset(args.val_frames_dir, args.val_masks_dir)
        print(f'Validation samples : {len(val_dataset)}')
        val_dataloader = DataLoader(
            val_dataset, batch_size=1,
            shuffle=False, num_workers=args.num_workers, pin_memory=False
        )

    # ---- Pre-cache Grounding DINO boxes ----
    # GDINO is frozen and deterministic, so we run it once per image before
    # training starts and reuse the boxes every epoch. This avoids N_epochs
    # redundant forward passes through GDINO.
    print('Pre-computing Grounding DINO boxes for training set …')
    gdino_box_cache: dict = {}   # img_path -> np.ndarray (4,) float32
    for img_path in tqdm(train_dataset.img_path_files, desc='GDINO cache'):
        pil_img = Image.open(img_path).convert('RGB')
        text_prompt = caption_from_path(img_path)
        gdino_box_cache[img_path] = get_gdino_box(
            pil_img, text_prompt, gdino_model, gdino_processor,
            device, args.box_threshold, args.text_threshold
        )
    print(f'Cached {len(gdino_box_cache)} boxes.')

    out_dir = args.out_dir or join(args.work_dir, 'pred_masks')

    # ---- Resume ----
    start_epoch   = 0
    best_accuracy = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        trainable_model.model.sam_mask_decoder.load_state_dict(ckpt['mask_decoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_accuracy = ckpt.get('best_accuracy', 0.0)
        print(f'Resumed from {args.resume} (epoch {ckpt["epoch"]})')

    losses = []

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        trainable_model.train()
        epoch_loss = 0.0

        for step, (_, gt2D, img_path) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}')):
            optimizer.zero_grad()

            img0 = img_path[0] if isinstance(img_path, (list, tuple)) else img_path
            pil_image  = Image.open(img0).convert('RGB')
            orig_array = np.array(pil_image)
            H_orig, W_orig = orig_array.shape[:2]

            # Resize GT to original image size for loss computation
            gt_orig = F.interpolate(
                gt2D.float(), size=(H_orig, W_orig), mode='nearest'
            ).to(device)

            # Look up pre-cached Grounding DINO box (computed once before training)
            box_xyxy = gdino_box_cache[img0]

            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = trainable_model(orig_array, box_xyxy)
                    loss = seg_loss(pred, gt_orig.float()) + ce_loss(pred, gt_orig.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = trainable_model(orig_array, box_xyxy)
                loss = seg_loss(pred, gt_orig.float()) + ce_loss(pred, gt_orig.float())
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            epoch_loss += loss.item()

        lr_scheduler.step()
        epoch_loss /= max(step, 1)
        losses.append(epoch_loss)

        print(
            f'[{datetime.now().strftime("%Y%m%d-%H%M")}] '
            f'Epoch {epoch}  Loss {epoch_loss:.4f}'
        )

        # ---- Periodic checkpoint ----
        if (epoch + 1) % args.save_every == 0:
            periodic_ckpt = {
                'mask_decoder':  trainable_model.model.sam_mask_decoder.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'epoch':         epoch,
                'best_accuracy': best_accuracy,
            }
            torch.save(periodic_ckpt, join(args.work_dir, f'gdino_sam2_epoch{epoch:03d}.pth'))

        # ---- Validation ----
        if val_dataloader is not None and (epoch + 1) % args.val_every == 0:
            sm, em, wfm, MAE = eval_epoch(
                val_dataloader, trainable_model, gdino_model, gdino_processor,
                device, args.box_threshold, args.text_threshold,
                out_dir=out_dir,
            )
            print({'Sm': sm, 'Em': em, 'wFm': wfm, 'MAE': MAE})

            epoch_accuracy = (sm + em + wfm) / 3.0

            # ---- Save latest ----
            checkpoint = {
                'mask_decoder':  trainable_model.model.sam_mask_decoder.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'epoch':         epoch,
                'best_accuracy': best_accuracy,
            }
            torch.save(checkpoint, join(args.work_dir, 'gdino_sam2_latest.pth'))

            # ---- Save best ----
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                checkpoint['best_accuracy'] = best_accuracy
                torch.save(checkpoint, join(args.work_dir, 'gdino_sam2_best.pth'))
                print(f'  -> New best model saved (accuracy={best_accuracy:.4f})')

        # ---- Loss curve ----
        plt.figure()
        plt.plot(range(start_epoch, start_epoch + len(losses)), losses)
        plt.title('Dice + BCE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(join(args.work_dir, 'gdino_sam2_loss.png'))
        plt.close()

    # ---- Final checkpoint (if we never saved a latest above) ----
    final_ckpt = {
        'mask_decoder':  trainable_model.model.sam_mask_decoder.state_dict(),
        'optimizer':     optimizer.state_dict(),
        'epoch':         args.epochs - 1,
        'best_accuracy': best_accuracy,
    }
    torch.save(final_ckpt, join(args.work_dir, 'gdino_sam2_final.pth'))
    print(f'Training complete. Checkpoints in {args.work_dir}')


if __name__ == '__main__':
    main()
