"""
dinov3_sam2_bridge.py
=====================
Wires DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m) features into SAM2 as a
dense mask prompt via `predictor.add_new_mask(...)`.

Pipeline
--------
frame (H, W, 3)
  │
  ▼
DINOv3FeatureExtractor  ── frozen ──► patch tokens  (B, N, C)
                                      │
                                      ▼  reshape
                                 feature map  (B, C, h, w)   h=H/14, w=W/14
                                      │
                                      ▼
  ┌──────────────────────────────────────────────────────┐
  │  SaliencyBridge                                      │
  │                                                      │
  │  ── ADD YOUR CONVOLUTIONS HERE (in_channels=C) ──    │
  │                                                      │
  │  output: (B, 1, h, w)  logits                       │
  └──────────────────────────────────────────────────────┘
                                      │
                                      ▼  bilinear upsample to (H, W)
                                 saliency  (B, 1, H, W)
                                      │
                                      ▼  squeeze → (H, W)
                         predictor.add_new_mask(
                             inference_state, frame_idx, obj_id, mask)
                                      │
                                      ▼
                         SAM2 MaskDownSampler → dense embedding
                                      │
                                      ▼
                              SAM2 propagation
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DINOv3 feature extractor  (frozen)
# ─────────────────────────────────────────────────────────────────────────────

DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_PATCH_SIZE = 16   # ViT-L/16 patch size (detected from model config at runtime)
DINOV3_EMBED_DIM  = 1024 # ViT-L hidden dim


class DINOv3FeatureExtractor(nn.Module):
    """
    Loads DINOv3 (ViT-L/14), extracts [CLS] + patch tokens, and returns
    the patch tokens reshaped to a 2-D feature map.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.
    freeze : bool
        If True (default) all DINOv3 parameters are frozen.
    """

    def __init__(
        self,
        model_id: str = DINOV3_MODEL_ID,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        # Load model directly as requested by user
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.backbone   = AutoModel.from_pretrained(model_id)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.embed_dim  = self.backbone.config.hidden_size  # 1024 for ViT-L
        # Detect patch size from config (fallback to constant)
        self.patch_size = getattr(
            self.backbone.config, "patch_size", DINOV3_PATCH_SIZE
        )
        self._shape_verified = False  # print token layout once on first forward

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,  # (B, 3, H, W)  pre-normalised
    ) -> torch.Tensor:
        """
        Returns
        -------
        features : (B, C, h, w)
            Spatial patch-token feature map.
            h = H // patch_size,  w = W // patch_size
        """
        B, _, H, W = pixel_values.shape

        with torch.no_grad():
            out = self.backbone(pixel_values=pixel_values, output_hidden_states=False)

        # last_hidden_state: (B, num_special + N, C)
        # DINOv3 prepends CLS + optional register tokens before patch tokens.
        # We take the last h*w tokens, which are always the patch tokens.
        h = H // self.patch_size
        w = W // self.patch_size
        N = h * w
        total_tokens = out.last_hidden_state.shape[1]
        if not self._shape_verified:
            num_special = total_tokens - N
            print(
                f"[DINOv3] token layout: total={total_tokens}, "
                f"patch N={N} ({h}x{w}), special (CLS+registers)={num_special}, "
                f"patch_size={self.patch_size}, input=({H},{W})"
            )
            self._shape_verified = True
        patch_tokens = out.last_hidden_state[:, -N:, :]  # (B, N, C)

        features = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
        return features  # (B, C, h, w)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def preprocess(
        self,
        images: list[Image.Image] | Image.Image,
        device: torch.device | str = "cpu",
        size: Optional[tuple[int, int]] = None,  # (H, W) explicit resize
    ) -> torch.Tensor:
        """
        PIL → normalised tensor using DINO's own mean/std, but WITHOUT
        letting AutoImageProcessor apply its own spatial transforms
        (which silently resize/crop and destroy mask alignment).

        Parameters
        ----------
        images : PIL image or list thereof
        device : target device
        size   : (H, W) to resize every image to before conversion.
                 Always pass this during training so the processed tensor
                 has exactly the same geometry as the GT mask.
        """
        import torchvision.transforms.functional as TF

        if isinstance(images, Image.Image):
            images = [images]

        # Re-use the processor's normalisation statistics without its spatial ops
        mean = self.processor.image_mean  # [0.485, 0.456, 0.406] for DINOv3
        std  = self.processor.image_std   # [0.229, 0.224, 0.225]

        tensors = []
        for img in images:
            if size is not None:
                img = img.resize((size[1], size[0]), Image.BILINEAR)  # PIL: (W, H)
            t = TF.to_tensor(img)                       # (3, H, W) float32 in [0,1]
            t = TF.normalize(t, mean=mean, std=std)
            tensors.append(t)

        pixel_values = torch.stack(tensors, dim=0).to(device)  # (B, 3, H, W)

        # Pad to multiples of patch_size (no-op when size is already a valid multiple)
        _, _, H, W = pixel_values.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h))

        return pixel_values


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Saliency bridge  (placeholder for user's convolutions)
# ─────────────────────────────────────────────────────────────────────────────

class SaliencyBridge(nn.Module):
    """
    Converts a DINOv3 feature map (B, C, h, w) into a saliency map
    (B, 1, H, W) at full frame resolution.

    Steps
    -----
    1. Your convolution stack  ← INSERT YOUR LAYERS HERE
    2. Project to 1 channel
    3. Bilinear upsample to target (H, W)

    Parameters
    ----------
    in_channels : int
        Channel depth of the DINOv3 feature map (1024 for ViT-L).
    target_size : tuple[int, int] | None
        (H, W) of the full frame.  Can also be set at call-time.
    """

    def __init__(
        self,
        in_channels: int = DINOV3_EMBED_DIM,
        target_size: Optional[tuple[int, int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        print(f'Dropout Rate: {dropout}')
        self.target_size = target_size  # (H, W) of the original frame

        self.dino_proj = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.GELU(),
        )

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU()
        )

        self.rgb_proj = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.GELU(),
        )

        self.fuse_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.norm_1 = nn.GroupNorm(num_groups=16, num_channels=256)
        self.gelu_1 = nn.GELU()
        self.dropout_1 = nn.Dropout2d(dropout)

        self.fuse_2 = nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False)
        self.norm_2 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.gelu_2 = nn.GELU()
        self.dropout_2 = nn.Dropout2d(dropout)
        
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=True)
        

    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,               # (B, C, h, w)
        image: torch.Tensor,                  # (B, 3, H, W)
        target_size: Optional[tuple[int, int]] = None,  # (H, W)
    ) -> torch.Tensor:
        """
        Returns
        -------
        saliency : (B, 1, H, W)
            Raw logits.  Positive values → foreground for SAM2.
        """
        _, _, h, w = features.shape

        dino_feat = self.dino_proj(features)    # (B, 256, h, w)

        rgb_feat = self.rgb_encoder(image)            # (B, 128, H, W)
        rgb_feat = F.interpolate(rgb_feat, size=(h, w), mode="bilinear", align_corners=False)  # (B, 128, h, w)
        rgb_feat = self.rgb_proj(rgb_feat)           # (B, 256, h, w)

        x = torch.cat([dino_feat, rgb_feat], dim=1)  # (B, 512, h, w)

        x = self.fuse_1(x)
        x = self.norm_1(x)
        x = self.gelu_1(x)
        #x = self.dropout_1(x)

        x = self.fuse_2(x)
        x = self.norm_2(x)
        x = self.gelu_2(x)
        #x = self.dropout_2(x)

        x = self.out_conv(x)  # (B, 1, h, w)

        size = target_size or self.target_size
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        return x        


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Combined model
# ─────────────────────────────────────────────────────────────────────────────

class DINOv3SAM2Bridge(nn.Module):
    """
    Full bridge: DINOv3 (frozen) + SaliencyBridge (trainable).

    Usage
    -----
    >>> bridge = DINOv3SAM2Bridge(device="cuda")
    >>> pixel_values = bridge.extractor.preprocess(pil_image, device="cuda")
    >>> saliency = bridge(pixel_values, target_size=(H, W))
    >>> mask = saliency[0, 0]           # (H, W)
    >>> _, obj_ids, mask_logits = predictor.add_new_mask(
    ...     inference_state, frame_idx=0, obj_id=1, mask=mask)
    """

    def __init__(
        self,
        model_id: str = DINOV3_MODEL_ID,
        freeze_backbone: bool = True,
        target_size: Optional[tuple[int, int]] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        self.extractor = DINOv3FeatureExtractor(model_id=model_id, freeze=freeze_backbone)
        self.bridge    = SaliencyBridge(
            in_channels=self.extractor.embed_dim,
            target_size=target_size,
        )
        self.to(device)

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,             # (B, 3, H, W)
        target_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Returns saliency (B, 1, H, W).
        """
        features = self.extractor(pixel_values)           # (B, C, h, w)
        saliency = self.bridge(features, image=pixel_values, target_size=target_size)     # (B, 1, H, W)
        return saliency


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Helper: register the saliency map with a SAM2 predictor
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def add_saliency_to_sam2(
    predictor,
    inference_state,
    frame_idx: int,
    obj_id: int,
    saliency: torch.Tensor,  # (1, 1, H, W) or (1, H, W) or (H, W)
) -> tuple:
    """
    Wraps `predictor.add_new_mask()` to accept the bridge output directly.

    Parameters
    ----------
    predictor       : SAM2VideoPredictor
    inference_state : object returned by predictor.init_state(...)
    frame_idx       : frame index in the video
    obj_id          : integer object id
    saliency        : raw saliency logits from DINOv3SAM2Bridge

    Returns
    -------
    (frame_idx, obj_ids, mask_logits) — same as add_new_mask
    """
    # Normalise shape to (H, W)
    mask = saliency
    while mask.dim() > 2:
        mask = mask.squeeze(0)

    # SAM2 add_new_mask expects a float or bool mask on the correct device
    # Positive values = foreground
    return predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask=mask.float(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  End-to-end pipeline example
# ─────────────────────────────────────────────────────────────────────────────

def run_bridge_pipeline(
    frame_paths: list[Path | str],
    predictor,                            # SAM2VideoPredictor
    bridge: DINOv3SAM2Bridge,
    video_dir: Path | str,                # directory passed to init_state
    prompt_frame_idx: int = 0,            # which frame to compute saliency on
    obj_id: int = 1,
    device: str | torch.device = "cpu",
) -> dict[int, torch.Tensor]:
    """
    Minimal example pipeline:
      1. Compute saliency on `prompt_frame_idx`
      2. Register as dense mask prompt with SAM2
      3. Propagate through the whole video
      4. Return {frame_idx: mask (H, W)}

    Returns
    -------
    results : dict mapping frame_idx → binary mask (H, W, bool)
    """
    import torchvision.transforms.functional as TF

    # ── 1. Load prompt frame and compute saliency ──────────────────────
    pil_img = Image.open(frame_paths[prompt_frame_idx]).convert("RGB")
    H, W = pil_img.height, pil_img.width

    pixel_values = bridge.extractor.preprocess(pil_img, device=device)
    # Use the padded tensor dims as target_size so the feature map is not
    # interpolated from a mismatched resolution, then crop back to (H, W).
    pH, pW = pixel_values.shape[-2], pixel_values.shape[-1]
    bridge.eval()
    with torch.no_grad():
        saliency = bridge(pixel_values, target_size=(pH, pW))  # (1, 1, pH, pW)
        saliency = saliency[..., :H, :W]                       # crop padding → (1, 1, H, W)

    # ── 2. Init SAM2 inference state ────────────────────────────────────
    inference_state = predictor.init_state(video_path=str(video_dir))
    predictor.reset_state(inference_state)

    # ── 3. Register saliency as dense mask prompt ────────────────────────
    add_saliency_to_sam2(
        predictor, inference_state,
        frame_idx=prompt_frame_idx,
        obj_id=obj_id,
        saliency=saliency,
    )

    # ── 4. Propagate ─────────────────────────────────────────────────────
    results: dict[int, torch.Tensor] = {}
    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        # mask_logits: (num_objs, 1, H, W)
        mask = (mask_logits[0, 0] > 0.0)  # bool (H, W)
        results[frame_idx] = mask.cpu()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (no SAM2 needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Loading DINOv3SAM2Bridge …")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bridge = DINOv3SAM2Bridge(device=device)
    print(f"  backbone embed_dim : {bridge.extractor.embed_dim}")
    print(f"  patch_size         : {bridge.extractor.patch_size}")

    # Dummy image
    dummy = Image.fromarray(
        torch.randint(0, 255, (448, 448, 3), dtype=torch.uint8).numpy()
    )
    pixel_values = bridge.extractor.preprocess(dummy, device=device)
    print(f"  pixel_values shape : {pixel_values.shape}")

    bridge.eval()
    with torch.no_grad():
        saliency = bridge(pixel_values, target_size=(448, 448))
    print(f"  saliency shape     : {saliency.shape}")   # expect (1, 1, 448, 448)
    print("Sanity check passed.")
    print()
    print("NOTE: SaliencyBridge.proj is currently nn.Identity() — add your")
    print("      convolutions in the SaliencyBridge class before training.")
