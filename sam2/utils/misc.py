# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from threading import Thread

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import decord


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from sam2 import _C

    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


class VideoFrameLoader:
    """
    Loads video frames from a video file using decord.
    """
    def __init__(
        self,
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        self.video_path = video_path
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.compute_device = compute_device
        self.video_height = None
        self.video_width = None

        # Load first frame to get video size
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path)
        first_frame = vr[0]
        self.video_height, self.video_width, _ = first_frame.shape
        self._vr = vr

    def __getitem__(self, index):
        frame = self._vr[index].permute(2, 0, 1).float() / 255.0
        img = frame
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        return img

    def __len__(self):
        return len(self._vr)


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
):
    """
    Load the video frames from video_path. The frames are resized to image_size as in
    the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
    """
    is_mp4_path = video_path.lower().endswith(".mp4")
    if is_mp4_path:
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        images = VideoFrameLoader(
            video_path,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
        )
        return images, images.video_height, images.video_width
    else:
        raise NotImplementedError(
            "Only MP4 video is supported"
        )


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    # Holes are those connected components in background with area <= self.max_area
    # (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"

    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        # We fill holes with a small positive mask score (0.1) to change them to foreground.
        mask = torch.where(is_hole, 0.1, mask)
    except Exception as e:
        # Skip the post-processing step on removing small holes if the CUDA kernel fails
        warnings.warn(
            f"{e}\n\nSkipping the post-processing step due to the error above. You can "
            "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
            "functionality may be limited (which doesn't affect the results in most cases; see "
            "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}
