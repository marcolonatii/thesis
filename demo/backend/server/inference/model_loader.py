# /home/david_elliott/github/sam2/demo/backend/server/inference/model_loader.py
"""
Module for loading the SAM2 video predictor model and handling device configuration.
"""

import contextlib
import logging
import os
from pathlib import Path
import torch
# Ensure build_sam2_video_predictor is correctly imported
# Assuming it's accessible from the installed sam2 package or project structure
from sam2.build_sam import build_sam2_video_predictor
from typing import Tuple

logger = logging.getLogger(__name__)

def select_device(force_cpu: bool = False) -> torch.device:
    """
    Selects the appropriate compute device (CUDA, MPS, or CPU).

    Args:
        force_cpu (bool): If True, forces the use of the CPU even if others are available.

    Returns:
        torch.device: The selected PyTorch device.
    """
    if force_cpu:
        logger.info("Forcing CPU device for SAM 2 demo")
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.get_device_properties(0).major >= 8:
            # Enable TF32 for compatible GPUs (Ampere and later) for better performance
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 on CUDA device.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.warning(
            "Support for MPS devices is preliminary. SAM 2 is trained with CUDA and might give "
            "numerically different outputs and sometimes degraded performance on MPS."
        )
    else:
        device = torch.device("cpu")

    logger.info(f"Selected device: {device}")
    return device

def load_predictor(model_size: str, app_root: str, device: torch.device) -> object:
    """
    Loads and initializes the SAM2 video predictor model.

    Args:
        model_size (str): The size of the model to load ('tiny', 'small', 'base_plus', 'large').
        app_root (str): The root directory path where checkpoints are stored (defined by APP_ROOT env var).
        device (torch.device): The device to load the model onto.

    Returns:
        object: The initialized SAM2 video predictor instance.

    Raises:
        ValueError: If the model_size is invalid.
        FileNotFoundError: If the checkpoint file is not found.
        Exception: For any other model loading errors.
    """
    # Determine checkpoint filename and relative config path string based on model_size
    if model_size == "tiny":
        checkpoint_name = "sam2.1_hiera_tiny.pt"
        # Use the relative path string as in the original code
        model_cfg_relative_path = "configs/sam2.1/sam2.1_hiera_t.yaml"
    elif model_size == "small":
        checkpoint_name = "sam2.1_hiera_small.pt"
        model_cfg_relative_path = "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif model_size == "large":
        checkpoint_name = "sam2.1_hiera_large.pt"
        model_cfg_relative_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif model_size == "base_plus":
        checkpoint_name = "sam2.1_hiera_base_plus.pt"
        model_cfg_relative_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    else:
         raise ValueError(f"Unknown MODEL_SIZE: {model_size}. Choose from 'tiny', 'small', 'base_plus', 'large'.")

    # Construct the absolute path to the checkpoint file using app_root
    # Checkpoints are expected under APP_ROOT/checkpoints/ based on Dockerfile ADD command
    checkpoint_path = Path(app_root) / "checkpoints" / checkpoint_name

    # Verify that the checkpoint file exists
    if not checkpoint_path.exists():
         raise FileNotFoundError(f"Checkpoint file not found at expected location: {checkpoint_path}")
    if not checkpoint_path.is_file():
         raise FileNotFoundError(f"Checkpoint path exists but is not a file: {checkpoint_path}")

    logger.info(f"Initializing SAM2 video predictor (Size: {model_size})")
    logger.info(f"Using checkpoint: {checkpoint_path}")
    # Log the relative config path being passed to the builder
    logger.info(f"Using relative model config path: {model_cfg_relative_path}")

    try:
        # Call the builder function with the relative config path string and absolute checkpoint path
        predictor = build_sam2_video_predictor(model_cfg_relative_path, str(checkpoint_path), device=device)
        logger.info("Successfully initialized SAM2 video predictor")
        return predictor
    except FileNotFoundError as e:
         # Catch potential FileNotFoundError from within build_sam2_video_predictor if it fails to find the config
         logger.error(f"Failed to initialize predictor. Possible issue finding config '{model_cfg_relative_path}' relative to package structure? Error: {e}", exc_info=True)
         raise # Re-raise the specific error
    except Exception as e:
        logger.error(f"Failed to initialize SAM2 video predictor: {e}", exc_info=True)
        raise # Re-raise any other exception during model building

def get_autocast_context(device: torch.device) -> contextlib.AbstractContextManager:
    """
    Returns an automatic mixed-precision context if on CUDA, otherwise a no-op context.

    Args:
        device (torch.device): The compute device being used.

    Returns:
        contextlib.AbstractContextManager: The appropriate autocast context.
    """
    if device.type == "cuda":
        # Use bfloat16 if available, otherwise float16? For now, stick to bfloat16 based on common practice
        logger.debug("Using CUDA autocast context with dtype=bfloat16.")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        logger.debug("Using null context (no autocast).")
        return contextlib.nullcontext()