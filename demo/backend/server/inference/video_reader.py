# /home/david_elliott/github/sam2/demo/backend/server/inference/video_reader.py
"""
Module for reading frames from video files using OpenCV.
"""

import logging
import cv2
from pathlib import Path
from typing import List
import numpy as np
import os
import glob

logger = logging.getLogger(__name__)

def read_raw_frames(relative_path: str, data_path: Path) -> List[np.ndarray]:
    """
    Read raw frames from a video file or directory of frame images using OpenCV.

    Args:
        relative_path (str): Relative path to the video (e.g., 'gallery/video.mp4')
                            or directory containing frames.
        data_path (Path): The base data directory path.

    Returns:
        List[np.ndarray]: List of frames (as BGR NumPy arrays).

    Raises:
        FileNotFoundError: If the video file or directory does not exist.
        RuntimeError: If the frames cannot be read.
    """
    full_path = data_path / relative_path
    if not full_path.exists():
        logger.error(f"Path not found: {full_path}")
        raise FileNotFoundError(f"Path not found: {full_path}")

    frames = []
    
    # Check if it's a directory (containing frame images)
    if full_path.is_dir():
        logger.info(f"Reading frames from directory: {full_path}")
        # Find all image files in the directory
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(glob.glob(str(full_path / f"*{ext}")))
            image_files.extend(glob.glob(str(full_path / f"*{ext.upper()}")))  # Also look for uppercase extensions
        
        # Sort the files to ensure correct order
        image_files.sort()
        
        if not image_files:
            logger.warning(f"No image files found in directory: {full_path}")
            return []
        
        # Read each image file
        for img_path in image_files:
            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    logger.warning(f"Failed to read image: {img_path}")
                    continue
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Error reading image {img_path}: {e}")
                # Continue with other images rather than failing completely
        
        logger.info(f"Loaded {len(frames)} frames from directory {full_path}")
    
    # It's a video file
    else:
        logger.info(f"Reading frames from video file: {full_path}")
        cap = cv2.VideoCapture(str(full_path))
        if not cap.isOpened():
            logger.error(f"Could not open video at {full_path}")
            raise RuntimeError(f"Could not open video file: {full_path}")

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
        except Exception as e:
            logger.error(f"Error reading frames from video {full_path}: {e}", exc_info=True)
            raise RuntimeError(f"Error reading video file {full_path}: {e}") from e
        finally:
            cap.release()

        logger.info(f"Loaded {frame_count} raw frames from video {full_path}")
        
    if not frames:
        logger.warning(f"No frames were read from {full_path}")

    return frames