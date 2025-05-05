# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Dict, Optional, List

import imagesize
from app_conf import DATA_PATH, POSTERS_PATH, POSTERS_PREFIX
from data.data_types import Video
from tqdm import tqdm

logger = logging.getLogger(__name__)


def preload_data() -> Dict[str, Video]:
    """
    Preload data including directories of image frames.
    """
    # Dictionaries for videos and datasets on the backend.
    # Note that since Python 3.7, dictionaries preserve their insert order, so
    # when looping over its `.values()`, elements inserted first also appear first.
    # https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
    all_videos = {}

    # Look for directories containing frame images
    frame_dirs = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    logger.info(f"Found {len(frame_dirs)} frame directories")
    logger.info(f"Frame directories: {frame_dirs}")
    for dir_name in tqdm(frame_dirs):
        dir_path = os.path.join(DATA_PATH, dir_name)
        video = get_video(dir_path, DATA_PATH)
        all_videos[video.code] = video

    logger.info(f"All videos: {all_videos}")
    return all_videos


def get_video(
    dirpath: os.PathLike,
    absolute_path: Path,
    file_key: Optional[str] = None,
    generate_poster: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    verbose: Optional[bool] = False,
) -> Video:
    """
    Get video object from a directory of frame images
    """
    # Use absolute_path to include the parent directory in the video
    dir_rel_path = os.path.relpath(dirpath, absolute_path)
    poster_path = None
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png']
    frame_files = []
    for ext in image_extensions:
        frame_files.extend(glob(os.path.join(dirpath, f"*{ext}")))
    
    # Sort the frame files to ensure consistent ordering
    frame_files.sort()
    
    if not frame_files:
        # No image frames found in directory
        return Video(
            code=dir_rel_path,
            path=dir_rel_path if file_key is None else file_key,
            poster_path=None,
            width=width,
            height=height,
        )
    
    # Use the first frame as the poster
    first_frame = frame_files[0]
    
    if generate_poster:
        poster_id = os.path.basename(dirpath)
        poster_filename = f"{str(poster_id)}.jpg"
        poster_path = f"{POSTERS_PREFIX}/{poster_filename}"

        # Copy the first frame to the posters directory
        poster_output_path = os.path.join(POSTERS_PATH, poster_filename)
        shutil.copy(first_frame, poster_output_path)

        # Get image dimensions
        width, height = imagesize.get(first_frame)
        logger.info(f"Video {dir_rel_path} has dimensions {width}x{height}")
        

    return Video(
        code=dir_rel_path,
        path=dirpath if file_key is None else file_key,
        poster_path=poster_path,
        width=width,
        height=height,
    )
