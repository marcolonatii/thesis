# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging # Added import
from typing import Dict

from data.data_types import Video

logger = logging.getLogger(__name__) # Added logger

ALL_VIDEOS: Dict[str, Video] = {} # Initialized as empty dict


def set_videos(videos: Dict[str, Video]) -> None:
    """
    Set the initial videos available in the backend (typically at startup).
    The data is kept in-memory.
    """
    global ALL_VIDEOS
    ALL_VIDEOS = videos
    logger.info(f"Initialized video store with {len(ALL_VIDEOS)} videos.")


# Removed the add_video function


def get_videos() -> Dict[str, Video]:
    """
    Return all videos available in the backend store.
    """
    global ALL_VIDEOS
    return ALL_VIDEOS