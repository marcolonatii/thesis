# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict # Added Dict

import strawberry
from app_conf import API_URL
from data.resolver import resolve_videos
from dataclasses_json import dataclass_json
from strawberry import relay


@strawberry.type
class Video(relay.Node):
    """Core type for video."""

    code: relay.NodeID[str]
    path: str
    poster_path: Optional[str]
    width: int
    height: int

    @strawberry.field
    def url(self) -> str:
        return f"{API_URL}/{self.path}"

    @strawberry.field
    def poster_url(self) -> str:
        # Ensure poster_path exists before creating URL
        if self.poster_path:
            return f"{API_URL}/{self.poster_path}"
        return "" # Return empty string or handle as appropriate if no poster

    @classmethod
    def resolve_nodes(
        cls,
        *,
        info: relay.PageInfo,
        node_ids: Iterable[str],
        required: bool = False,
    ):
        return resolve_videos(node_ids, required)


@strawberry.type
class RLEMask:
    """Core type for Onevision GraphQL RLE mask."""

    size: List[int]
    counts: str
    order: str


@strawberry.type
class RLEMaskForObject:
    """Type for RLE mask associated with a specific object id."""

    object_id: int
    rle_mask: RLEMask
    name: Optional[str] = None # Added optional name field


@strawberry.type
class RLEMaskListOnFrame:
    """Type for a list of object-associated RLE masks on a specific video frame."""

    frame_index: int
    rle_mask_list: List[RLEMaskForObject]


@strawberry.input
class DownloadMasksInput:
    """Input type for the downloadMasks mutation."""
    session_id: str


@strawberry.type
class DownloadMasksResponse:
    """Response type for the downloadMasks mutation."""
    masks: List[RLEMaskListOnFrame]


@strawberry.input
class DownloadBoxesInput:
    """Input type for the downloadBoxes mutation."""
    session_id: str
    format: str  # e.g., "yolo"


@strawberry.type
class YOLOBoxForObject:
    """Type for a YOLO bounding box associated with a specific object id."""
    object_id: int
    box: List[float]  # [x_center, y_center, width, height] normalized between 0 and 1
    name: Optional[str] = None # Added optional name field


@strawberry.type
class BoxesListOnFrame:
    """Type for a list of YOLO bounding boxes on a specific video frame."""
    frame_index: int
    boxes: List[YOLOBoxForObject]


@strawberry.type
class DownloadBoxesResponse:
    """Response type for the downloadBoxes mutation."""
    boxes: List[BoxesListOnFrame]


@strawberry.input
class StartSessionInput:
    path: str


@strawberry.type
class StartSession:
    session_id: str


@strawberry.input
class PingInput:
    session_id: str


@strawberry.type
class Pong:
    success: bool


@strawberry.input
class CloseSessionInput:
    session_id: str


@strawberry.type
class CloseSession:
    success: bool


@strawberry.input
class AddPointsInput:
    session_id: str
    frame_index: int
    clear_old_points: bool
    object_id: int
    labels: List[int]
    points: List[List[float]]


@strawberry.input
class ClearPointsInFrameInput:
    session_id: str
    frame_index: int
    object_id: int


@strawberry.input
class ClearPointsInVideoInput:
    session_id: str


@strawberry.type
class ClearPointsInVideo:
    success: bool


@strawberry.input
class RemoveObjectInput:
    session_id: str
    object_id: int


@strawberry.input
class PropagateInVideoInput:
    session_id: str
    start_frame_index: int


@strawberry.input
class CancelPropagateInVideoInput:
    session_id: str


@strawberry.type
class CancelPropagateInVideo:
    success: bool


@strawberry.type
class SessionExpiration:
    session_id: str
    expiration_time: int
    max_expiration_time: int
    ttl: int


@strawberry.type
class SessionInfo:
    """Type representing metadata about an active inference session."""
    session_id: str
    start_time: float  # Timestamp when the session was started
    last_use_time: float  # Timestamp of the last interaction with the session
    num_frames: int  # Number of frames in the session's video
    num_objects: int  # Number of tracked objects in the session

# --- New types for setting object names ---
@strawberry.input
class SetObjectNameInput:
    """Input type for the setObjectName mutation."""
    session_id: str
    object_id: int
    name: str # The new name for the object (empty string clears the custom name)

@strawberry.type
class SetObjectNameResponse:
    """Response type for the setObjectName mutation."""
    success: bool
    object_id: int # Return the affected object_id
    name: Optional[str] # Return the name that was set (or None if cleared)