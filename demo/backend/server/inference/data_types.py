# /home/david_elliott/github/sam2/demo/backend/server/inference/data_types.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from dataclasses_json import dataclass_json
from torch import Tensor


@dataclass_json
@dataclass
class Mask:
    size: List[int]
    counts: str


@dataclass_json
@dataclass
class BaseRequest:
    type: str


@dataclass_json
@dataclass
class StartSessionRequest(BaseRequest):
    type: str
    path: str
    session_id: Optional[str] = None


@dataclass_json
@dataclass
class SaveSessionRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class LoadSessionRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class RenewSessionRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class CloseSessionRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class AddPointsRequest(BaseRequest):
    type: str
    session_id: str
    frame_index: int
    clear_old_points: bool
    object_id: int
    labels: List[int]
    points: List[List[float]]


@dataclass_json
@dataclass
class AddMaskRequest(BaseRequest):
    type: str
    session_id: str
    frame_index: int
    object_id: int
    mask: Mask


@dataclass_json
@dataclass
class ClearPointsInFrameRequest(BaseRequest):
    type: str
    session_id: str
    frame_index: int
    object_id: int


@dataclass_json
@dataclass
class ClearPointsInVideoRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class RemoveObjectRequest(BaseRequest):
    type: str
    session_id: str
    object_id: int


@dataclass_json
@dataclass
class PropagateInVideoRequest(BaseRequest):
    type: str
    session_id: str
    start_frame_index: int


@dataclass_json
@dataclass
class CancelPropagateInVideoRequest(BaseRequest):
    type: str
    session_id: str


# --- NEW: Set Object Name Request ---
@dataclass_json
@dataclass
class SetObjectNameRequest(BaseRequest):
    type: str
    session_id: str
    object_id: int
    name: str


@dataclass_json
@dataclass
class StartSessionResponse:
    session_id: str


@dataclass_json
@dataclass
class SaveSessionResponse:
    session_id: str


@dataclass_json
@dataclass
class LoadSessionResponse:
    session_id: str


@dataclass_json
@dataclass
class RenewSessionResponse:
    session_id: str


@dataclass_json
@dataclass
class CloseSessionResponse:
    success: bool


@dataclass_json
@dataclass
class ClearPointsInVideoResponse:
    success: bool


@dataclass_json
@dataclass
class PropagateDataValue:
    object_id: int
    mask: Mask
    # Note: Name is added at the GraphQL layer from persisted names


@dataclass_json
@dataclass
class PropagateDataResponse:
    frame_index: int
    results: List[PropagateDataValue]


@dataclass_json
@dataclass
class RemoveObjectResponse:
    results: List[PropagateDataResponse]


@dataclass_json
@dataclass
class CancelPorpagateResponse:
    success: bool


# --- NEW: Set Object Name Response ---
@dataclass_json
@dataclass
class SetObjectNameResponse:
    success: bool
    object_id: int
    name: str # Return the name that was actually set


# <<< Added ClickData definition >>>
@dataclass_json
@dataclass
class ClickData:
    """Represents user click data for a specific frame and object."""
    frame_index: int
    object_id: int
    points: List[List[float]] # Coordinates [[x1, y1], [x2, y2], ...]
    labels: List[int]          # Click labels (e.g., 1 for positive, 0 for negative)


@dataclass_json
@dataclass
class InferenceSession:
    """Represents the state of an active inference session in memory (simplified)."""
    start_time: float
    last_use_time: float
    session_id: str
    # Note: The actual 'state' is complex and managed within InferenceAPI/predictor,
    # this definition might be used for basic session tracking if needed elsewhere.
    # state: Dict[str, Dict[str, Union[Tensor, Dict[int, Tensor]]]] # Example structure


@dataclass_json
@dataclass
class DownloadMasksRequest(BaseRequest):
    """
    Inference request type for downloading masks.
    Contains the session ID for which masks are requested.
    """
    type: str
    session_id: str

@dataclass_json
@dataclass
class DownloadMasksResponse:
    """Response containing masks for multiple frames."""
    results: List[PropagateDataResponse]