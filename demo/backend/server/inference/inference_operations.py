# /home/delliott@int-episci.com/sam2/demo/backend/server/inference/inference_operations.py
"""
Module containing inference operations that modify the segmentation state.
This includes adding prompts, clearing points, removing objects, and propagating masks.
"""

from typing import Generator
import torch
from inference.data_types import (
    PropagateDataResponse,
    ClearPointsInVideoResponse,
    RemoveObjectResponse,
    CancelPorpagateResponse,
)
from pycocotools.mask import decode as decode_masks
from .mask_utils import get_rle_mask_list

def add_points_operation(predictor, inference_state, request, score_thresh=0) -> PropagateDataResponse:
    """
    Add point prompts to a specific frame and return updated masks.

    Args:
        predictor: The SAM predictor instance.
        inference_state: The current inference state.
        request: An instance of AddPointsRequest.
        score_thresh: Score threshold to binarize masks.

    Returns:
        PropagateDataResponse: Updated masks for the frame.
    """
    frame_idx = request.frame_index
    obj_id = request.object_id
    points = request.points
    labels = request.labels
    clear_old_points = request.clear_old_points

    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
        clear_old_points=clear_old_points,
        normalize_coords=False,
    )
    masks_binary = (masks > score_thresh)[:, 0].cpu().numpy()
    rle_mask_list = get_rle_mask_list(object_ids, masks_binary)
    return PropagateDataResponse(frame_index=frame_idx, results=rle_mask_list)

def add_mask_operation(predictor, inference_state, request, score_thresh=0) -> PropagateDataResponse:
    """
    Add a mask prompt directly and update the segmentation mask.

    Args:
        predictor: The SAM predictor instance.
        inference_state: The current inference state.
        request: An instance of AddMaskRequest.
        score_thresh: Score threshold to binarize masks.

    Returns:
        PropagateDataResponse: Updated mask for the frame.
    """
    frame_idx = request.frame_index
    obj_id = request.object_id
    rle_mask = {"counts": request.mask.counts, "size": request.mask.size}
    mask = decode_masks(rle_mask)
    frame_idx, obj_ids, video_res_masks = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask=torch.tensor(mask > 0),
    )
    masks_binary = (video_res_masks > score_thresh)[:, 0].cpu().numpy()
    rle_mask_list = get_rle_mask_list(obj_ids, masks_binary)
    return PropagateDataResponse(frame_index=frame_idx, results=rle_mask_list)

def clear_points_in_frame_operation(predictor, inference_state, request, score_thresh=0) -> PropagateDataResponse:
    """
    Clear point prompts in a single frame for a given object.

    Args:
        predictor: The SAM predictor instance.
        inference_state: The current inference state.
        request: An instance of ClearPointsInFrameRequest.
        score_thresh: Score threshold to binarize masks.

    Returns:
        PropagateDataResponse: Updated mask for the frame.
    """
    frame_idx = request.frame_index
    obj_id = request.object_id
    frame_idx, obj_ids, video_res_masks = predictor.clear_all_prompts_in_frame(
        inference_state, frame_idx, obj_id
    )
    masks_binary = (video_res_masks > score_thresh)[:, 0].cpu().numpy()
    rle_mask_list = get_rle_mask_list(obj_ids, masks_binary)
    return PropagateDataResponse(frame_index=frame_idx, results=rle_mask_list)

def clear_points_in_video_operation(predictor, inference_state, request) -> ClearPointsInVideoResponse:
    """
    Clear all point prompts across all frames.

    Args:
        predictor: The SAM predictor instance.
        inference_state: The current inference state.
        request: An instance of ClearPointsInVideoRequest.

    Returns:
        ClearPointsInVideoResponse: Confirmation of the clearing operation.
    """
    predictor.reset_state(inference_state)
    from inference.data_types import ClearPointsInVideoResponse
    return ClearPointsInVideoResponse(success=True)

def remove_object_operation(predictor, inference_state, request, score_thresh=0) -> RemoveObjectResponse:
    """
    Remove an object from the segmentation state.

    Args:
        predictor: The SAM predictor instance.
        inference_state: The current inference state.
        request: An instance of RemoveObjectRequest.
        score_thresh: Score threshold to binarize masks.

    Returns:
        RemoveObjectResponse: Updated masks after object removal.
    """
    new_obj_ids, updated_frames = predictor.remove_object(inference_state, request.object_id)
    results = []
    for frame_index, video_res_masks in updated_frames:
        masks_binary = (video_res_masks > score_thresh)[:, 0].cpu().numpy()
        rle_mask_list = get_rle_mask_list(new_obj_ids, masks_binary)
        results.append(PropagateDataResponse(frame_index=frame_index, results=rle_mask_list))
    from inference.data_types import RemoveObjectResponse
    return RemoveObjectResponse(results=results)

def propagate_in_video_operation(predictor, inference_state, request, session, score_thresh=0) -> Generator[PropagateDataResponse, None, None]:
    """
    Propagate existing prompts throughout the video and yield updated masks.

    Args:
        predictor: The SAM predictor instance.
        inference_state: The current inference state.
        request: An instance of PropagateInVideoRequest.
        session: The session dictionary containing the inference state.
        score_thresh: Score threshold to binarize masks.

    Yields:
        PropagateDataResponse: Updated masks for each processed frame.
    """
    start_frame_idx = request.start_frame_index
    propagation_direction = "both"
    max_frame_num_to_track = None

    if "masks_cache" not in inference_state:
        inference_state["masks_cache"] = {}

    # Forward propagation
    if propagation_direction in ["both", "forward"]:
        for outputs in predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
        ):
            if session.get("canceled", False):
                return
            frame_idx, obj_ids, video_res_masks = outputs
            masks_binary = (video_res_masks > score_thresh)[:, 0].cpu().numpy()
            rle_mask_list = get_rle_mask_list(obj_ids, masks_binary)
            response = PropagateDataResponse(frame_index=frame_idx, results=rle_mask_list)
            inference_state["masks_cache"][frame_idx] = response
            yield response

    # Backward propagation
    if propagation_direction in ["both", "backward"]:
        for outputs in predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=True,
        ):
            if session.get("canceled", False):
                return
            frame_idx, obj_ids, video_res_masks = outputs
            masks_binary = (video_res_masks > score_thresh)[:, 0].cpu().numpy()
            rle_mask_list = get_rle_mask_list(obj_ids, masks_binary)
            response = PropagateDataResponse(frame_index=frame_idx, results=rle_mask_list)
            inference_state["masks_cache"][frame_idx] = response
            yield response

def cancel_propagate_in_video_operation(session, request) -> CancelPorpagateResponse:
    """
    Cancel an ongoing propagation process.

    Args:
        session: The session dictionary.
        request: An instance of CancelPropagateInVideoRequest.

    Returns:
        CancelPorpagateResponse: Response confirming the cancellation.
    """
    session["canceled"] = True
    return CancelPorpagateResponse(success=True)