# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import av
import strawberry
from app_conf import (
    DATA_PATH,
    DEFAULT_VIDEO_PATH,
    MAX_UPLOAD_VIDEO_DURATION,
    UPLOADS_PATH,
    UPLOADS_PREFIX,
)
from data.data_types import (
    AddPointsInput,
    CancelPropagateInVideo,
    CancelPropagateInVideoInput,
    ClearPointsInFrameInput,
    ClearPointsInVideo,
    ClearPointsInVideoInput,
    CloseSession,
    CloseSessionInput,
    DownloadMasksInput,
    DownloadMasksResponse,
    DownloadBoxesInput,
    YOLOBoxForObject,
    BoxesListOnFrame,
    DownloadBoxesResponse,
    RemoveObjectInput,
    RLEMask,
    RLEMaskForObject,
    RLEMaskListOnFrame,
    SessionInfo,
    SetObjectNameInput,
    SetObjectNameResponse,
    StartSession,
    StartSessionInput,
    Video,
)
from data.loader import get_video
from data.store import get_videos, set_videos
from data.transcoder import get_video_metadata, transcode, VideoMetadata
from inference.data_types import (
    AddPointsRequest,
    CancelPropagateInVideoRequest,
    ClearPointsInFrameRequest,
    ClearPointsInVideoRequest,
    CloseSessionRequest,
    DownloadMasksRequest,
    DownloadMasksResponse as InferenceDownloadMasksResponse,
    PropagateDataResponse,
    RemoveObjectRequest,
    SetObjectNameRequest,
    SetObjectNameResponse as InferenceSetObjectNameResponse,
    StartSessionRequest,
)
from inference.api import InferenceAPI
from strawberry import relay
from strawberry.file_uploads import Upload
from strawberry.exceptions import StrawberryGraphQLError
from tqdm import tqdm
import pycocotools.mask as maskUtils
from inference.session_manager import load_object_names

logger = logging.getLogger(__name__)


@strawberry.type
class Query:
    @strawberry.field
    def default_video(self) -> Video:
        """
        Return the default video.

        The default video can be set with the DEFAULT_VIDEO_PATH environment
        variable. It will return the video that matches this path. If no video
        is found, it will return the first video.
        """
        all_videos = get_videos()
        if not all_videos:
            raise ValueError("No videos available in the store.")

        if DEFAULT_VIDEO_PATH:
            for _, v in all_videos.items():
                if v.path == DEFAULT_VIDEO_PATH:
                    return v
            logger.warning(f"Default video path '{DEFAULT_VIDEO_PATH}' not found.")

        return next(iter(all_videos.values()))

    @relay.connection(relay.ListConnection[Video])
    def videos(
        self,
    ) -> Iterable[Video]:
        """
        Return all available videos from the store.
        """
        all_videos = get_videos()
        logger.info(f"Returning {len(all_videos)} videos")
        logger.info(f"Videos: {all_videos}")
        return all_videos.values()

    @strawberry.field
    def sessions(self, info: strawberry.Info) -> List[SessionInfo]:
        """
        Return a list of all active inference sessions.

        Args:
            info: Strawberry context info containing the inference_api.

        Returns:
            List[SessionInfo]: A list of metadata for each active session.
        """
        inference_api: InferenceAPI = info.context["inference_api"]
        return inference_api.list_sessions()


@strawberry.type
class Mutation:
    @strawberry.mutation
    def upload_video(
        self,
        file: Upload,
        start_time_sec: Optional[float] = None,
        duration_time_sec: Optional[float] = None,
    ) -> Video:
        """
        Receive a video file, process it (transcode, generate poster),
        store it, add it to the video store, and return its metadata.
        """
        logger.info(f"Received video upload: {file.filename}")
        max_time = MAX_UPLOAD_VIDEO_DURATION
        try:
            filepath, file_key, vm = process_video(
                file,
                max_time=max_time,
                start_time_sec=start_time_sec,
                duration_time_sec=duration_time_sec,
            )

            video = get_video(
                filepath=filepath,
                absolute_path=UPLOADS_PATH,
                file_key=file_key,
                width=vm.width,
                height=vm.height,
                generate_poster=True,
            )

            # Instead of add_video, get current videos, add to dict, and set back
            current_videos = get_videos()
            current_videos[video.code] = video
            set_videos(current_videos)
            logger.info(f"Successfully processed and stored video: {video.code}. Updated video store.")
            return video
        except Exception as e:
            logger.exception(f"Failed to process uploaded video {file.filename}")
            raise StrawberryGraphQLError(f"Failed to process video: {str(e)}") # Updated

    @strawberry.mutation
    def start_session(
        self, input: StartSessionInput, info: strawberry.Info
    ) -> StartSession:
        """
        Start an inference session for a video specified by its relative path.
        The path should correspond to the 'path' attribute of a Video object
        (e.g., 'gallery/video1.mp4' or 'uploads/hash.mp4').
        """
        inference_api: InferenceAPI = info.context["inference_api"]
        logger.info(f"Attempting to start session for video path: {input.path}")

        full_video_path = Path(DATA_PATH) / input.path
        if not full_video_path.exists():
            logger.error(f"Video file not found at expected location: {full_video_path}")
            raise StrawberryGraphQLError(f"Video path '{input.path}' not found on server.") # Updated

        request = StartSessionRequest(
            type="start_session",
            path=input.path,
        )

        try:
            response = inference_api.start_session(request=request)
            logger.info(f"Successfully started session {response.session_id} for path {input.path}")
            return StartSession(session_id=response.session_id)
        except Exception as e:
            logger.exception(f"Failed to start inference session for path {input.path}")
            raise StrawberryGraphQLError(f"Failed to start session: {str(e)}") # Updated

    @strawberry.mutation
    def close_session(
        self, input: CloseSessionInput, info: strawberry.Info
    ) -> CloseSession:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = CloseSessionRequest(
            type="close_session",
            session_id=input.session_id,
        )
        response = inference_api.close_session(request)
        return CloseSession(success=response.success)

    @strawberry.mutation
    def add_points(
        self, input: AddPointsInput, info: strawberry.Info
    ) -> RLEMaskListOnFrame:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = AddPointsRequest(
            type="add_points",
            session_id=input.session_id,
            frame_index=input.frame_index,
            object_id=input.object_id,
            points=input.points,
            labels=input.labels,
            clear_old_points=input.clear_old_points,
        )
        response = inference_api.add_points(request)

        # Get current names for the session to include in the response
        object_names = load_object_names(input.session_id)

        return RLEMaskListOnFrame(
            frame_index=response.frame_index,
            rle_mask_list=[
                RLEMaskForObject(
                    object_id=r.object_id,
                    rle_mask=RLEMask(counts=r.mask.counts, size=r.mask.size, order="F"),
                    name=object_names.get(r.object_id) # Add name here
                )
                for r in response.results
            ],
        )

    @strawberry.mutation
    def remove_object(
        self, input: RemoveObjectInput, info: strawberry.Info
    ) -> List[RLEMaskListOnFrame]:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = RemoveObjectRequest(
            type="remove_object", session_id=input.session_id, object_id=input.object_id
        )

        response = inference_api.remove_object(request)
        # Get current names for the session *after* removal
        object_names = load_object_names(input.session_id)

        return [
            RLEMaskListOnFrame(
                frame_index=res.frame_index,
                rle_mask_list=[
                    RLEMaskForObject(
                        object_id=r.object_id,
                        rle_mask=RLEMask(
                            counts=r.mask.counts, size=r.mask.size, order="F"
                        ),
                        name=object_names.get(r.object_id) # Add name here
                    )
                    for r in res.results
                ],
            )
            for res in response.results
        ]

    @strawberry.mutation
    def clear_points_in_frame(
        self, input: ClearPointsInFrameInput, info: strawberry.Info
    ) -> RLEMaskListOnFrame:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = ClearPointsInFrameRequest(
            type="clear_points_in_frame",
            session_id=input.session_id,
            frame_index=input.frame_index,
            object_id=input.object_id,
        )

        response = inference_api.clear_points_in_frame(request)
        # Get current names for the session
        object_names = load_object_names(input.session_id)

        return RLEMaskListOnFrame(
            frame_index=response.frame_index,
            rle_mask_list=[
                RLEMaskForObject(
                    object_id=r.object_id,
                    rle_mask=RLEMask(counts=r.mask.counts, size=r.mask.size, order="F"),
                    name=object_names.get(r.object_id) # Add name here
                )
                for r in response.results
            ],
        )

    @strawberry.mutation
    def clear_points_in_video(
        self, input: ClearPointsInVideoInput, info: strawberry.Info
    ) -> ClearPointsInVideo:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = ClearPointsInVideoRequest(
            type="clear_points_in_video",
            session_id=input.session_id,
        )
        response = inference_api.clear_points_in_video(request)
        return ClearPointsInVideo(success=response.success)

    @strawberry.mutation
    def cancel_propagate_in_video(
        self, input: CancelPropagateInVideoInput, info: strawberry.Info
    ) -> CancelPropagateInVideo:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = CancelPropagateInVideoRequest(
            type="cancel_propagate_in_video",
            session_id=input.session_id,
        )
        response = inference_api.cancel_propagate_in_video(request)
        return CancelPropagateInVideo(success=response.success)

    # --- NEW: setObjectName Mutation ---
    @strawberry.mutation
    def set_object_name(
        self, input: SetObjectNameInput, info: strawberry.Info
    ) -> SetObjectNameResponse:
        """
        Sets or clears the custom name for a tracked object within a session.
        """
        inference_api: InferenceAPI = info.context["inference_api"]
        logger.info(f"Received setObjectName request for session {input.session_id}, object {input.object_id}, name '{input.name}'")

        request = SetObjectNameRequest(
            type="set_object_name",
            session_id=input.session_id,
            object_id=input.object_id,
            name=input.name,
        )

        try:
            response = inference_api.set_object_name(request=request)
            logger.info(f"Successfully set name for object {response.object_id} in session {input.session_id} to '{response.name}'")
            # Map InferenceSetObjectNameResponse to GraphQL SetObjectNameResponse
            return SetObjectNameResponse(
                success=response.success,
                object_id=response.object_id,
                name=response.name if response.name else None # Return None if name is empty string
            )
        except RuntimeError as e:
            logger.error(f"Failed to set object name for session {input.session_id}, object {input.object_id}: {e}", exc_info=True)
            # Check for specific errors like session not found
            if "Cannot find session" in str(e):
                raise StrawberryGraphQLError(f"Session '{input.session_id}' not found.")
            else:
                raise StrawberryGraphQLError(f"Failed to set object name: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error in setObjectName for session {input.session_id}, object {input.object_id}")
            raise StrawberryGraphQLError(f"An unexpected error occurred while setting the object name: {str(e)}")

    # --- Updated Download Mutations ---
    @strawberry.mutation
    def download_masks(
        self, input: DownloadMasksInput, info: strawberry.Info
    ) -> DownloadMasksResponse:
        """
        Retrieve all masks for all frames in the specified session, including object names.
        """
        inference_api: InferenceAPI = info.context["inference_api"]

        request = DownloadMasksRequest(
            type="download_masks",
            session_id=input.session_id,
        )
        # This call now returns masks and potentially updates the cache
        response: InferenceDownloadMasksResponse = inference_api.download_masks(request)

        # Load names *after* potentially generating/caching masks
        object_names = load_object_names(input.session_id)

        gql_masks = []
        for frame_result in response.results:
            gql_rle_list = []
            for mask_data in frame_result.results:
                gql_rle_list.append(
                    RLEMaskForObject(
                        object_id=mask_data.object_id,
                        rle_mask=RLEMask(
                            counts=mask_data.mask.counts,
                            size=mask_data.mask.size,
                            order="F",
                        ),
                        name=object_names.get(mask_data.object_id) # Add name here
                    )
                )
            gql_masks.append(
                RLEMaskListOnFrame(
                    frame_index=frame_result.frame_index, rle_mask_list=gql_rle_list
                )
            )

        return DownloadMasksResponse(masks=gql_masks)

    @strawberry.mutation
    def download_boxes(
        self, input: DownloadBoxesInput, info: strawberry.Info
    ) -> DownloadBoxesResponse:
        """
        Retrieve bounding boxes derived from masks for all frames in the specified session
        in the requested format (currently only YOLO), including object names.
        """
        inference_api: InferenceAPI = info.context["inference_api"]
        logger.info(f"Processing download_boxes request for session {input.session_id} in format {input.format}")

        if input.format.lower() != "yolo":
            raise StrawberryGraphQLError(f"Unsupported format '{input.format}'. Only 'yolo' is supported.") # Updated

        # Get masks first (this ensures they are generated/cached)
        mask_request = DownloadMasksRequest(
            type="download_masks",
            session_id=input.session_id,
        )
        masks_response: InferenceDownloadMasksResponse = inference_api.download_masks(request=mask_request)

        # Load object names
        object_names = load_object_names(input.session_id)

        def rle_to_yolo(rle_mask_data):
            try:
                rle_dict = {"counts": rle_mask_data.counts.encode('utf-8'), "size": rle_mask_data.size}
                bbox = maskUtils.toBbox(rle_dict)
                x, y, w, h = bbox
                img_h, img_w = rle_mask_data.size
                if img_w <= 0 or img_h <= 0:
                    logger.warning(f"Invalid image dimensions in RLE mask: w={img_w}, h={img_h}. Cannot compute YOLO box.")
                    return None
                x_center = x + w / 2.0
                y_center = y + h / 2.0
                return [x_center / img_w, y_center / img_h, w / img_w, h / img_h]
            except Exception as e:
                logger.error(f"Error converting RLE to Bbox: {e}. RLE counts: {rle_mask_data.counts[:50]}..., size: {rle_mask_data.size}")
                return None

        boxes_response = []
        for frame_result in masks_response.results:
            boxes_for_frame = []
            for mask_data in frame_result.results:
                yolo_box = rle_to_yolo(mask_data.mask)
                if yolo_box is not None:
                    boxes_for_frame.append(
                        YOLOBoxForObject(
                            object_id=mask_data.object_id,
                            box=yolo_box,
                            name=object_names.get(mask_data.object_id) # Add name here
                        )
                    )
                else:
                    logger.warning(f"Skipping box for object {mask_data.object_id} on frame {frame_result.frame_index} due to conversion error.")

            boxes_response.append(
                BoxesListOnFrame(frame_index=frame_result.frame_index, boxes=boxes_for_frame)
            )

        logger.info(f"Generated YOLO boxes for {len(boxes_response)} frames for session {input.session_id}")
        return DownloadBoxesResponse(boxes=boxes_response)


# --- Helper Functions (Unchanged) ---

def get_file_hash(video_path_or_file) -> str:
    """Calculate SHA256 hash of a file or file-like object."""
    hasher = hashlib.sha256()
    chunk_size = 65536
    if isinstance(video_path_or_file, str):
        try:
            with open(video_path_or_file, "rb") as in_f:
                while True:
                    chunk = in_f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except FileNotFoundError:
            logger.error(f"File not found for hashing: {video_path_or_file}")
            raise
    else:
        try:
            original_pos = video_path_or_file.tell()
            video_path_or_file.seek(0)
            while True:
                chunk = video_path_or_file.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
            video_path_or_file.seek(original_pos)
        except Exception as e:
            logger.error(f"Error reading file-like object for hashing: {e}")
            raise
    return hasher.hexdigest()


def _get_start_sec_duration_sec(
    start_time_sec: Union[float, None],
    duration_time_sec: Union[float, None],
    max_time: float,
) -> Tuple[float, float]:
    default_seek_t = int(os.environ.get("VIDEO_ENCODE_SEEK_TIME", "0"))
    if start_time_sec is None:
        start_time_sec = float(default_seek_t)

    if duration_time_sec is not None:
        duration_time_sec = min(float(duration_time_sec), max_time)
    else:
        duration_time_sec = max_time
    return start_time_sec, duration_time_sec


def process_video(
    file: Upload,
    max_time: float,
    start_time_sec: Optional[float] = None,
    duration_time_sec: Optional[float] = None,
) -> Tuple[str, str, VideoMetadata]:
    """
    Process uploaded video: save temporarily, validate, transcode,
    calculate hash, move to final location, and return paths/metadata.

    Returns:
        Tuple[str, str, VideoMetadata]: (final_filepath, relative_file_key, out_video_metadata)
    """
    logger.info(f"Starting processing for uploaded file: {file.filename}")

    with tempfile.TemporaryDirectory() as tempdir:
        temp_dir_path = Path(tempdir)
        safe_filename = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in file.filename)
        in_path = temp_dir_path / f"in_{safe_filename}"
        out_path = temp_dir_path / f"out_{safe_filename}.mp4"

        try:
            with open(in_path, "wb") as in_f:
                shutil.copyfileobj(file, in_f)
            logger.info(f"Uploaded file '{file.filename}' saved to temporary location: {in_path}")
        except Exception as e:
            logger.exception(f"Failed to save uploaded file {file.filename} to temp dir.")
            raise IOError(f"Could not save uploaded file: {e}")

        try:
            video_metadata = get_video_metadata(str(in_path))
            logger.info(f"Video metadata extracted for '{file.filename}': duration={video_metadata.duration_sec}, "
                        f"resolution=({video_metadata.width}x{video_metadata.height}), fps={video_metadata.fps}")
        except (av.InvalidDataError, av.error.FileNotFoundError) as e:
            logger.error(f"Invalid or unreadable video file provided: {file.filename}. Error: {e}")
            raise ValueError(f"Not a valid or readable video file: {file.filename}")
        except Exception as e:
            logger.exception(f"Error getting metadata for {file.filename}")
            raise ValueError(f"Could not get video metadata: {e}")

        if video_metadata.num_video_streams == 0:
            logger.error(f"Video container '{file.filename}' does not contain a video stream.")
            raise ValueError(f"Video container '{file.filename}' does not contain a video stream")
        if video_metadata.width is None or video_metadata.height is None:
            logger.error(f"Video container '{file.filename}' does not contain width or height metadata.")
            raise ValueError(f"Video container '{file.filename}' does not contain width or height metadata")
        if video_metadata.duration_sec is None or video_metadata.duration_sec <= 0:
            logger.error(f"Video container '{file.filename}' does not have valid duration metadata.")
            raise ValueError(f"Video container '{file.filename}' lacks valid time duration metadata")

        start_time_sec_proc, duration_time_sec_proc = _get_start_sec_duration_sec(
            max_time=max_time,
            start_time_sec=start_time_sec,
            duration_time_sec=duration_time_sec,
        )
        logger.info(f"Video processing parameters for '{file.filename}': "
                    f"start_time_sec={start_time_sec_proc}, duration_time_sec={duration_time_sec_proc}")

        logger.info(f"Starting video transcoding process for '{file.filename}'.")
        transcode(
            str(in_path),
            str(out_path),
            video_metadata,
            seek_t=start_time_sec_proc,
            duration_time_sec=duration_time_sec_proc,
        )
        logger.info(f"Video transcoding completed for '{file.filename}'. Output: {out_path}")

        try:
            os.remove(in_path)
            logger.debug(f"Removed temporary input file: {in_path}")
        except OSError as e:
            logger.warning(f"Could not remove temporary input file {in_path}: {e}")

        out_video_metadata = get_video_metadata(str(out_path))
        if out_video_metadata.num_video_frames == 0:
            logger.error(f"Transcode produced empty video for '{file.filename}'. Check input or parameters.")
            raise ValueError(
                "Transcode produced empty video; check seek time or your input video"
            )
        logger.info(f"Transcoded video metadata for '{file.filename}': "
                    f"duration={out_video_metadata.duration_sec}, "
                    f"frames={out_video_metadata.num_video_frames}, "
                    f"resolution=({out_video_metadata.width}x{out_video_metadata.height})")

        file_hash = get_file_hash(str(out_path))
        logger.info(f"Calculated hash for transcoded video '{file.filename}': {file_hash}")

        final_filename = f"{file_hash}.mp4"
        file_key = f"{UPLOADS_PREFIX}/{final_filename}"
        final_filepath = UPLOADS_PATH / final_filename

        try:
            UPLOADS_PATH.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out_path), final_filepath)
            logger.info(f"Processed video '{file.filename}' moved to final destination: {final_filepath}")
        except Exception as e:
            logger.exception(f"Failed to move processed video {out_path} to {final_filepath}")
            try:
                os.remove(out_path)
            except OSError:
                pass
            raise IOError(f"Could not move processed video to final location: {e}")

        return str(final_filepath), file_key, out_video_metadata


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
)