# /home/david_elliott/github/sam2/demo/backend/server/inference/api.py
"""
This module provides the main InferenceAPI class that manages inference sessions,
delegating operations to specialized modules.
"""

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Generator, List, Tuple, Dict # Added Dict
import time
from io import BytesIO # Added BytesIO

import torch

from app_conf import MODEL_SIZE, APP_ROOT, DATA_PATH, SESSIONS_PATH
from data.data_types import SessionInfo
from inference.data_types import (
    AddPointsRequest, AddMaskRequest,
    CancelPorpagateResponse, CancelPropagateInVideoRequest,
    ClearPointsInFrameRequest, ClearPointsInVideoRequest,
    ClearPointsInVideoResponse, CloseSessionRequest, CloseSessionResponse,
    DownloadMasksRequest, DownloadMasksResponse, PropagateDataResponse,
    PropagateInVideoRequest, RemoveObjectRequest, RemoveObjectResponse,
    SetObjectNameRequest, SetObjectNameResponse, # <-- Added
    StartSessionRequest, StartSessionResponse
)
# Import from refactored modules
from . import model_loader
from . import video_reader
from . import state_persistor
# Import existing helper modules
from . import inference_operations
from . import downloaders # Keep this
from . import session_manager

# Import specific downloader operations
from .downloaders import (
    download_frames_operation,
    download_masks_operation,
    download_images_zip_operation,
    download_yolo_labels_operation,
    download_yolo_format_operation
)


logger = logging.getLogger(__name__)

def sanitize_path(path: str) -> str:
    """
    Sanitize the video path to use as session_id by replacing '/' with '_'.

    Args:
        path (str): The relative video path (e.g., 'gallery/video1.mp4').

    Returns:
        str: A sanitized string safe for use as a directory or file name component (e.g., 'gallery_video1.mp4').
    """
    sanitized = path.replace('/', '_').replace('\\', '_')
    # Additional sanitization: remove potentially problematic characters for filenames/paths
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in ('_', '.', '-')).strip()
    # Prevent excessively long names if paths are deep (optional)
    max_len = 100
    if len(sanitized) > max_len:
        # Simple truncation, consider adding a hash if collisions are likely
        sanitized = sanitized[-max_len:]
        logger.warning(f"Sanitized path truncated to: {sanitized}")

    # Prevent names starting/ending with dots or consisting only of dots
    if sanitized.startswith('.'): sanitized = '_' + sanitized[1:]
    if sanitized.endswith('.'): sanitized = sanitized[:-1] + '_'
    if all(c == '.' for c in sanitized): sanitized = 'default_session'
    if not sanitized: sanitized = 'default_session' # Handle empty case

    return sanitized


class InferenceAPI:
    """
    Core class responsible for managing inference sessions and performing segmentation.
    Operations are delegated to helper modules for clarity and maintainability.
    Persists user clicks, masks, and object names to disk in the SESSIONS_PATH directory.
    """

    def __init__(self) -> None:
        super(InferenceAPI, self).__init__()
        self.session_states: Dict[str, Dict] = {} # Added type hint
        self.score_thresh = 0 # TODO: Make configurable if needed

        # Load model and determine device using the loader module
        force_cpu_device = os.environ.get("SAM2_DEMO_FORCE_CPU_DEVICE", "0") == "1"
        self.device = model_loader.select_device(force_cpu=force_cpu_device)
        self.predictor = model_loader.load_predictor(MODEL_SIZE, APP_ROOT, self.device)

        self.inference_lock = Lock() # Lock for thread safety on shared resources (predictor, session_states)

    def autocast_context(self):
        """
        Returns an automatic mixed-precision contextmanager based on the device.
        """
        return model_loader.get_autocast_context(self.device)

    # --- Session Management ---
    def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        """
        Start an inference session for a given video path.
        Uses the sanitized video path as the session ID to ensure persistence per video.
        Loads persisted state (clicks, masks, names) if available, otherwise initializes a new state.

        Args:
            request (StartSessionRequest): Request containing the video path.

        Returns:
            StartSessionResponse: Response containing the session ID.

        Raises:
            FileNotFoundError: If the video file specified in the request is not found.
            RuntimeError: If video reading fails or predictor initialization fails.
        """
        # Lock to prevent race conditions when creating/accessing sessions
        with self.inference_lock:
            # Use sanitized video path as session_id for persistence
            sanitized_video_path = sanitize_path(request.path)
            session_id = sanitized_video_path # Use this as the key

            if session_id in self.session_states:
                logger.info(f"Session for video '{request.path}' (ID: {session_id}) already exists. Returning existing session.")
                # Update last use time when session is accessed
                session = self.session_states[session_id]
                if "state" in session:
                    session["state"]["last_use_time"] = time.time()
                else:
                    # This case should ideally not happen if sessions are created correctly
                    logger.warning(f"Session {session_id} exists but has no 'state' key.")
                    # Attempt to re-initialize or handle error? Let's re-initialize cleanly.
                    # This ensures the session is usable even if something went wrong before.
                    try:
                        logger.warning(f"Re-initializing state for existing session {session_id} due to missing 'state' key.")
                        # Force re-initialization logic here (similar to the 'else' block below)
                        # Ensure to pass the existing session dict to update it in place if needed
                        self._initialize_session_state(session_id, request.path, session) # Pass session dict
                        logger.info(f"Successfully re-initialized state for session {session_id}")
                    except Exception as reinit_e:
                        logger.error(f"Failed to re-initialize state for session {session_id}: {reinit_e}", exc_info=True)
                        # If re-initialization fails, remove the faulty session entry?
                        del self.session_states[session_id]
                        raise RuntimeError(f"Failed to recover session {session_id}: {reinit_e}") from reinit_e
                return StartSessionResponse(session_id=session_id)

            logger.info(f"Starting new session for video '{request.path}' (ID: {session_id})")
            try:
                # Create a new session dictionary
                new_session = {
                    "canceled": False, # Flag for canceling long operations like propagation
                    "video_path": request.path # Store original path for reference if needed
                    # "state" will be added by _initialize_session_state
                }
                # Initialize state and add it to the new_session dict
                self._initialize_session_state(session_id, request.path, new_session)

                # Add the fully initialized session to session_states
                self.session_states[session_id] = new_session

                logger.info(f"Successfully started session {session_id}. Current sessions: {len(self.session_states)}")
                logger.debug(f"Session stats: {session_manager.get_session_stats(self.session_states)}")
                return StartSessionResponse(session_id=session_id)

            except FileNotFoundError as e:
                logger.error(f"Failed to start session for {request.path}: {e}", exc_info=True)
                raise # Re-raise specific error for Flask/GraphQL handler
            except RuntimeError as e:
                logger.error(f"Runtime error starting session for {request.path}: {e}", exc_info=True)
                raise # Re-raise specific error for Flask/GraphQL handler
            except Exception as e:
                logger.error(f"Unexpected error starting session for {request.path}: {e}", exc_info=True)
                # Clean up partial session state if creation failed midway
                if session_id in self.session_states:
                    del self.session_states[session_id]
                raise RuntimeError(f"Failed to start session due to an unexpected error: {e}") from e

    def _initialize_session_state(self, session_id: str, video_path: str, session_dict: dict) -> None:
        """Helper to initialize the 'state' part of a session dictionary."""
        session_dir = SESSIONS_PATH / session_id
        current_time = time.time() # Get current time once

        # Load raw frames using the video reader module
        raw_frames = video_reader.read_raw_frames(video_path, DATA_PATH)
        num_frames_loaded = len(raw_frames)
        if num_frames_loaded == 0:
            logger.warning(f"Video {video_path} loaded 0 frames.")
            # Allow empty session for now, download ops might fail later.

        # Initialize predictor state
        offload_video_to_cpu = self.device.type == "mps" # Specific handling for MPS
        absolute_video_path = str(DATA_PATH / video_path) # Predictor expects absolute path

        logger.debug(f"Initializing predictor state for {absolute_video_path}, offload_to_cpu={offload_video_to_cpu}")
        inference_state = self.predictor.init_state(
            absolute_video_path,
            offload_video_to_cpu=offload_video_to_cpu,
        )
        logger.debug("Predictor state initialized.")

        # Store raw frames and count in the state (used by downloaders)
        inference_state["images_original"] = raw_frames
        inference_state["num_frames"] = num_frames_loaded
        # Use time.time() for timestamps
        inference_state["start_time"] = current_time
        inference_state["last_use_time"] = current_time

        # Load persisted clicks/masks if the session directory exists
        if session_dir.exists():
            logger.info(f"Loading existing session state from {session_dir}")
            state_persistor.load_session_state(session_id, inference_state, SESSIONS_PATH, self.device)
            # Load object names
            loaded_names = session_manager.load_object_names(session_id)
            # Store names in the inference state
            inference_state["object_names"] = loaded_names # Use this dict to store names
            logger.info(f"Loaded {len(loaded_names)} object names for session {session_id}.")
        else:
            logger.info(f"No persisted state found at {session_dir}, starting fresh.")
            # Ensure necessary keys exist even if not loaded
            if "points" not in inference_state: inference_state["points"] = {}
            if "labels" not in inference_state: inference_state["labels"] = {}
            if "masks_cache" not in inference_state: inference_state["masks_cache"] = {}
            if "object_names" not in inference_state: inference_state["object_names"] = {} # Ensure names dict exists

        # Add the initialized state to the session dictionary provided
        session_dict["state"] = inference_state


    def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        """
        Close an inference session, saving its state (clicks, masks, names) to disk before removing from memory.

        Args:
            request (CloseSessionRequest): Request containing the session ID.

        Returns:
            CloseSessionResponse: Response indicating success or failure.
        """
        session_id = request.session_id
        success = False
        # Use a shorter timeout for the lock acquisition here?
        if not self.inference_lock.acquire(timeout=5.0):
            logger.error(f"Timeout acquiring lock to close session {session_id}.")
            return CloseSessionResponse(success=False)

        try:
            if session_id in self.session_states:
                logger.info(f"Closing session {session_id}...")
                session = self.session_states[session_id]
                inference_state = session.get("state", {})

                # Save state (clicks, masks) before removing from memory
                logger.debug(f"Saving final state for session {session_id} before closing.")
                if inference_state:
                    state_persistor.save_session_state(session_id, inference_state, SESSIONS_PATH)
                    # Save object names (if they exist in the state)
                    if "object_names" in inference_state:
                        session_manager.save_object_names(session_id, inference_state["object_names"])
                    else:
                         logger.debug(f"No 'object_names' key found in state for session {session_id}, skipping name save.")
                else:
                    logger.warning(f"Session {session_id} has no state to save.")

                # Clear from memory using session manager
                success = session_manager.clear_session_state(self.session_states, session_id)

                # Clean up GPU memory if possible
                if self.device.type == "cuda":
                    logger.debug("Attempting to clear CUDA cache after closing session.")
                    torch.cuda.empty_cache()
                logger.info(f"Session {session_id} closed. Success: {success}")
            else:
                logger.warning(f"Attempted to close non-existent session {session_id}.")
                success = False # Or True if "already closed" is considered success? Let's say False.

            logger.debug(f"Session stats after close: {session_manager.get_session_stats(self.session_states)}")
            return CloseSessionResponse(success=success)
        except Exception as e:
            logger.error(f"Error during closing of session {session_id}: {e}", exc_info=True)
            # Ensure state is consistent even if saving failed
            if session_id in self.session_states:
                 logger.warning(f"Session {session_id} might still be in memory due to close error.")
                 # Attempt to clear from memory again? Or leave it? Leave it for now.
                 success = False # Ensure success is false on error
            else:
                 success = False # Session wasn't found initially or error happened before removal
            return CloseSessionResponse(success=success)
        finally:
            # Ensure lock is released even if errors occur
            self.inference_lock.release()

    # --- Inference Operations ---
    def add_points(self, request: AddPointsRequest) -> PropagateDataResponse:
        """
        Add point prompts to a frame, update masks, and persist the state.

        Args:
            request (AddPointsRequest): Request with point data and session ID.

        Returns:
            PropagateDataResponse: Updated masks for the affected frame.
        """
        # Lock needed for modifying state and using predictor
        with torch.no_grad(), self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            # Delegate to inference_operations
            response = inference_operations.add_points_operation(
                self.predictor, inference_state, request, score_thresh=self.score_thresh
            )
            # Persist state after operation
            state_persistor.save_session_state(request.session_id, inference_state, SESSIONS_PATH)
            # Update last use time
            inference_state["last_use_time"] = time.time()
            return response

    def add_mask(self, request: AddMaskRequest) -> PropagateDataResponse:
        """
        Add a mask prompt directly, update masks, and persist the state.

        Args:
            request: Request with mask data and session ID.

        Returns:
            PropagateDataResponse: Updated mask for the frame.
        """
        # Lock needed for modifying state and using predictor
        with torch.no_grad(), self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            # Delegate to inference_operations
            response = inference_operations.add_mask_operation(
                self.predictor, inference_state, request, score_thresh=self.score_thresh
            )
            # Persist state after operation
            state_persistor.save_session_state(request.session_id, inference_state, SESSIONS_PATH)
            # Update last use time
            inference_state["last_use_time"] = time.time()
            return response

    def clear_points_in_frame(self, request: ClearPointsInFrameRequest) -> PropagateDataResponse:
        """
        Clear point prompts in a specific frame, update masks, and persist the state.

        Args:
            request (ClearPointsInFrameRequest): Request identifying frame, object, and session.

        Returns:
            PropagateDataResponse: Updated mask for the frame.
        """
        # Lock needed for modifying state and using predictor
        with torch.no_grad(), self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            # Delegate to inference_operations
            response = inference_operations.clear_points_in_frame_operation(
                 self.predictor, inference_state, request, score_thresh=self.score_thresh
            )
            # Persist state after operation
            state_persistor.save_session_state(request.session_id, inference_state, SESSIONS_PATH)
            # Update last use time
            inference_state["last_use_time"] = time.time()
            return response

    def clear_points_in_video(self, request: ClearPointsInVideoRequest) -> ClearPointsInVideoResponse:
        """
        Clear all point prompts in the video, update state, and persist.

        Args:
            request (ClearPointsInVideoRequest): Request with the session ID.

        Returns:
            ClearPointsInVideoResponse: Confirmation of clearing.
        """
        # Lock needed for modifying state and using predictor
        # Let's keep no_grad for safety, although unlikely needed for just reset.
        with torch.no_grad(), self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            # Delegate to inference_operations
            response = inference_operations.clear_points_in_video_operation(
                self.predictor, inference_state, request
            )
            # Persist state after operation (cleared state)
            state_persistor.save_session_state(request.session_id, inference_state, SESSIONS_PATH)
            # Save cleared object names (should be empty after reset)
            session_manager.save_object_names(request.session_id, inference_state.get("object_names", {}))

            # Update last use time
            inference_state["last_use_time"] = time.time()
            return response

    def remove_object(self, request: RemoveObjectRequest) -> RemoveObjectResponse:
        """
        Remove an object from the segmentation state, update masks, and persist state and names.

        Args:
            request (RemoveObjectRequest): Request with the object ID and session ID.

        Returns:
            RemoveObjectResponse: Updated masks after removal (potentially multiple frames).
        """
        # Lock needed for modifying state and using predictor
        with torch.no_grad(), self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            obj_id_to_remove = request.object_id

            # Delegate to inference_operations
            response = inference_operations.remove_object_operation(
                self.predictor, inference_state, request, score_thresh=self.score_thresh
            )

            # Remove object name if it exists in state
            name_removed = False
            if "object_names" in inference_state and obj_id_to_remove in inference_state["object_names"]:
                del inference_state["object_names"][obj_id_to_remove]
                name_removed = True
                logger.info(f"Removed name for object {obj_id_to_remove} during removal.")

            # Persist state after operation
            state_persistor.save_session_state(request.session_id, inference_state, SESSIONS_PATH)
            # Save the updated names only if a name was actually removed
            if name_removed:
                session_manager.save_object_names(request.session_id, inference_state["object_names"])

            # Update last use time
            inference_state["last_use_time"] = time.time()
            return response

    def propagate_in_video(self, request: PropagateInVideoRequest) -> Generator[PropagateDataResponse, None, None]:
        """
        Propagate prompts throughout the video, yield updated masks per frame, and persist state periodically and finally.

        Args:
            request (PropagateInVideoRequest): Request with propagation parameters and session ID.

        Yields:
            Generator[PropagateDataResponse]: Yields updated masks per processed frame.

        Raises:
            RuntimeError: If propagation fails (e.g., CUDA errors).
        """
        # Lock needed for the entire duration as it modifies state heavily
        with torch.no_grad(), self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            session["canceled"] = False # Reset cancel flag at the start
            inference_state = session["state"]
            session_id = request.session_id

            logger.info(f"Starting propagation for session {session_id} from frame {request.start_frame_index}")
            logger.debug(f"Session stats before propagation: {session_manager.get_session_stats(self.session_states)}")

            # Define how often to save state during propagation (e.g., every N frames)
            save_interval = 50 # Save state every 50 frames processed
            frames_processed_since_save = 0

            try:
                # Delegate to inference_operations, passing the session dict for cancellation check
                generator = inference_operations.propagate_in_video_operation(
                    self.predictor, inference_state, request, session, score_thresh=self.score_thresh
                )

                for response in generator:
                    # Update last use time on each yielded frame
                    inference_state["last_use_time"] = time.time()
                    yield response

                    frames_processed_since_save += 1
                    # Periodically save state and clear mask cache to manage memory
                    if frames_processed_since_save >= save_interval:
                        logger.debug(f"Saving intermediate state for session {session_id} during propagation (frame {response.frame_index})")
                        state_persistor.save_session_state(session_id, inference_state, SESSIONS_PATH)
                        # Object names don't change during propagation, no need to save them here.
                        # Optionally clear *older* parts of the cache if memory is tight
                        # For now, just save. The cache itself is updated by propagate_in_video_operation
                        frames_processed_since_save = 0
                        # Release unused GPU memory if possible
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()

                # Check cancellation status after generator finishes (in case it was cancelled just before finishing)
                if session.get("canceled", False):
                    logger.info(f"Propagation cancelled for session {session_id}.")
                else:
                    logger.info(f"Propagation finished successfully for session {session_id}.")

            except RuntimeError as e:
                logger.error(f"Runtime error during propagation for session {session_id}: {e}", exc_info=True)
                # Persist whatever state we have upon failure
                logger.warning(f"Attempting to save state for session {session_id} after propagation error.")
                state_persistor.save_session_state(session_id, inference_state, SESSIONS_PATH)
                # Also save names in case they were loaded but not saved yet
                session_manager.save_object_names(session_id, inference_state.get("object_names", {}))
                raise # Re-raise to signal failure to the caller
            except Exception as e:
                logger.error(f"Unexpected error during propagation for session {session_id}: {e}", exc_info=True)
                logger.warning(f"Attempting to save state for session {session_id} after unexpected propagation error.")
                state_persistor.save_session_state(session_id, inference_state, SESSIONS_PATH)
                session_manager.save_object_names(session_id, inference_state.get("object_names", {}))
                raise RuntimeError(f"Unexpected error during propagation: {e}") from e
            finally:
                # Final save after propagation completes or is cancelled/errors out
                if session_id in self.session_states: # Check if session still exists (wasn't closed concurrently)
                    logger.info(f"Saving final state for session {session_id} after propagation attempt.")
                    # Update last use time one last time
                    inference_state["last_use_time"] = time.time()
                    state_persistor.save_session_state(session_id, inference_state, SESSIONS_PATH)
                    # Save names one last time as well
                    session_manager.save_object_names(session_id, inference_state.get("object_names", {}))

                    # Clear the mask cache *after* propagation is fully done to save memory?
                    # The cache is useful for immediate display, maybe keep it?
                    # Let's keep it for now, download_masks will use it. If memory becomes
                    # an issue, clearing it here is an option.
                    # if "masks_cache" in inference_state:
                    #     logger.debug(f"Clearing masks cache for session {session_id} after propagation.")
                    #     inference_state["masks_cache"].clear()

                    # Final GPU memory cleanup
                    if self.device.type == "cuda":
                        logger.debug("Attempting final CUDA cache clear after propagation.")
                        torch.cuda.empty_cache()
                logger.debug(f"Session stats after propagation: {session_manager.get_session_stats(self.session_states)}")


    def cancel_propagate_in_video(self, request: CancelPropagateInVideoRequest) -> CancelPorpagateResponse:
        """
        Set the cancellation flag for an ongoing propagation process.

        Args:
            request (CancelPropagateInVideoRequest): Request with the session ID.

        Returns:
            CancelPorpagateResponse: Confirmation of cancellation request.
        """
        # Lock required to safely modify the session's 'canceled' flag
        with self.inference_lock:
            # Check if session exists before trying to get it
            if request.session_id not in self.session_states:
                logger.warning(f"Attempted to cancel propagation for non-existent session {request.session_id}")
                # Return success=False as the intended action (cancelling) didn't apply.
                return CancelPorpagateResponse(success=False)

            session = session_manager.get_session(self.session_states, request.session_id)
            # Delegate to inference_operations
            response = inference_operations.cancel_propagate_in_video_operation(session, request)
            logger.info(f"Cancellation requested for session {request.session_id}. Result: {response.success}")
            # Update last use time ? Maybe not for cancellation.
            return response

    # --- NEW: Set Object Name Method ---
    def set_object_name(self, request: SetObjectNameRequest) -> SetObjectNameResponse:
        """
        Sets or clears the custom name for a tracked object within a session and persists it.

        Args:
            request (SetObjectNameRequest): Request containing session_id, object_id, and name.

        Returns:
            SetObjectNameResponse: Confirmation and the name that was set.

        Raises:
            RuntimeError: If the session is not found.
        """
        session_id = request.session_id
        object_id = request.object_id
        name = request.name.strip() # Trim whitespace

        # Lock needed to modify session state (object_names) and save it
        with self.inference_lock:
            session = session_manager.get_session(self.session_states, session_id)
            inference_state = session["state"]

            # Ensure the object_names dictionary exists
            if "object_names" not in inference_state:
                inference_state["object_names"] = {}

            # Update or remove the name
            if name:
                inference_state["object_names"][object_id] = name
                logger.info(f"Set name for object {object_id} in session {session_id} to '{name}'.")
            else:
                # If name is empty, remove the custom name entry
                if object_id in inference_state["object_names"]:
                    del inference_state["object_names"][object_id]
                    logger.info(f"Cleared custom name for object {object_id} in session {session_id}.")
                else:
                    logger.info(f"No custom name existed for object {object_id} in session {session_id}, no change needed.")

            # Persist the updated names
            try:
                session_manager.save_object_names(session_id, inference_state["object_names"])
            except IOError as e:
                logger.error(f"Failed to persist object names for session {session_id}: {e}", exc_info=True)
                # Re-raise as a runtime error to signal failure to the caller (GraphQL mutation)
                raise RuntimeError(f"Failed to save object names: {e}") from e

            # Update last use time
            inference_state["last_use_time"] = time.time()

            return SetObjectNameResponse(
                success=True,
                object_id=object_id,
                name=name # Return the name that was set (empty string if cleared)
            )


    # --- Download Operations ---

    def download_masks(self, request: DownloadMasksRequest) -> DownloadMasksResponse:
        """
        Download all masks for a session, generating them if not cached.
        This method provides the raw RLE mask data, suitable for JSON output or further processing.
        It ensures the masks_cache is populated if needed.

        Args:
            request (DownloadMasksRequest): Request with the session ID.

        Returns:
            DownloadMasksResponse: Response with RLE masks for each frame.
        """
        # Lock needed to access state, predictor, and potentially generate masks
        with torch.no_grad(), self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            session_id = request.session_id

            # Ensure masks cache exists
            if "masks_cache" not in inference_state:
                inference_state["masks_cache"] = {}

            num_frames = inference_state.get("num_frames", 0)
            if num_frames <= 0:
                logger.warning(f"Cannot download masks for session {session_id}: No frames available.")
                return DownloadMasksResponse(results=[])

            logger.info(f"Starting download_masks operation for session {session_id} ({num_frames} frames).")
            # Delegate to downloaders module - this will use cache or run predictor.propagate_in_video
            # It *populates* the inference_state["masks_cache"]
            response = download_masks_operation(
                self.predictor, inference_state, num_frames, score_thresh=self.score_thresh
            )
            logger.info(f"Finished download_masks operation for session {session_id}. Got {len(response.results)} frame results.")

            # Persist the potentially updated masks cache (optional, but consistent)
            state_persistor.save_session_state(session_id, inference_state, SESSIONS_PATH)
            # Update last use time
            inference_state["last_use_time"] = time.time()

            # Clean up memory after potentially large operation
            if self.device.type == "cuda":
                 logger.debug("Attempting CUDA cache clear after download_masks.")
                 torch.cuda.empty_cache()

            # The response from download_masks_operation contains the mask data
            return response

    def download_frames(self, session_id: str) -> Generator[Tuple[int, bytes], None, None]:
        """
        Stream raw video frames as JPEG images.

        Args:
            session_id (str): The session ID.

        Yields:
            Generator[Tuple[int, bytes]]: Yields tuples of (frame_index, jpeg_bytes).
        """
        # Acquire lock briefly to get session and update time, then release for generation
        session = None
        with self.inference_lock:
            session = session_manager.get_session(self.session_states, session_id)
            inference_state = session["state"]
            # Update last use time
            inference_state["last_use_time"] = time.time()

        # Delegate to downloaders module (no lock during generation for streaming)
        if session:
            logger.info(f"Starting download_frames stream for session {session_id}")
            # Pass the session dict, which contains the 'state' needed by the downloader
            yield from download_frames_operation(session)
            logger.info(f"Finished download_frames stream for session {session_id}")
        else:
            # Should not happen if get_session raises error, but handle defensively
            logger.error(f"Session {session_id} not found for download_frames stream.")
            return # Empty generator


    # --- NEW Download Endpoints Returning Files/Zips ---

    def download_images_zip(self, session_id: str) -> BytesIO:
        """
        Download original video frames as a zip archive.

        Args:
            session_id (str): The session ID.

        Returns:
            BytesIO: A byte stream containing the zip archive.

        Raises:
            RuntimeError: If session not found or zip creation fails.
        """
        # Lock needed to access session state safely
        with self.inference_lock:
            session = session_manager.get_session(self.session_states, session_id)
            # Update last use time
            session["state"]["last_use_time"] = time.time()
            # Delegate to downloader
            logger.info(f"Initiating download_images_zip for session {session_id}")
            try:
                zip_buffer = download_images_zip_operation(session)
                logger.info(f"Successfully generated images zip for session {session_id}")
                return zip_buffer
            except Exception as e:
                logger.error(f"Failed download_images_zip for session {session_id}: {e}", exc_info=True)
                # Re-raise as RuntimeError for the web server to handle
                raise RuntimeError(f"Failed to generate images zip: {e}") from e

    def download_yolo_labels(self, session_id: str) -> BytesIO:
        """
        Download YOLO format labels (labels/*.txt + classes.txt) as a zip archive.

        Args:
            session_id (str): The session ID.

        Returns:
            BytesIO: A byte stream containing the zip archive.

        Raises:
            RuntimeError: If session not found or zip creation fails.
        """
        # Lock needed to ensure consistent state access during mask/name retrieval and generation
        # Need full lock for download_masks call inside downloader if it modifies cache
        with torch.no_grad(), self.autocast_context(), self.inference_lock:
            # First, ensure session exists and update last use time
            session = session_manager.get_session(self.session_states, session_id)
            session["state"]["last_use_time"] = time.time()

            logger.info(f"Initiating download_yolo_labels for session {session_id}")
            try:
                # Pass self (InferenceAPI instance) to the operation function
                # This allows download_yolo_labels_operation to call self.download_masks
                zip_buffer = download_yolo_labels_operation(session_id, self)
                logger.info(f"Successfully generated YOLO labels zip for session {session_id}")
                return zip_buffer
            except Exception as e:
                logger.error(f"Failed download_yolo_labels for session {session_id}: {e}", exc_info=True)
                raise RuntimeError(f"Failed to generate YOLO labels zip: {e}") from e


    def download_yolo_format(self, session_id: str) -> BytesIO:
        """
        Download full YOLO dataset (images/*, labels/*.txt, classes.txt) as a zip archive.

        Args:
            session_id (str): The session ID.

        Returns:
            BytesIO: A byte stream containing the zip archive.

        Raises:
            RuntimeError: If session not found or zip creation fails.
        """
        # Lock needed for the duration due to mask generation and state access
        # Need full lock for download_masks call inside downloader if it modifies cache
        with torch.no_grad(), self.autocast_context(), self.inference_lock:
            # First, ensure session exists and update last use time
            session = session_manager.get_session(self.session_states, session_id)
            session["state"]["last_use_time"] = time.time()

            logger.info(f"Initiating download_yolo_format for session {session_id}")
            try:
                 # Pass self (InferenceAPI instance)
                zip_buffer = download_yolo_format_operation(session_id, self)
                logger.info(f"Successfully generated YOLO format zip for session {session_id}")
                return zip_buffer
            except Exception as e:
                logger.error(f"Failed download_yolo_format for session {session_id}: {e}", exc_info=True)
                raise RuntimeError(f"Failed to generate YOLO format zip: {e}") from e


    # --- Other Methods ---
    def list_sessions(self) -> List[SessionInfo]:
        """
        Retrieve metadata for all currently active inference sessions.

        Returns:
            List[SessionInfo]: A list of SessionInfo objects.
        """
        with self.inference_lock: # Lock needed to safely iterate over session_states
            sessions_info = []
            current_time = time.time() # Get time once for TTL calculation
            for session_id, session_data in self.session_states.items():
                inference_state = session_data.get("state", {})
                # Safely get metadata from inference_state
                num_frames = inference_state.get("num_frames", 0) # Default to 0

                # Calculate num_objects based on the object_names dictionary keys
                num_objects = len(inference_state.get("object_names", {}).keys())
                # Alternative: Count based on points/labels structure if names aren't definitive
                # obj_ids = set()
                # points_dict = inference_state.get("points", {})
                # for frame_idx, frame_points in points_dict.items():
                #      obj_ids.update(frame_points.keys())
                # num_objects = len(obj_ids)

                start_time = inference_state.get("start_time", 0.0)
                last_use_time = inference_state.get("last_use_time", start_time) # Use start_time if last_use not updated yet

                sessions_info.append(
                    SessionInfo(
                        session_id=session_id,
                        start_time=start_time,
                        last_use_time=last_use_time,
                        num_frames=num_frames,
                        num_objects=num_objects,
                        # Add other relevant info if needed
                    )
                )
            logger.info(f"Listed {len(sessions_info)} active sessions.")
            logger.debug(f"Session stats: {session_manager.get_session_stats(self.session_states)}")
            return sessions_info