# File: /home/david_elliott/github/sam2-git/demo/backend/server/app.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import atexit
import signal
import threading # Added for lock check
from typing import Any, Generator, Tuple
from io import BytesIO # Added BytesIO

import torch
from flask import (Flask, make_response, Request, request, Response,
                   send_from_directory, jsonify, send_file) # Added send_file
from flask_cors import CORS
from strawberry.flask.views import GraphQLView

from app_conf import (
    GALLERY_PATH, GALLERY_PREFIX, POSTERS_PATH, POSTERS_PREFIX,
    UPLOADS_PATH, UPLOADS_PREFIX, API_URL, DATA_PATH, SESSIONS_PATH # Added SESSIONS_PATH
)
from data.loader import preload_data
from data.schema import schema
from data.store import set_videos, get_videos
# Import specific request/response types used in routes
from inference.data_types import (
    PropagateDataResponse, PropagateInVideoRequest, CloseSessionRequest,
    DownloadMasksRequest # Import if download_masks route uses it explicitly
)
from inference.multipart import MultipartResponseBuilder
# Updated import path for InferenceAPI
from inference.api import InferenceAPI

logger = logging.getLogger(__name__)
# Force DEBUG logs to see detailed messages from inference modules
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set higher level for noisy libraries if needed
# logging.getLogger("PIL").setLevel(logging.WARNING)

app = Flask(__name__)
# Ensure CORS allows necessary headers and methods for streaming/GraphQL
cors = CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": "*", # Adjust in production
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
    }
})


# --- Initialization ---
try:
    logger.info("Preloading video data...")
    videos = preload_data()
    set_videos(videos)
    logger.info(f"Finished preloading {len(videos)} videos.")

    logger.info("Initializing Inference API...")
    inference_api = InferenceAPI()
    logger.info("Inference API initialized successfully.")
except Exception as e:
     logger.critical(f"Failed to initialize application: {e}", exc_info=True)
     # Depending on deployment, might want to exit or enter a degraded state
     raise SystemExit(f"Application initialization failed: {e}")


# --- Shutdown Handling ---
def shutdown_handler(signum=None, frame=None):
    """Clean up inference sessions and GPU memory on shutdown."""
    logger.warning("Received shutdown signal, initiating cleanup...")
    try:
        # Create a copy of keys to avoid issues if close_session modifies the dict
        session_ids_to_close = list(inference_api.session_states.keys())
        logger.info(f"Found {len(session_ids_to_close)} active sessions to close.")
        for session_id in session_ids_to_close:
            logger.debug(f"Attempting to close session {session_id}...")
            try:
                # Use the dedicated CloseSessionRequest data type
                close_request = CloseSessionRequest(type="close_session", session_id=session_id)
                response = inference_api.close_session(close_request)
                logger.debug(f"Close session {session_id} response: Success={response.success}")
            except Exception as close_exc:
                logger.error(f"Error closing session {session_id} during shutdown: {close_exc}", exc_info=True)

        # Final attempt to clear GPU cache
        if torch.cuda.is_available():
            logger.info("Clearing GPU memory cache as part of shutdown.")
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared.")
        else:
             logger.info("No GPU available, skipping cache clearing.")

    except Exception as e:
        logger.error(f"Error during shutdown cleanup process: {e}", exc_info=True)
    finally:
        logger.warning("Shutdown cleanup process completed.")

# Register shutdown handlers
atexit.register(shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler) # Handle Ctrl+C


# --- Health Check Route ---
@app.route("/healthy")
def healthy() -> Response:
    """Simple health check endpoint."""
    # Could potentially add checks here, e.g., model loaded status
    # is_model_loaded = hasattr(inference_api, 'predictor') and inference_api.predictor is not None
    # status = "OK" if is_model_loaded else "DEGRADED"
    # code = 200 if is_model_loaded else 503
    return make_response("OK", 200)


# --- Static File Routes ---
# (No changes needed in these routes)
@app.route(f"/{GALLERY_PREFIX}/<path:path>", methods=["GET"])
def send_gallery_video(path: str) -> Response:
    """Serves video files from the gallery directory."""
    try:
        return send_from_directory(GALLERY_PATH, path)
    except FileNotFoundError:
        logger.warning(f"Gallery resource not found: {path}")
        return make_response("Resource not found", 404)
    except Exception as e:
        logger.error(f"Error sending gallery video '{path}': {e}", exc_info=True)
        return make_response("Internal server error", 500)


@app.route(f"/{POSTERS_PREFIX}/<path:path>", methods=["GET"])
def send_poster_image(path: str) -> Response:
    """Serves poster image files from the posters directory."""
    try:
        return send_from_directory(POSTERS_PATH, path)
    except FileNotFoundError:
        logger.warning(f"Poster resource not found: {path}")
        return make_response("Resource not found", 404)
    except Exception as e:
        logger.error(f"Error sending poster image '{path}': {e}", exc_info=True)
        return make_response("Internal server error", 500)


@app.route(f"/{UPLOADS_PREFIX}/<path:path>", methods=["GET"])
def send_uploaded_video(path: str) -> Response:
    """Serves video files from the uploads directory."""
    try:
        return send_from_directory(UPLOADS_PATH, path)
    except FileNotFoundError:
        logger.warning(f"Uploaded video resource not found: {path}")
        return make_response("Resource not found", 404)
    except Exception as e:
        logger.error(f"Error sending uploaded video '{path}': {e}", exc_info=True)
        return make_response("Internal server error", 500)


# --- Inference Streaming Routes ---
# (No changes needed in propagate_in_video or gen_track_with_mask_stream)
@app.route("/propagate_in_video", methods=["POST"])
def propagate_in_video() -> Response:
    """
    Handles POST requests to propagate masks in a video session.
    Expects JSON data with 'session_id' and optional 'start_frame_index'.
    Returns a multipart stream response using the 'multipart/x-savi-stream' mimetype.
    """
    if not request.is_json:
        logger.error("propagate_in_video: Request is not JSON")
        return make_response("Request must be JSON", 415)

    try:
        data = request.get_json()
        if data is None:
             logger.error("propagate_in_video: Failed to parse JSON body")
             return make_response("Invalid JSON body", 400)

        session_id = data.get("session_id")
        start_frame_index = data.get("start_frame_index", 0) # Default to 0 if not provided

        if not session_id:
            logger.error("propagate_in_video: Missing 'session_id' in request data")
            return make_response("Missing 'session_id'", 400)
        if not isinstance(start_frame_index, int) or start_frame_index < 0:
            logger.error(f"propagate_in_video: Invalid 'start_frame_index': {start_frame_index}")
            return make_response("Invalid 'start_frame_index'", 400)

        logger.info(f"Received propagate_in_video request for session_id: {session_id}, start_frame_index: {start_frame_index}")

        boundary = "frame" # Boundary used to separate parts in the stream

        # Ensure session exists before starting generation
        # Use lock for check to be safe with concurrent requests potentially closing sessions
        with inference_api.inference_lock:
            if session_id not in inference_api.session_states:
                logger.error(f"propagate_in_video: Session {session_id} not found.")
                return make_response(f"Session '{session_id}' not found", 404)

        frame_stream = gen_track_with_mask_stream(boundary, session_id, start_frame_index)
        # Set MIME type to 'multipart/x-savi-stream' as expected by the frontend
        return Response(frame_stream, mimetype=f"multipart/x-savi-stream; boundary={boundary}")

    except Exception as e:
        # Catch JSON parsing errors or other unexpected issues
        logger.error(f"Error processing propagate_in_video request: {e}", exc_info=True)
        return make_response("Internal server error during propagation setup", 500)


def gen_track_with_mask_stream(
    boundary: str,
    session_id: str,
    start_frame_index: int,
) -> Generator[bytes, None, None]:
    """
    Generator function to stream mask propagation results as multipart messages.
    Handles potential errors during generation.
    """
    logger.info(f"Starting mask stream generation for session: {session_id}, start_frame: {start_frame_index}")
    try:
        # Use the InferenceAPI method which handles autocast and locking
        request_obj = PropagateInVideoRequest(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_index,
        )

        # The propagate_in_video method is itself a generator
        for chunk in inference_api.propagate_in_video(request=request_obj):
            if not isinstance(chunk, PropagateDataResponse):
                 logger.warning(f"Unexpected chunk type ({type(chunk)}) received from propagate_in_video for session {session_id}, skipping.")
                 continue

            # Construct the multipart part
            try:
                 json_body = chunk.to_json() # Check if this can fail
                 part = MultipartResponseBuilder.build(
                    boundary=boundary,
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        # Include Frame-Index from the chunk if needed by client
                        "Frame-Index": str(chunk.frame_index),
                        "Mask-Type": "RLE", # Indicate mask format
                    },
                    body=json_body.encode("UTF-8"),
                 ).get_message()
                 yield part
            except Exception as build_exc:
                 logger.error(f"Failed to build or yield multipart chunk for frame {chunk.frame_index}, session {session_id}: {build_exc}", exc_info=True)
                 # Decide whether to break or continue
                 break # Stop streaming on error

        logger.info(f"Finished mask stream generation normally for session: {session_id}")

    except RuntimeError as e:
        # Catch errors specifically raised by InferenceAPI/operations (e.g., session not found, CUDA errors)
        logger.error(f"Runtime error during gen_track_with_mask_stream for session {session_id}: {e}", exc_info=True)
        # Propagate the error by raising it, allowing Flask to potentially catch it
        # or let the connection break naturally. Avoid yielding error parts unless client handles them.
        raise
    except Exception as e:
        logger.error(f"Unexpected error in gen_track_with_mask_stream for session {session_id}: {e}", exc_info=True)
        raise # Propagate unexpected errors
    finally:
         logger.debug(f"Exiting mask stream generator for session {session_id}")
         # Final cleanup if needed (e.g., releasing resources specific to this stream)


@app.route("/download_frames", methods=["POST"])
def download_frames() -> Response:
    """
    Endpoint to download video frames for a given session as a stream of JPEG images.
    NOTE: This endpoint provides a raw stream suitable for specific clients.
    For user downloads, prefer the /sessions/<session_id>/download_images_zip endpoint.
    """
    if not request.is_json:
        logger.error("download_frames: Request is not JSON")
        return make_response("Request must be JSON", 415)

    try:
        data = request.get_json()
        if data is None:
            logger.error("download_frames: Failed to parse JSON body")
            return make_response("Invalid JSON body", 400)

        session_id = data.get("session_id")
        if not session_id:
            logger.error("download_frames: Received request with missing session_id")
            return make_response("Missing session_id", 400)

        # Check session existence before starting the generator
        with inference_api.inference_lock:
            if session_id not in inference_api.session_states:
                logger.error(f"download_frames: Session {session_id} not found")
                return make_response(f"Session {session_id} not found", 404)

        logger.info(f"Processing download_frames request for session {session_id}")
        boundary = "frame"
        frame_stream_generator = gen_frames_stream(boundary, session_id)
        # Use multipart/x-mixed-replace for streaming images
        return Response(frame_stream_generator, mimetype=f"multipart/x-mixed-replace; boundary={boundary}")

    except Exception as e:
        logger.error(f"Error setting up download_frames stream for session {session_id}: {e}", exc_info=True)
        return make_response("Internal server error during frame download setup", 500)


def gen_frames_stream(boundary: str, session_id: str) -> Generator[bytes, None, None]:
    """
    Generator function to stream JPEG-encoded video frames as multipart responses.
    """
    logger.info(f"Starting frame streaming generation for session {session_id}")
    frames_yielded = 0
    try:
        # The download_frames method handles getting the session and is a generator
        for frame_idx, jpeg_bytes in inference_api.download_frames(session_id):
            try:
                 part = MultipartResponseBuilder.build(
                    boundary=boundary,
                    headers={
                        "Content-Type": "image/jpeg",
                        "Frame-Index": str(frame_idx),
                    },
                    body=jpeg_bytes,
                 ).get_message()
                 yield part
                 frames_yielded += 1
            except Exception as build_exc:
                 logger.error(f"Failed to build or yield frame {frame_idx} for session {session_id}: {build_exc}", exc_info=True)
                 break # Stop streaming on error

        logger.info(f"Completed frame streaming generation for session {session_id}. Yielded {frames_yielded} frames.")
    except RuntimeError as e:
        # Catch errors from InferenceAPI.download_frames (e.g., session gone, frame processing error)
        logger.error(f"Runtime error during frame streaming for session {session_id}: {e}", exc_info=True)
        raise # Propagate
    except Exception as e:
        logger.error(f"Unexpected error in gen_frames_stream for session {session_id}: {e}", exc_info=True)
        raise # Propagate
    finally:
         logger.debug(f"Exiting frame stream generator for session {session_id}")


# --- Session Data Routes (RESTful style) ---

# Endpoint to download RLE masks as JSON
@app.route('/sessions/<session_id>/download_masks', methods=['GET'])
def download_masks_route(session_id: str) -> Response:
    """
    Downloads RLE segmentation masks as a JSON object for the specified session.
    """
    logger.info(f"Received download_masks request for session {session_id}")
    try:
        # InferenceAPI.download_masks expects a DownloadMasksRequest object
        # Create one here based on the route parameter
        masks_req = DownloadMasksRequest(type="download_masks", session_id=session_id)
        response_data = inference_api.download_masks(masks_req)
        # Return the response dataclass as JSON
        return jsonify(response_data.to_dict())
    except RuntimeError as e:
        # Catch specific errors like session not found from inference_api
        logger.error(f"Error downloading masks for session {session_id}: {e}", exc_info=True)
        # Check if error message indicates session not found
        if "Cannot find session" in str(e) or "not found" in str(e).lower():
            return jsonify({"error": f"Session '{session_id}' not found or expired."}), 404
        else:
            return jsonify({"error": f"Server error generating masks: {str(e)}"}), 500
    except Exception as e:
        logger.exception(f"Unexpected error in /download_masks for session {session_id}")
        return jsonify({"error": "An unexpected server error occurred."}), 500

# --- NEW Routes for ZIP Downloads ---

@app.route('/sessions/<session_id>/download_images_zip', methods=['GET'])
def download_images_zip_route(session_id: str) -> Response:
    """
    Downloads original video frames as a zip archive.
    """
    logger.info(f"Received download_images_zip request for session {session_id}")
    try:
        zip_buffer: BytesIO = inference_api.download_images_zip(session_id)
        # Ensure buffer has content before sending
        if zip_buffer.getbuffer().nbytes == 0:
             logger.warning(f"Generated images zip for session {session_id} is empty.")
             return jsonify({"error": "Failed to generate zip file: No images found or processed."}), 404 # Or 500?

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{session_id}_images.zip' # Use sanitized session_id
        )
    except RuntimeError as e:
        logger.error(f"Error generating images zip for session {session_id}: {e}", exc_info=True)
        if "Cannot find session" in str(e) or "not found" in str(e).lower():
             return jsonify({"error": f"Session '{session_id}' not found or expired."}), 404
        else:
             return jsonify({"error": f"Server error generating images zip: {str(e)}"}), 500
    except Exception as e:
        logger.exception(f"Unexpected error in /download_images_zip for session {session_id}")
        return jsonify({"error": "An unexpected server error occurred."}), 500

@app.route('/sessions/<session_id>/download_yolo_labels', methods=['GET'])
def download_yolo_labels_route(session_id: str) -> Response:
    """
    Downloads YOLO format labels (labels/*.txt + classes.txt) as a zip archive.
    """
    logger.info(f"Received download_yolo_labels request for session {session_id}")
    try:
        zip_buffer: BytesIO = inference_api.download_yolo_labels(session_id)
        if zip_buffer.getbuffer().nbytes == 0:
             logger.warning(f"Generated YOLO labels zip for session {session_id} is empty.")
             return jsonify({"error": "Failed to generate zip file: No labels found or processed."}), 404

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{session_id}_yolo_labels.zip'
        )
    except RuntimeError as e:
        logger.error(f"Error generating YOLO labels zip for session {session_id}: {e}", exc_info=True)
        if "Cannot find session" in str(e) or "not found" in str(e).lower():
             return jsonify({"error": f"Session '{session_id}' not found or expired."}), 404
        elif "No mask data available" in str(e): # Catch specific error from downloader
             return jsonify({"error": "Cannot generate labels: No mask data found for this session."}), 404
        else:
             return jsonify({"error": f"Server error generating YOLO labels zip: {str(e)}"}), 500
    except Exception as e:
        logger.exception(f"Unexpected error in /download_yolo_labels for session {session_id}")
        return jsonify({"error": "An unexpected server error occurred."}), 500

@app.route('/sessions/<session_id>/download_yolo_format', methods=['GET'])
def download_yolo_format_route(session_id: str) -> Response:
    """
    Downloads full YOLO dataset (images/*, labels/*.txt, classes.txt) as a zip archive.
    """
    logger.info(f"Received download_yolo_format request for session {session_id}")
    try:
        zip_buffer: BytesIO = inference_api.download_yolo_format(session_id)
        if zip_buffer.getbuffer().nbytes == 0:
             logger.warning(f"Generated YOLO format zip for session {session_id} is empty.")
             return jsonify({"error": "Failed to generate zip file: No data found or processed."}), 404

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{session_id}_yolo_dataset.zip'
        )
    except RuntimeError as e:
        logger.error(f"Error generating YOLO format zip for session {session_id}: {e}", exc_info=True)
        if "Cannot find session" in str(e) or "not found" in str(e).lower():
             return jsonify({"error": f"Session '{session_id}' not found or expired."}), 404
        elif "No mask data available" in str(e) or "No original images found" in str(e):
             return jsonify({"error": "Cannot generate dataset: No mask or image data found for this session."}), 404
        else:
             return jsonify({"error": f"Server error generating YOLO format zip: {str(e)}"}), 500
    except Exception as e:
        logger.exception(f"Unexpected error in /download_yolo_format for session {session_id}")
        return jsonify({"error": "An unexpected server error occurred."}), 500


# --- GraphQL Setup ---
class MyGraphQLView(GraphQLView):
    """Custom GraphQLView to inject context."""
    def get_context(self, request: Request, response: Response) -> Any:
        """Injects the inference_api instance into the GraphQL context."""
        # This allows GraphQL resolvers to access the InferenceAPI instance
        return {"inference_api": inference_api}


app.add_url_rule(
    "/graphql",
    view_func=MyGraphQLView.as_view(
        "graphql_view",
        schema=schema,
        graphiql=True, # Enable GraphiQL interface in browser for dev
        allow_queries_via_get=False, # Usually False for mutations
        # Enable multipart request handling for file uploads (upload_video mutation)
        multipart_uploads_enabled=True,
        # Define max size/parts if needed, using default Strawberry settings for now
        # upload_max_parts=...,
        # upload_max_total_size=...,
    ),
)


# --- Video Management Routes ---
# (No changes needed in list_videos or clear_videos routes)
@app.route("/videos", methods=["GET"])
def list_videos() -> Response:
    """
    Endpoint to list all videos available in the store.

    Returns a JSON response containing a list of video objects, each with metadata
    like code, path, URL, poster URL, width, and height.
    """
    try:
        all_videos = get_videos() # Retrieve from the data store
        video_list = []
        for video in all_videos.values():
             try:
                 video_data = {
                     "code": video.code,
                     "path": video.path,
                     "url": video.url(), # Call method to get full URL
                     "posterUrl": video.poster_url() if video.poster_path else None, # Call method
                     "width": video.width,
                     "height": video.height,
                 }
                 video_list.append(video_data)
             except Exception as video_err:
                  logger.error(f"Error processing video data for '{video.code}': {video_err}", exc_info=True)
                  # Skip this video or add partial data? Skipping for now.

        logger.info(f"Listed {len(video_list)} videos available for labeling")
        return jsonify({"videos": video_list}), 200
    except Exception as e:
        logger.error(f"Failed to list videos: {str(e)}", exc_info=True)
        return make_response(f"Error listing videos: {str(e)}", 500)


def _clear_single_video(video_code: str, all_videos: dict) -> Tuple[bool, bool, bool]:
    """
    Helper function to clear a single video from the filesystem and the store.

    Args:
        video_code: The code (identifier) of the video to clear.
        all_videos: The current dictionary of all videos in the store (passed for lookup).

    Returns:
        Tuple[bool, bool, bool]: (video_found, video_deleted, poster_deleted)
    """
    if video_code not in all_videos:
        logger.warning(f"Video code '{video_code}' not found in store for clearing.")
        return False, False, False

    video = all_videos[video_code]
    video_path_relative = video.path
    poster_path_relative = video.poster_path

    deleted_video_file = False
    deleted_poster_file = False
    error_occurred = False # Flag specific errors during deletion

    # Construct full paths based on DATA_PATH
    full_video_path = DATA_PATH / video_path_relative
    full_poster_path = DATA_PATH / poster_path_relative if poster_path_relative else None

    # --- Delete Video File ---
    try:
        if full_video_path.exists():
            if full_video_path.is_file():
                 os.remove(full_video_path)
                 logger.info(f"Removed video file: {full_video_path}")
                 deleted_video_file = True
            else:
                 logger.warning(f"Path exists but is not a file (cannot remove): {full_video_path}")
                 error_occurred = True
        else:
            logger.warning(f"Video file not found at expected location for deletion: {full_video_path}")
            # Consider if this is an error or just means already gone
            # error_occurred = True # Treat missing file as an issue?

    except OSError as e:
        logger.error(f"Failed to remove video file {full_video_path} for video code {video_code}: {str(e)}", exc_info=True)
        error_occurred = True # Mark error if deletion fails

    # --- Delete Poster File ---
    if full_poster_path:
        try:
            if full_poster_path.exists():
                 if full_poster_path.is_file():
                    os.remove(full_poster_path)
                    logger.info(f"Removed poster file: {full_poster_path}")
                    deleted_poster_file = True
                 else:
                      logger.warning(f"Path exists but is not a file (cannot remove): {full_poster_path}")
                      error_occurred = True # Or just warning?
            else:
                 logger.warning(f"Poster file not found at expected location for deletion: {full_poster_path}")
                 # error_occurred = True # Treat missing file as an issue?

        except OSError as e:
            logger.error(f"Failed to remove poster file {full_poster_path} for video code {video_code}: {str(e)}", exc_info=True)
            error_occurred = True # Mark error if deletion fails

    # Return success based on finding the video and *not* encountering deletion errors
    # A missing file might not be considered an error in the clearing process itself.
    return True, deleted_video_file, deleted_poster_file # Report what was actually deleted


@app.route("/clear_videos", methods=["POST"])
def clear_videos() -> Response:
    """
    Endpoint to clear all or specific videos from the store and filesystem.
    Also attempts to clear associated session data.
    """
    if not request.is_json:
        logger.error("clear_videos: Request is not JSON")
        return make_response("Request must be JSON", 415)

    try:
        data = request.get_json()
        video_codes_to_clear = data.get("video_codes", []) if data else []

        current_videos = get_videos() # Get a mutable copy? No, get_videos returns the dict.
        videos_to_process = list(current_videos.keys()) # Get keys before potential modification
        codes_actually_cleared = [] # Store codes removed from the video *store*
        codes_not_found = []
        filesystem_errors = [] # Keep track of specific file deletion errors
        session_clear_errors = [] # Keep track of session clearing errors

        if not video_codes_to_clear: # If list is empty, clear all
            logger.info("Clearing all videos and associated session data requested.")
            target_codes = videos_to_process
        else:
            logger.info(f"Clearing specific videos requested: {video_codes_to_clear}")
            target_codes = video_codes_to_clear

        updated_videos = current_videos.copy() # Work on a copy

        for video_code in target_codes:
            if video_code not in updated_videos:
                 codes_not_found.append(video_code)
                 logger.warning(f"Video code '{video_code}' specified for clearing but not found in current video store.")
                 continue

            # --- Clear Filesystem Data ---
            video_found, video_deleted, poster_deleted = False, False, False
            try:
                 # Pass updated_videos for lookup, but deletion happens based on path
                video_found, video_deleted, poster_deleted = _clear_single_video(video_code, updated_videos)
                if not video_found: # Should not happen if check above passed, but defensive
                      codes_not_found.append(video_code)
                      logger.error(f"Consistency issue: Video code '{video_code}' was in store but helper couldn't find it.")
                      continue # Skip to next code

                 # Log success/partial success of file deletion
                if video_deleted or poster_deleted:
                      logger.info(f"Filesystem cleanup for video '{video_code}': Video Deleted={video_deleted}, Poster Deleted={poster_deleted}")
                else:
                      logger.warning(f"Filesystem cleanup for video '{video_code}': No files were deleted (might have been missing).")

            except Exception as fs_err: # Catch any error from _clear_single_video itself
                 logger.error(f"Error during filesystem cleanup for video '{video_code}': {fs_err}", exc_info=True)
                 filesystem_errors.append(video_code)
                 # Continue to try and remove from store and clear session data


            # --- Clear Session Data ---
            # Session ID is the sanitized video path (which is video.code for gallery/uploads)
            session_id_to_clear = video_code # Assuming video_code IS the sanitized path used as session ID
            session_dir_to_clear = SESSIONS_PATH / session_id_to_clear
            session_cleared_from_memory = False
            try:
                 # Close the session if it's active in memory (this also saves state one last time)
                if session_id_to_clear in inference_api.session_states:
                       logger.info(f"Closing active session '{session_id_to_clear}' before clearing data.")
                       close_req = CloseSessionRequest(type="close_session", session_id=session_id_to_clear)
                       close_resp = inference_api.close_session(close_req)
                       session_cleared_from_memory = close_resp.success
                       if not session_cleared_from_memory:
                             logger.warning(f"Failed to close active session '{session_id_to_clear}' cleanly from memory.")
                             # Proceed with directory removal anyway?

                 # Remove the persisted session directory
                if session_dir_to_clear.exists():
                     if session_dir_to_clear.is_dir():
                          import shutil
                          shutil.rmtree(session_dir_to_clear)
                          logger.info(f"Removed session data directory: {session_dir_to_clear}")
                     else:
                          logger.warning(f"Expected session directory, but found a file: {session_dir_to_clear}. Removing file.")
                          os.remove(session_dir_to_clear)

                else:
                      logger.info(f"No persisted session data directory found for session '{session_id_to_clear}'.")

            except Exception as session_err:
                 logger.error(f"Error clearing session data for '{session_id_to_clear}': {session_err}", exc_info=True)
                 session_clear_errors.append(session_id_to_clear)
                 # Continue to remove video from store

            # --- Remove from Video Store ---
            # Only remove from the store if filesystem and session clearing didn't have major issues?
            # Or remove regardless? Let's remove it from the list served by /videos.
            if video_code in updated_videos:
                 del updated_videos[video_code]
                 codes_actually_cleared.append(video_code)
                 logger.info(f"Removed video '{video_code}' from the active video store.")


        # Update the global video store *after* processing all targets
        set_videos(updated_videos)

        # --- Construct Response ---
        if not video_codes_to_clear:
            message = f"Attempted cleanup for all {len(videos_to_process)} videos. {len(codes_actually_cleared)} removed from store."
        else:
            message = f"Processed request to clear {len(target_codes)} videos. {len(codes_actually_cleared)} removed from store."

        response_data = {
             "message": message,
             "cleared_video_codes": codes_actually_cleared,
             "not_found_video_codes": codes_not_found,
             "filesystem_error_codes": filesystem_errors,
             "session_clear_error_codes": session_clear_errors
        }

        status_code = 200
        if filesystem_errors or session_clear_errors:
             status_code = 207 # Multi-Status: indicates partial success/failure
             logger.warning(f"Clear videos completed with errors. Filesystem errors: {filesystem_errors}, Session clear errors: {session_clear_errors}")
        elif codes_not_found and not codes_actually_cleared:
             status_code = 404 # If specific codes were requested but none found/cleared
             logger.warning(f"Clear videos: Specified codes not found: {codes_not_found}")

        logger.info(f"Clear videos finished. Status: {status_code}. Response: {response_data}")
        return jsonify(response_data), status_code

    except Exception as e:
        logger.error(f"Unexpected error in /clear_videos endpoint: {str(e)}", exc_info=True)
        return make_response(f"Error clearing videos: {str(e)}", 500)


# --- Main Execution ---
if __name__ == "__main__":
    # Use environment variables for host/port, fallback to defaults
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ["true", "1"]

    # Note: Flask's built-in server is not recommended for production.
    # The Dockerfile uses Gunicorn. This block is for local development.
    logger.info(f"Starting Flask development server on {host}:{port} (Debug: {debug_mode})")
    app.run(host=host, port=port, debug=debug_mode)