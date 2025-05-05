# File: /home/david_elliott/github/sam2-git/demo/backend/server/inference/downloaders.py
# /home/delliott@int-episci.com/sam2/demo/backend/server/inference/downloaders.py
"""
Module for downloading frames and masks, and generating downloadable archives.
Provides functions to generate JPEG-encoded frames, compute/download masks,
and create zip archives for images and YOLO formatted data.
"""

import logging
import cv2
from PIL import Image
from io import BytesIO
import zipfile
import os
import pycocotools.mask as maskUtils # Import pycocotools
from typing import Generator, Tuple, List, Dict, Any, Optional

# Assuming InferenceAPI and other necessary types are accessible
# This might require adjustments based on actual project structure/imports
# Removed circular import: from .api import InferenceAPI
from .data_types import DownloadMasksResponse, PropagateDataResponse, Mask as RLEMaskData, DownloadMasksRequest # Added DownloadMasksRequest
from .session_manager import load_object_names # Import name loading function

logger = logging.getLogger(__name__)


def download_frames_operation(session: dict) -> Generator[Tuple[int, bytes], None, None]:
    """
    Generate a stream of JPEG-encoded frames from the original video data.

    Args:
        session (dict): The session dictionary containing 'images_original' in its state.

    Yields:
        Tuple[int, bytes]: Frame index and JPEG-encoded frame bytes.

    Raises:
        RuntimeError: If no original frames are found.
    """
    inference_state = session.get("state", {})
    if "images_original" not in inference_state:
        logger.error(f"No original frames found in session")
        raise RuntimeError("Session has no 'images_original' data")
    video_frames = inference_state["images_original"]
    num_frames = len(video_frames)
    logger.info(f"Found {num_frames} raw frames in session")
    if num_frames == 0:
        logger.warning("Session contains zero frames (original)")
        return
    for frame_idx, frame_bgr in enumerate(video_frames):
        try:
            # Ensure frame is usable (e.g., not None)
            if frame_bgr is None:
                logger.warning(f"Frame {frame_idx} is None, skipping encoding.")
                continue
            if frame_bgr.size == 0:
                logger.warning(f"Frame {frame_idx} is empty, skipping encoding.")
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            buf = BytesIO()
            frame_pil.save(buf, format="JPEG", quality=85)
            jpeg_bytes = buf.getvalue()
            buf.close()
            yield frame_idx, jpeg_bytes
        except Exception as e:
            logger.error(f"Failed to process frame {frame_idx}: {str(e)}", exc_info=True)
            # Continue to next frame instead of raising? Or raise to stop the stream?
            # Let's raise for now, as it indicates a problem processing the video data.
            raise RuntimeError(f"Error processing frame {frame_idx}: {str(e)}")

def download_masks_operation(predictor, inference_state: dict, num_frames: int, score_thresh=0) -> DownloadMasksResponse:
    """
    Retrieve masks for all frames. Uses a cache to avoid recomputation.

    Args:
        predictor: The SAM predictor instance.
        inference_state (dict): The current inference state.
        num_frames (int): Total number of frames.
        score_thresh: Score threshold to binarize masks.

    Returns:
        DownloadMasksResponse: Response containing masks for all frames.
    """
    if "masks_cache" not in inference_state:
        inference_state["masks_cache"] = {}
    results = []
    # Ensure imports are correct
    from .inference_operations import PropagateDataResponse # Assuming this defines the structure
    from .mask_utils import get_rle_mask_list # Assuming this creates RLEs

    for frame_idx in range(num_frames):
        if frame_idx in inference_state["masks_cache"]:
            # Ensure cached data is the correct type
            cached_data = inference_state["masks_cache"][frame_idx]
            if isinstance(cached_data, PropagateDataResponse):
                results.append(cached_data)
                continue # Use cached data and move to next frame
            else:
                logger.warning(f"Invalid data found in masks_cache for frame {frame_idx}, attempting regeneration.")
                # Fall through to regeneration logic below

        # If regeneration is needed or cache miss:
        mask_response = None
        try:
            outputs = predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1, # Only process current frame
                reverse=False,
            )

            # Process the generator output for the current frame
            for frame_idx_result, obj_ids, video_res_masks in outputs:
                if frame_idx_result == frame_idx: # Make sure we got the right frame
                    # Check if video_res_masks is valid before processing
                    if video_res_masks is None or video_res_masks.numel() == 0:
                        logger.warning(f"Predictor returned empty/invalid masks for frame {frame_idx_result}, skipping.")
                        # If predictor returns empty, should we cache an empty response? Yes.
                        mask_response = PropagateDataResponse(frame_index=frame_idx_result, results=[])
                        break # Break inner loop, go to cache update

                    masks_binary = (video_res_masks > score_thresh)[:, 0].cpu().numpy()
                    rle_mask_list = get_rle_mask_list(obj_ids, masks_binary)
                    mask_response = PropagateDataResponse(frame_index=frame_idx_result, results=rle_mask_list)
                    break # Got the frame we needed

        except Exception as e:
            logger.error(f"Error generating mask for frame {frame_idx}: {e}", exc_info=True)
            # Append an empty response for this frame if generation fails
            mask_response = PropagateDataResponse(frame_index=frame_idx, results=[])

        # Ensure mask_response is always a PropagateDataResponse
        if mask_response is None:
            logger.warning(f"No mask output generated for frame {frame_idx} despite processing.")
            mask_response = PropagateDataResponse(frame_index=frame_idx, results=[])

        inference_state["masks_cache"][frame_idx] = mask_response
        results.append(mask_response)


    return DownloadMasksResponse(results=results)


# --- NEW YOLO Formatting Helper ---
def _format_yolo_data(
    masks_response: DownloadMasksResponse,
    object_names: Dict[int, str]
) -> Tuple[Dict[int, List[str]], str, Dict[int, int]]:
    """
    Converts RLE mask data to YOLO format labels and generates classes file content.

    Args:
        masks_response: The response containing RLE masks per frame.
        object_names: A dictionary mapping object_id to object name.

    Returns:
        A tuple containing:
        - labels_by_frame (Dict[int, List[str]]): YOLO label lines grouped by frame index.
        - classes_content (str): Content for the classes.txt file.
        - object_id_to_class_index (Dict[int, int]): Mapping used for classes.txt.
    """
    labels_by_frame: Dict[int, List[str]] = {}
    object_id_to_class_index: Dict[int, int] = {}
    class_index_counter = 0
    classes_lines = []

    all_object_ids = set()
    for frame_result in masks_response.results:
        for mask_data in frame_result.results:
            all_object_ids.add(mask_data.object_id)

    # Sort object IDs for consistent class indexing
    sorted_object_ids = sorted(list(all_object_ids))

    # Create mapping and classes.txt content
    for obj_id in sorted_object_ids:
        if obj_id not in object_id_to_class_index:
            object_id_to_class_index[obj_id] = class_index_counter
            name = object_names.get(obj_id, f"object_{obj_id}") # Use default name if missing
            classes_lines.append(f"{name}") # Just the name per line for classes.txt
            class_index_counter += 1
    classes_content = "\n".join(classes_lines)

    # Convert masks to YOLO boxes and format lines
    for frame_result in masks_response.results:
        frame_index = frame_result.frame_index
        yolo_lines_for_frame = []
        for mask_data in frame_result.results:
            obj_id = mask_data.object_id
            # Ensure mask_data.mask is the expected type (RLEMaskData)
            if not hasattr(mask_data, 'mask') or not isinstance(mask_data.mask, RLEMaskData):
                 logger.warning(f"Unexpected mask data structure for obj {obj_id} frame {frame_index}. Skipping.")
                 continue
            rle_mask_data: RLEMaskData = mask_data.mask

            # Convert RLE to Bbox using pycocotools
            try:
                # pycocotools expects counts to be bytes
                # Handle potential empty counts string
                if not rle_mask_data.counts:
                    logger.warning(f"Empty RLE counts for object {obj_id} on frame {frame_index}. Skipping bbox conversion.")
                    continue

                rle_dict = {"counts": rle_mask_data.counts.encode('utf-8'), "size": rle_mask_data.size}
                # toBbox returns [[x,y,width,height]]
                bbox = maskUtils.toBbox(rle_dict) # No [0] needed if it returns a single list
                x, y, w, h = bbox
                img_h, img_w = rle_mask_data.size

                if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
                    logger.warning(f"Invalid dimensions for object {obj_id} on frame {frame_index}: img({img_w}x{img_h}), box({w}x{h}). Skipping.")
                    continue

                # Convert to YOLO format: [x_center/img_w, y_center/img_h, w/img_w, h/img_h]
                x_center_norm = (x + w / 2.0) / img_w
                y_center_norm = (y + h / 2.0) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # Clamp values to [0.0, 1.0] defensively
                x_center_norm = max(0.0, min(1.0, x_center_norm))
                y_center_norm = max(0.0, min(1.0, y_center_norm))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                class_index = object_id_to_class_index[obj_id]
                yolo_line = f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                yolo_lines_for_frame.append(yolo_line)

            except Exception as e:
                logger.error(f"Error converting RLE to YOLO for object {obj_id} on frame {frame_index}: {e}", exc_info=True)
                # Continue processing other objects/frames

        if yolo_lines_for_frame:
            labels_by_frame[frame_index] = yolo_lines_for_frame

    return labels_by_frame, classes_content, object_id_to_class_index

# --- NEW Download Operations ---

def download_images_zip_operation(session: dict) -> BytesIO:
    """
    Generates a zip archive containing all original video frames as JPEGs.

    Args:
        session (dict): The session dictionary.

    Returns:
        BytesIO: A byte stream containing the zip archive.

    Raises:
        RuntimeError: If frame generation fails or no frames are available.
    """
    logger.info("Starting creation of images zip archive.")
    zip_buffer = BytesIO()
    frame_generator = download_frames_operation(session) # Use the existing generator

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        frame_count = 0
        try:
            for frame_idx, jpeg_bytes in frame_generator:
                if jpeg_bytes:
                    # Use 6-digit padding for frame numbers
                    filename = f"frame_{frame_idx:06d}.jpg"
                    zipf.writestr(filename, jpeg_bytes)
                    frame_count += 1
            logger.info(f"Added {frame_count} frames to the images zip archive.")
        except Exception as e:
            logger.error(f"Error during image zip creation: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create images zip: {e}")

    if frame_count == 0:
        logger.warning("Images zip archive created but contains no frames.")
        # Optionally raise an error or return empty buffer? Let's return empty for now.

    zip_buffer.seek(0)
    return zip_buffer


# Type hint for InferenceAPI using forward reference (string) if needed,
# but it's passed as an argument, so the type hint in the signature works.
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#    from .api import InferenceAPI

def download_yolo_labels_operation(session_id: str, inference_api) -> BytesIO:
    """
    Generates a zip archive containing YOLO format label files and classes.txt.

    Args:
        session_id (str): The session ID.
        inference_api: Instance of the InferenceAPI to fetch masks.
                       Type hinted locally if needed or rely on runtime object.

    Returns:
        BytesIO: A byte stream containing the zip archive.

    Raises:
        RuntimeError: If session not found, mask retrieval fails, or zip creation fails.
    """
    logger.info(f"Starting creation of YOLO labels zip archive for session {session_id}.")
    zip_buffer = BytesIO()

    try:
        # 1. Get Mask Data using the InferenceAPI instance passed as argument
        masks_request = DownloadMasksRequest(type="download_masks", session_id=session_id)
        # Call the download_masks *method* of the passed inference_api instance
        masks_response = inference_api.download_masks(masks_request)
        if not masks_response or not masks_response.results:
            logger.warning(f"No mask data found for session {session_id}. Cannot generate YOLO labels.")
            # Return an empty zip instead of raising an error? Or raise?
            # Let's raise a specific error that the route handler can catch.
            raise RuntimeError("No mask data available for session {session_id}")


        # 2. Get Object Names
        object_names = load_object_names(session_id)
        logger.info(f"Loaded {len(object_names)} object names for session {session_id}.")

        # 3. Format YOLO data
        labels_by_frame, classes_content, _ = _format_yolo_data(masks_response, object_names)

        # 4. Create Zip
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add classes.txt
            if classes_content:
                zipf.writestr("classes.txt", classes_content.encode('utf-8'))
                logger.debug("Added classes.txt to zip.")
            else:
                logger.warning("No classes.txt content generated.")

            # Add label files
            label_files_count = 0
            for frame_index, yolo_lines in labels_by_frame.items():
                if yolo_lines:
                    filename = f"labels/frame_{frame_index:06d}.txt"
                    label_content = "\n".join(yolo_lines)
                    zipf.writestr(filename, label_content.encode('utf-8'))
                    label_files_count += 1
            logger.info(f"Added {label_files_count} label files to the zip archive.")
            if label_files_count == 0 and not classes_content:
                 logger.warning("YOLO labels zip archive created but contains no labels or classes.")
                 # Return empty zip in this case.

    except RuntimeError as e: # Catch specific errors like "No mask data"
        logger.error(f"Error creating YOLO labels zip for session {session_id}: {e}")
        raise # Re-raise specific runtime errors
    except Exception as e:
        logger.error(f"Unexpected error creating YOLO labels zip for session {session_id}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create YOLO labels zip: {e}") from e # Wrap unexpected errors

    zip_buffer.seek(0)
    return zip_buffer


def download_yolo_format_operation(session_id: str, inference_api) -> BytesIO:
    """
    Generates a zip archive containing images, YOLO labels, and classes.txt.

    Args:
        session_id (str): The session ID.
        inference_api: Instance of the InferenceAPI.

    Returns:
        BytesIO: A byte stream containing the zip archive.

    Raises:
        RuntimeError: If session not found, data retrieval fails, or zip creation fails.
    """
    logger.info(f"Starting creation of full YOLO format zip archive for session {session_id}.")
    zip_buffer = BytesIO()

    try:
        # 1. Get Session State (for images)
        # Access session state via the passed inference_api instance
        session = inference_api.session_states.get(session_id)
        if not session:
            raise RuntimeError(f"Session {session_id} not found for YOLO format download.")
        inference_state = session.get("state", {})
        if "images_original" not in inference_state or not inference_state["images_original"]:
            raise RuntimeError(f"No original images found in session {session_id}.")

        # 2. Get Mask Data
        masks_request = DownloadMasksRequest(type="download_masks", session_id=session_id)
        # Call download_masks method via the instance
        masks_response = inference_api.download_masks(masks_request)
        if not masks_response or not masks_response.results:
            logger.warning(f"No mask data found for session {session_id}. YOLO format zip will lack labels.")
            # Allow creating zip with just images and classes? Or fail?
            # Let's raise an error if masks are missing, as labels are core to YOLO format.
            raise RuntimeError("No mask data available for session {session_id}")


        # 3. Get Object Names
        object_names = load_object_names(session_id)
        logger.info(f"Loaded {len(object_names)} object names for session {session_id}.")

        # 4. Format YOLO data
        labels_by_frame, classes_content, _ = _format_yolo_data(masks_response, object_names)

        # 5. Create Zip
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add classes.txt
            if classes_content:
                zipf.writestr("classes.txt", classes_content.encode('utf-8'))
                logger.debug("Added classes.txt to zip.")
            else:
                logger.warning("No classes.txt content generated.")

            # Add label files
            label_files_count = 0
            for frame_index, yolo_lines in labels_by_frame.items():
                if yolo_lines:
                    # Ensure labels directory exists conceptually in zip
                    filename = f"labels/frame_{frame_index:06d}.txt"
                    label_content = "\n".join(yolo_lines)
                    zipf.writestr(filename, label_content.encode('utf-8'))
                    label_files_count += 1
            logger.info(f"Added {label_files_count} label files to the zip archive.")

            # Add images
            image_files_count = 0
            # Use the same session dict retrieved earlier
            frame_generator = download_frames_operation(session) # Generate frames again
            for frame_idx, jpeg_bytes in frame_generator:
                if jpeg_bytes:
                    # Ensure images directory exists conceptually in zip
                    filename = f"images/frame_{frame_idx:06d}.jpg"
                    zipf.writestr(filename, jpeg_bytes)
                    image_files_count += 1
            logger.info(f"Added {image_files_count} image files to the zip archive.")

            # Sanity check counts
            expected_frames = inference_state.get("num_frames", -1)
            if image_files_count != expected_frames:
                logger.warning(f"Number of images added to zip ({image_files_count}) does not match expected frame count ({expected_frames})")
            # Check if label files cover all images (optional, depends on requirements)
            # if label_files_count != image_files_count and label_files_count > 0:
            #    logger.warning(f"Mismatch between number of label files ({label_files_count}) and image files ({image_files_count}).")
            if image_files_count == 0 and label_files_count == 0 and not classes_content:
                 logger.warning("YOLO format zip archive created but is empty.")


    except RuntimeError as e: # Catch specific errors like "No mask data" or "No images found"
        logger.error(f"Error creating YOLO format zip for session {session_id}: {e}")
        raise # Re-raise specific runtime errors
    except Exception as e:
        logger.error(f"Unexpected error creating YOLO format zip for session {session_id}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create YOLO format zip: {e}") from e # Wrap unexpected errors

    zip_buffer.seek(0)
    return zip_buffer