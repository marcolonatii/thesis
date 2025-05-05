# /home/david_elliott/github/sam2/demo/backend/server/inference/state_persistor.py
"""
Module for persisting and loading inference session state (clicks, masks cache) to/from disk.
"""

import logging
import os
import json
from pathlib import Path
import torch

from inference.data_types import ClickData, PropagateDataResponse

logger = logging.getLogger(__name__)

def save_session_state(session_id: str, inference_state: dict, sessions_path: Path) -> None:
    """
    Save the session's clicks and masks cache to disk within the specified sessions path.

    Args:
        session_id (str): The ID of the session (typically the sanitized video path).
        inference_state (dict): The inference state containing 'points', 'labels', and 'masks_cache'.
        sessions_path (Path): The base directory where session data is stored.
    """
    session_dir = sessions_path / session_id
    try:
        os.makedirs(session_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create session directory {session_dir}: {e}", exc_info=True)
        # Decide if we should raise or just log and continue
        # For now, log and attempt to save files anyway, maybe the dir exists but perms changed?
        # If file saving fails, it will be logged there.

    # Save clicks
    clicks_file = session_dir / "clicks.json"
    clicks = []
    try:
        if "points" in inference_state and "labels" in inference_state:
            for frame_idx_str, points_dict in inference_state["points"].items():
                frame_idx = int(frame_idx_str) # Ensure frame_idx is int
                if frame_idx in inference_state["labels"]:
                    for obj_id_str, points_tensor in points_dict.items():
                        obj_id = int(obj_id_str) # Ensure obj_id is int
                        if obj_id in inference_state["labels"][frame_idx]:
                            labels_tensor = inference_state["labels"][frame_idx][obj_id]
                            # Ensure tensors are moved to CPU before converting to list
                            points_list = points_tensor.cpu().tolist()
                            labels_list = labels_tensor.cpu().tolist()
                            clicks.append(ClickData(
                                frame_index=frame_idx,
                                object_id=obj_id,
                                points=points_list,
                                labels=labels_list
                            ))
                        else:
                             logger.warning(f"Labels missing for object {obj_id} in frame {frame_idx} during save for session {session_id}")
                else:
                     logger.warning(f"Labels missing for frame {frame_idx} during save for session {session_id}")

        with open(clicks_file, "w") as f:
            json.dump([click.to_dict() for click in clicks], f, indent=2)
        logger.debug(f"Saved {len(clicks)} click entries to {clicks_file}")
    except Exception as e:
        logger.error(f"Failed to save clicks to {clicks_file} for session {session_id}: {e}", exc_info=True)

    # Save masks cache
    masks_file = session_dir / "masks.json"
    try:
        if "masks_cache" in inference_state and inference_state["masks_cache"]:
            # Ensure all items in cache are serializable PropagateDataResponse objects
            serializable_cache = {}
            num_saved = 0
            for frame_idx, mask_data in inference_state["masks_cache"].items():
                 if isinstance(mask_data, PropagateDataResponse):
                      serializable_cache[frame_idx] = mask_data.to_dict()
                      num_saved +=1
                 else:
                     logger.warning(f"Non-serializable data found in masks_cache for frame {frame_idx}, session {session_id}. Type: {type(mask_data)}. Skipping.")

            with open(masks_file, "w") as f:
                 json.dump(serializable_cache, f, indent=2) # Save as object {frame_idx: data}
            logger.debug(f"Saved masks cache for {num_saved} frames to {masks_file}")
        else:
            # If the cache is empty or doesn't exist, ensure the file is empty or removed
            if masks_file.exists():
                try:
                    os.remove(masks_file)
                    logger.debug(f"Removed empty or outdated masks cache file: {masks_file}")
                except OSError as e:
                    logger.error(f"Failed to remove empty masks cache file {masks_file}: {e}", exc_info=True)
            else:
                logger.debug(f"No masks cache to save for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save masks cache to {masks_file} for session {session_id}: {e}", exc_info=True)


def load_session_state(session_id: str, inference_state: dict, sessions_path: Path, device: torch.device) -> None:
    """
    Load the session's clicks and masks cache from disk into the inference state.

    Args:
        session_id (str): The ID of the session.
        inference_state (dict): The inference state dictionary to populate.
        sessions_path (Path): The base directory where session data is stored.
        device (torch.device): The device to load tensors onto.
    """
    session_dir = sessions_path / session_id
    if not session_dir.exists():
        logger.info(f"No persisted state found for session {session_id} at {session_dir}")
        return

    # Load clicks
    clicks_file = session_dir / "clicks.json"
    if clicks_file.exists():
        try:
            with open(clicks_file, "r") as f:
                clicks_data = json.load(f)
            loaded_clicks = [ClickData.from_dict(data) for data in clicks_data]

            # Reconstruct points and labels in inference_state
            inference_state["points"] = {}
            inference_state["labels"] = {}
            for click in loaded_clicks:
                frame_idx = click.frame_index
                obj_id = click.object_id
                if frame_idx not in inference_state["points"]:
                    inference_state["points"][frame_idx] = {}
                    inference_state["labels"][frame_idx] = {}
                # Load points and labels onto the correct device
                inference_state["points"][frame_idx][obj_id] = torch.tensor(click.points, device=device, dtype=torch.float32)
                inference_state["labels"][frame_idx][obj_id] = torch.tensor(click.labels, device=device, dtype=torch.int64)
            logger.debug(f"Loaded {len(loaded_clicks)} click entries from {clicks_file}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from clicks file: {clicks_file}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to load clicks from {clicks_file}: {e}", exc_info=True)

    # Load masks cache
    masks_file = session_dir / "masks.json"
    if masks_file.exists():
        try:
            with open(masks_file, "r") as f:
                 # Load as object {frame_idx: data}
                masks_cache_data = json.load(f)

            inference_state["masks_cache"] = {}
            loaded_count = 0
            for frame_idx_str, data in masks_cache_data.items():
                try:
                     # Convert frame_idx key back to int
                    frame_idx = int(frame_idx_str)
                    inference_state["masks_cache"][frame_idx] = PropagateDataResponse.from_dict(data)
                    loaded_count += 1
                except (ValueError, TypeError) as conversion_error:
                    logger.warning(f"Skipping invalid entry in masks cache for session {session_id}, frame key '{frame_idx_str}': {conversion_error}")
                except Exception as deser_error:
                     logger.warning(f"Failed to deserialize mask data for session {session_id}, frame key '{frame_idx_str}': {deser_error}", exc_info=True)

            logger.debug(f"Loaded masks cache for {loaded_count} frames from {masks_file}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from masks file: {masks_file}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to load masks cache from {masks_file}: {e}", exc_info=True)

    logger.info(f"Finished loading persisted state for session {session_id}")