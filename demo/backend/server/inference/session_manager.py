# /home/delliott@int-episci.com/sam2/demo/backend/server/inference/session_manager.py
"""
Module for managing inference sessions.
Provides functions for retrieving a session, clearing session state,
generating session statistics, and handling persistence of object names.
"""

import logging
import torch
import json # Added import
import os # Added import
from pathlib import Path # Added import
from typing import Dict # Added import
from app_conf import SESSIONS_PATH # Added import

logger = logging.getLogger(__name__) # Define logger at module level

def get_session(session_states: dict, session_id: str):
    """
    Retrieve a session from the session_states dictionary.

    Args:
        session_states (dict): Dictionary containing active sessions.
        session_id (str): The ID of the session to retrieve.

    Returns:
        dict: The session data.

    Raises:
        RuntimeError: If the session is not found.
    """
    session = session_states.get(session_id, None)
    if session is None:
        raise RuntimeError(f"Cannot find session {session_id}; it might have expired")
    return session

def get_session_stats(session_states: dict) -> str:
    """
    Generate a summary string for all active sessions and GPU memory usage.

    Args:
        session_states (dict): Dictionary containing active sessions.

    Returns:
        str: A summary string.
    """
    live_session_strs = []
    for sid, sdata in session_states.items():
        # Ensure 'state' key exists before accessing sub-keys
        state = sdata.get("state", {})
        frames = state.get("num_frames", 0)
        objs = len(state.get("obj_ids", []))
        live_session_strs.append(f"'{sid}' ({frames} frames, {objs} objects)")

    if torch.cuda.is_available():
        try:
            mem_alloc = torch.cuda.memory_allocated() // 1024**2
            mem_reserved = torch.cuda.memory_reserved() // 1024**2
            mem_alloc_max = torch.cuda.max_memory_allocated() // 1024**2
            mem_reserved_max = torch.cuda.max_memory_reserved() // 1024**2
            mem_str = (f"{mem_alloc} MiB used, {mem_reserved} MiB reserved (max: "
                       f"{mem_alloc_max} / {mem_reserved_max})")
        except Exception as e:
            logger.warning(f"Could not get GPU memory stats: {e}")
            mem_str = "GPU memory stats unavailable"
    else:
        mem_str = "GPU not in use"
    session_stats_str = f"Live sessions: [{', '.join(live_session_strs)}], GPU mem: {mem_str}"
    return session_stats_str

def clear_session_state(session_states: dict, session_id: str) -> bool:
    """
    Remove a session from the session_states dictionary.
    Note: This does NOT clear persisted data on disk (like names.json).

    Args:
        session_states (dict): Dictionary of active sessions.
        session_id (str): The ID of the session to remove.

    Returns:
        bool: True if the session was successfully removed from memory; False otherwise.
    """
    session = session_states.pop(session_id, None)
    if session is None:
        logger.warning(
            f"Cannot close session {session_id} as it does not exist in memory; {get_session_stats(session_states)}"
        )
        return False
    else:
        logger.info(f"Removed session {session_id} from memory; {get_session_stats(session_states)}")
        # Consider clearing GPU cache if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

def _get_names_filepath(session_id: str) -> Path:
    """Helper function to get the full path to the names.json file."""
    session_dir = SESSIONS_PATH / session_id
    return session_dir / "names.json"

def save_object_names(session_id: str, names: Dict[int, str]) -> None:
    """
    Save the object names dictionary to names.json for the given session.

    Args:
        session_id (str): The ID of the session.
        names (Dict[int, str]): Dictionary mapping object_id to name.
    """
    names_file = _get_names_filepath(session_id)
    session_dir = names_file.parent
    try:
        os.makedirs(session_dir, exist_ok=True)
        # Convert keys to strings for JSON compatibility
        names_to_save = {str(k): v for k, v in names.items()}
        with open(names_file, "w") as f:
            json.dump(names_to_save, f, indent=2)
        logger.debug(f"Saved {len(names)} object names to {names_file}")
    except Exception as e:
        logger.error(f"Failed to save object names for session {session_id} to {names_file}: {e}", exc_info=True)
        # Optionally re-raise or handle the error appropriately
        raise IOError(f"Could not save object names: {e}")

def load_object_names(session_id: str) -> Dict[int, str]:
    """
    Load the object names dictionary from names.json for the given session.
    Returns an empty dictionary if the file doesn't exist or is invalid.

    Args:
        session_id (str): The ID of the session.

    Returns:
        Dict[int, str]: Dictionary mapping object_id to name.
    """
    names_file = _get_names_filepath(session_id)
    if not names_file.exists():
        logger.debug(f"Names file not found for session {session_id}, returning empty dict.")
        return {}

    try:
        with open(names_file, "r") as f:
            names_loaded = json.load(f)
        # Convert string keys back to integers
        names = {int(k): v for k, v in names_loaded.items()}
        logger.debug(f"Loaded {len(names)} object names from {names_file}")
        return names
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from names file {names_file} for session {session_id}: {e}")
        return {} # Return empty dict on decode error
    except Exception as e:
        logger.error(f"Failed to load object names for session {session_id} from {names_file}: {e}", exc_info=True)
        return {} # Return empty dict on other errors