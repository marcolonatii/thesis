#!/usr/bin/env python3

"""
Script to download bounding boxes for a given session from the SAM 2 backend
and save them to disk as YOLO format labels and a classes.txt file.

Usage:
    python download_boxes.py [--session-id <SESSION_ID>] --output-dir <OUTPUT_DIR> [--endpoint <ENDPOINT_DOMAIN>]

- If --session-id is not provided, the script queries active sessions and prompts for selection.
- Object names are fetched from the backend.
- A classes.txt file is generated with unique object names, sorted alphabetically.
- Label files (frame_*.txt) are generated using 0-based class indices corresponding
  to the order in classes.txt.
- Default names like "object_{id}" are used if a name isn't provided by the backend.

The endpoint domain should point to the server where the backend is hosted.
The script constructs the GraphQL endpoint URL by appending '/graphql'.

Requirements:
    - pip install requests tqdm
"""

import argparse
import logging
import os
import math
from typing import List, Dict, Tuple, Optional # Added Tuple, Optional

import requests
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- GraphQL Interaction ---

def list_sessions(graphql_endpoint: str) -> List[Dict]:
    """Retrieve a list of active inference sessions from the backend."""
    logging.info("Fetching list of active sessions from %s", graphql_endpoint)
    query = """
    query {
        sessions {
            sessionId
            startTime
            lastUseTime
            numFrames
            numObjects
        }
    }
    """
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(graphql_endpoint, json={"query": query}, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        if "errors" in result:
            logging.error("GraphQL errors: %s", result["errors"])
            raise Exception(f"GraphQL error fetching sessions: {result['errors'][0]['message']}")
        sessions = result.get("data", {}).get("sessions")
        if sessions is None:
             raise Exception("Invalid response structure: 'data.sessions' not found.")
        logging.info("Found %d active sessions", len(sessions))
        return sessions
    except requests.exceptions.RequestException as e:
        logging.error("HTTP request failed: %s", e)
        raise Exception(f"HTTP request failed: {str(e)}")
    except Exception as e:
        logging.exception("Failed to list sessions")
        raise Exception(f"Failed to list sessions: {str(e)}")


def download_boxes(session_id: str, graphql_endpoint: str) -> List[Dict]:
    """
    Download bounding boxes and names for a given session from the backend.

    Args:
        session_id: The ID of the session.
        graphql_endpoint: The GraphQL endpoint URL.

    Returns:
        List of dictionaries per frame:
        {
            "frameIndex": int,
            "boxes": [
                {
                    "objectId": int,
                    "name": Optional[str], # Added name
                    "box": [x_center, y_center, width, height]
                }, ...
            ]
        }

    Raises:
        Exception: If the GraphQL request fails or the response is invalid.
    """
    logging.info("Setting up GraphQL query for endpoint %s", graphql_endpoint)
    # *** Updated Query to include 'name' ***
    query = """
    mutation DownloadBoxes($input: DownloadBoxesInput!) {
        downloadBoxes(input: $input) {
            boxes {
                frameIndex # Corrected field name from python version
                boxes {
                    objectId # Corrected field name from python version
                    name    # <-- Added name field
                    box
                }
            }
        }
    }
    """
    variables = {
        "input": {
            "sessionId": session_id,
            "format": "yolo"
        }
    }
    headers = {"Content-Type": "application/json"}
    logging.info("Requesting boxes for session: %s", session_id)
    try:
        # Increased timeout for potentially long operations on the backend
        response = requests.post(graphql_endpoint, json={"query": query, "variables": variables}, headers=headers, timeout=1800) # 30 min timeout
        response.raise_for_status() # Raise HTTP errors
        result = response.json()
        if "errors" in result:
            logging.error("GraphQL errors: %s", result["errors"])
            raise Exception(f"GraphQL error downloading boxes: {result['errors'][0]['message']}")

        download_data = result.get("data", {}).get("downloadBoxes")
        if download_data is None:
             raise Exception("Invalid response structure: 'data.downloadBoxes' not found.")

        boxes_per_frame = download_data.get("boxes")
        if boxes_per_frame is None:
             raise Exception("Invalid response structure: 'data.downloadBoxes.boxes' not found.")

        logging.info("Downloaded bounding boxes for %d frames", len(boxes_per_frame))
        return boxes_per_frame
    except requests.exceptions.RequestException as e:
        logging.error("HTTP request failed: %s", e)
        raise Exception(f"HTTP request failed: {str(e)}")
    except Exception as e:
        logging.exception("Failed to download bounding boxes for session %s", session_id)
        # Don't raise the original low-level error, raise a more informative one
        raise Exception(f"Failed to download bounding boxes: {str(e)}")


# --- Data Processing ---

def process_names_and_create_mapping(boxes_data: List[Dict]) -> Tuple[Dict[int, int], str]:
    """
    Processes downloaded box data to extract unique names, create a sorted
    class list, and generate a mapping from original objectId to the new
    0-based class index.

    Args:
        boxes_data: The list of dictionaries returned by download_boxes.

    Returns:
        A tuple containing:
        - object_id_map (Dict[int, int]): Mapping from original objectId to new 0-based class index.
        - classes_content (str): Newline-separated string of sorted unique class names.
    """
    logging.info("Processing object names and creating class mapping...")
    object_info: Dict[int, str] = {} # Stores objectId -> name

    # Extract all object IDs and their names
    for frame_data in boxes_data:
        for box_obj in frame_data.get("boxes", []):
            obj_id = box_obj.get("objectId")
            name = box_obj.get("name")

            if obj_id is None:
                logging.warning("Found box data without objectId in frame %d. Skipping.", frame_data.get("frameIndex", -1))
                continue

            # Use default name if backend provides None or empty string, or if name field is missing
            if not name:
                name = f"object_{obj_id}"
                logging.debug("Using default name '%s' for objectId %d", name, obj_id)

            # Store the name. If an object appears multiple times, use the first valid name found.
            if obj_id not in object_info:
                 object_info[obj_id] = name
            elif object_info[obj_id].startswith("object_") and not name.startswith("object_"):
                 # Prefer a non-default name if encountered later
                 object_info[obj_id] = name


    if not object_info:
        logging.warning("No objects with names found in the downloaded data.")
        return {}, ""

    # Create sorted list of unique names
    unique_names = sorted(list(set(object_info.values())))
    logging.info("Found %d unique object names: %s", len(unique_names), unique_names)

    # Create mapping from name to 0-based index
    name_to_class_index = {name: idx for idx, name in enumerate(unique_names)}

    # Create mapping from original objectId to new 0-based index
    object_id_map: Dict[int, int] = {}
    for obj_id, name in object_info.items():
        object_id_map[obj_id] = name_to_class_index[name] # Name will always be in the map

    # Create content for classes.txt
    classes_content = "\n".join(unique_names)

    logging.info("Created mapping for %d object IDs.", len(object_id_map))
    return object_id_map, classes_content


# --- File Saving ---

def save_yolo_data_to_disk(
    boxes_data: List[Dict],
    output_dir: str,
    object_id_map: Dict[int, int],
    classes_content: str
) -> None:
    """
    Saves the processed bounding boxes as YOLO label files and writes classes.txt.

    - Creates `classes.txt` in the output directory.
    - Creates label files (`frame_*.txt`) using the mapped 0-based class indices.
    - Performs validation on bounding box coordinates.

    Args:
        boxes_data: List of dictionaries containing bounding boxes per frame.
        output_dir: Directory where the files will be saved.
        object_id_map: Mapping from original objectId to new 0-based class index.
        classes_content: String content for the classes.txt file.
    """
    logging.info("Ensuring output directory %s exists", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save classes.txt
    classes_filepath = os.path.join(output_dir, "classes.txt")
    try:
        with open(classes_filepath, "w") as f:
            f.write(classes_content)
        logging.info("Saved class names to %s", classes_filepath)
    except IOError as e:
        logging.error("Failed to write classes.txt: %s", e)
        # Decide if this is fatal or if we should continue writing labels
        raise Exception(f"Failed to write classes.txt: {str(e)}")


    logging.info("Saving YOLO label files to disk...")
    frames_saved_count = 0
    boxes_saved_count = 0
    for frame_data in tqdm(boxes_data, desc="Saving labels", unit="frame"):
        frame_index = frame_data.get("frameIndex")
        if frame_index is None:
            logging.warning("Skipping frame data with missing 'frameIndex'.")
            continue

        filename = os.path.join(output_dir, f"frame_{frame_index:06d}.txt")
        lines = []
        has_valid_boxes = False # Track if any boxes are written for this frame

        for box_obj in frame_data.get("boxes", []):
            original_object_id = box_obj.get("objectId")
            box_values = box_obj.get("box")

            if original_object_id is None:
                logging.warning("Skipping box with missing objectId in frame %d", frame_index)
                continue

            # Map original objectId to new class index
            if original_object_id not in object_id_map:
                 logging.warning("Skipping box for objectId %d in frame %d: ID not found in class map (perhaps no name was assigned).", original_object_id, frame_index)
                 continue
            class_index = object_id_map[original_object_id]

            # --- Bounding Box Validation ---
            if not isinstance(box_values, list) or len(box_values) != 4:
                logging.warning("Skipping invalid bounding box for frame %d, obj %d: expected list of 4 elements, got %s", frame_index, original_object_id, box_values)
                continue
            if not all(isinstance(val, (int, float)) and math.isfinite(val) for val in box_values):
                logging.warning("Skipping invalid bounding box for frame %d, obj %d: non-numeric or non-finite values in %s", frame_index, original_object_id, box_values)
                continue

            x_center, y_center, width, height = box_values

            if not (0 <= x_center <= 1) or not (0 <= y_center <= 1):
                logging.warning("Skipping invalid bounding box for frame %d, obj %d: center coordinates out of range (x=%.6f, y=%.6f)", frame_index, original_object_id, x_center, y_center)
                continue
            if not (0 < width <= 1) or not (0 < height <= 1):
                logging.warning("Skipping invalid bounding box for frame %d, obj %d: dimensions out of range (0, 1] (w=%.6f, h=%.6f)", frame_index, original_object_id, width, height)
                continue
            # --- End Validation ---

            line = f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            lines.append(line)
            boxes_saved_count += 1
            has_valid_boxes = True

        # Only write file if it contains valid boxes for that frame
        if has_valid_boxes:
            try:
                with open(filename, "w") as f:
                    f.write("\n".join(lines))
                frames_saved_count += 1
            except IOError as e:
                 logging.error("Failed to write label file %s: %s", filename, e)
                 # Continue to next frame
        # Do not log every single frame save unless debugging is needed
        # logging.debug("Saved labels for frame %d to %s", frame_index, filename)

    logging.info("Finished saving data. Saved %d label files and %d total bounding boxes.", frames_saved_count, boxes_saved_count)


# --- User Interaction ---

def select_session(sessions: List[Dict]) -> str:
    """Prompts the user to select a session from a list."""
    print("\nMultiple active sessions found:")
    for i, session in enumerate(sessions, 1):
        start_time_str = f"{session.get('startTime', 0):.0f}" # Basic timestamp format
        print(f"{i}. Session ID: {session.get('sessionId', 'N/A')}, "
              f"Frames: {session.get('numFrames', 'N/A')}, "
              f"Objects: {session.get('numObjects', 'N/A')}")
    print(f"{len(sessions) + 1}. Exit")

    while True:
        try:
            choice = int(input(f"\nEnter the number of the session to download (1-{len(sessions) + 1}): "))
            if choice == len(sessions) + 1:
                logging.info("User chose to exit")
                raise SystemExit("Exiting at user request.")
            if 1 <= choice <= len(sessions):
                selected_session_id = sessions[choice - 1].get("sessionId")
                if selected_session_id:
                    logging.info("User selected session %s", selected_session_id)
                    return selected_session_id
                else:
                    print("Error: Selected session data is invalid (missing sessionId). Please try again.")
            else:
                print(f"Please enter a number between 1 and {len(sessions) + 1}.")
        except ValueError:
            print("Please enter a valid number.")
        except IndexError:
             print("Internal error: Choice index out of range. Please try again.")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Download YOLO format labels and classes.txt for a session from the SAM 2 backend."
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="The session ID to download data for. If omitted, active sessions will be listed."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the label files (frame_*.txt) and classes.txt."
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:5000",
        help="Backend endpoint domain (e.g., http://localhost:5000 or https://your.domain.com). Default: http://localhost:5000"
    )

    args = parser.parse_args()

    # Construct the GraphQL endpoint URL
    graphql_endpoint = args.endpoint.rstrip("/") + "/graphql"
    logging.info("Using GraphQL endpoint: %s", graphql_endpoint)

    try:
        session_id_to_use = args.session_id
        if not session_id_to_use:
            logging.info("Session ID not provided, listing active sessions...")
            sessions = list_sessions(graphql_endpoint)
            if not sessions:
                logging.error("No active sessions found.")
                print("\nError: No active sessions found on the backend.")
                print(f"Please start a session via the UI or ensure the backend is running at {args.endpoint}.")
                exit(1)
            elif len(sessions) == 1:
                session_id_to_use = sessions[0].get("sessionId")
                if not session_id_to_use:
                     logging.error("The only active session has an invalid ID.")
                     print("\nError: Found one session, but its ID is invalid.")
                     exit(1)
                logging.info("Automatically selected the only active session: %s", session_id_to_use)
                print(f"\nUsing the only active session: {session_id_to_use}")
            else:
                session_id_to_use = select_session(sessions) # Handles exit internally

        logging.info("Starting data download for session: %s", session_id_to_use)
        boxes_result = download_boxes(session_id_to_use, graphql_endpoint)

        if not boxes_result:
            logging.warning("No bounding box data returned from the backend for session %s.", session_id_to_use)
            print(f"\nWarning: No bounding box data found for session {session_id_to_use}.")
            # Exit gracefully even if no boxes, maybe the video had none
            exit(0)

        # Process names and create mappings
        object_id_map, classes_content = process_names_and_create_mapping(boxes_result)

        if not classes_content:
             logging.warning("No class names generated (no objects found?). Saving empty classes.txt and no labels.")
             # Still save empty classes.txt for consistency
             save_yolo_data_to_disk([], args.output_dir, {}, "")
        else:
            # Save the data using the new function
            save_yolo_data_to_disk(boxes_result, args.output_dir, object_id_map, classes_content)

        print(f"\nSuccessfully saved YOLO data to {args.output_dir}")

    except SystemExit as e:
        # Raised by select_session on user exit
        print(f"\n{str(e)}")
        exit(0)
    except Exception as e:
        logging.exception("An error occurred:")
        print(f"\nError: {str(e)}")
        print("Please check the logs and ensure the backend is running and accessible.")
        exit(1)


if __name__ == "__main__":
    main()