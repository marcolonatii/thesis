#!/usr/bin/env python3

"""
Script to download all masks for a given session from the SAM 2 backend and save them to disk.

Usage:
    python download_masks.py --session-id <SESSION_ID> --output-dir <OUTPUT_DIR> [--endpoint <ENDPOINT>]

Requirements:
    - pip install gql aiohttp tqdm
"""

import argparse
import json
import os
import logging
from typing import Dict, List

from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from tqdm import tqdm


def download_masks(session_id: str, endpoint: str) -> List[Dict]:
    """
    Download all masks for a given session from the backend.

    Args:
        session_id: The ID of the session to download masks for.
        endpoint: The GraphQL endpoint URL (e.g., "http://localhost:5000/graphql").

    Returns:
        A list of dictionaries containing frame_index and rle_mask_list data.

    Raises:
        Exception: If the query fails or the response is invalid.
    """
    logging.info("Setting up GraphQL client with endpoint %s", endpoint)
    # Set up the GraphQL client
    transport = AIOHTTPTransport(url=endpoint)
    client = Client(transport=transport, fetch_schema_from_transport=True, execute_timeout=3600)

    # Define the GraphQL mutation
    query = gql("""
        mutation DownloadMasks($input: DownloadMasksInput!) {
            downloadMasks(input: $input) {
                masks {
                    frameIndex
                    rleMaskList {
                        objectId
                        rleMask {
                            size
                            counts
                            order
                        }
                    }
                }
            }
        }
    """)

    # Execute the query with the session ID
    variables = {"input": {"sessionId": session_id}}
    logging.info("Executing GraphQL query for session ID: %s", session_id)
    try:
        result = client.execute(query, variable_values=variables)
        masks = result["downloadMasks"]["masks"]
        logging.info("Downloaded %d masks for session %s", len(masks), session_id)
        return masks
    except Exception as e:
        logging.exception("Failed to download masks for session %s", session_id)
        raise Exception(f"Failed to download masks: {str(e)}")


def save_masks_to_disk(masks: List[Dict], output_dir: str) -> None:
    """
    Save the downloaded masks to disk as JSON files, one per frame.

    Args:
        masks: List of mask data dictionaries from the backend.
        output_dir: Directory where mask files will be saved.

    Raises:
        OSError: If the output directory cannot be created or written to.
    """
    logging.info("Ensuring output directory %s exists", output_dir)
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Saving masks to disk with progress bar")
    # Iterate over masks with a progress bar
    for mask_data in tqdm(masks, desc="Saving masks", unit="frame"):
        frame_index = mask_data["frameIndex"]
        filename = os.path.join(output_dir, f"frame_{frame_index:06d}.json")
        
        # Prepare the data to save (consistent with RLEMaskListOnFrame structure)
        data_to_save = {
            "frame_index": mask_data["frameIndex"],
            "rle_mask_list": [
                {
                    "object_id": rle_mask["objectId"],
                    "rle_mask": {
                        "size": rle_mask["rleMask"]["size"],
                        "counts": rle_mask["rleMask"]["counts"],
                        "order": rle_mask["rleMask"]["order"],
                    },
                }
                for rle_mask in mask_data["rleMaskList"]
            ],
        }

        try:
            with open(filename, "w") as f:
                json.dump(data_to_save, f, indent=2)
            logging.debug("Saved masks for frame %d to %s", frame_index, filename)
        except OSError as os_err:
            logging.error("Failed to write file %s: %s", filename, str(os_err))
            raise


def main():
    """Main function to parse arguments and execute the mask download."""
    parser = argparse.ArgumentParser(
        description="Download masks for a session from the SAM 2 backend and save them to disk."
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="The session ID to download masks for."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where mask files will be saved."
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:5000/graphql",
        help="GraphQL endpoint URL (default: http://localhost:5000/graphql)."
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logging.info("Starting mask download for session %s", args.session_id)
    try:
        # Download masks from the backend
        masks = download_masks(args.session_id, args.endpoint)
        if not masks:
            logging.warning("No masks found for session %s", args.session_id)
            return

        # Save masks to disk
        save_masks_to_disk(masks, args.output_dir)
        logging.info("Successfully downloaded and saved %d frames of masks.", len(masks))
    except Exception as e:
        logging.exception("An error occurred during processing:")
        exit(1)


if __name__ == "__main__":
    main()