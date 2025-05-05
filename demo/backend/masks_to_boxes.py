#!/usr/bin/env python3
"""
Script to process downloaded masks and convert them into YOLO format bounding boxes.

This script reads JSON mask files from an input directory. Each file is expected to contain 
a "frame_index" and a "rle_mask_list", where each entry has an "object_id" and an RLE mask (with 
"size", "counts", and optionally "order"). The script decodes each RLE mask using pycocotools, 
computes the corresponding bounding box in absolute coordinates, converts it to YOLO format 
(normalized center x, center y, width, and height), and saves the results to text files in the 
specified output directory.

The YOLO format for each bounding box is:
    <object_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

Usage:
    python process_masks_to_yolo_boxes.py --input-dir <INPUT_DIR> --output-dir <OUTPUT_DIR>

Requirements:
    pip install pycocotools numpy tqdm
"""

import argparse
import json
import os
import logging
from glob import glob

import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils


def compute_bbox_from_rle(rle: dict) -> list:
    """
    Compute the absolute bounding box for a given RLE mask.

    Args:
        rle (dict): RLE mask dictionary containing "counts" and "size".
    
    Returns:
        list: Absolute bounding box in the format [x, y, width, height].
    """
    bbox = maskUtils.toBbox(rle)
    # Convert the numpy array to a list of floats.
    return bbox.tolist()


def convert_bbox_to_yolo(bbox: list, img_size: list) -> list:
    """
    Convert an absolute bounding box to YOLO format.

    Args:
        bbox (list): Absolute bounding box [x, y, width, height].
        img_size (list): Image size in the format [height, width].

    Returns:
        list: YOLO format bounding box [x_center_norm, y_center_norm, width_norm, height_norm].
    """
    x, y, width, height = bbox
    img_h, img_w = img_size

    # Compute center coordinates.
    x_center = x + width / 2.0
    y_center = y + height / 2.0

    # Normalize the coordinates.
    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    width_norm = width / img_w
    height_norm = height / img_h

    return [x_center_norm, y_center_norm, width_norm, height_norm]


def process_mask_file(file_path: str) -> list:
    """
    Process a single mask JSON file and convert its RLE masks to YOLO bounding boxes.

    Args:
        file_path (str): Path to the mask JSON file.
        
    Returns:
        list: A list of strings, each representing one bounding box in YOLO format:
              "<object_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>"
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    rle_mask_list = data.get("rle_mask_list", [])
    yolo_lines = []
    for item in rle_mask_list:
        object_id = item.get("object_id")
        rle_mask = item.get("rle_mask", {})
        # Remove the "order" field if present; it is not used in bbox computation.
        if "order" in rle_mask:
            rle_mask = {k: v for k, v in rle_mask.items() if k != "order"}
        
        try:
            # Compute absolute bounding box [x, y, width, height].
            bbox = compute_bbox_from_rle(rle_mask)
            # Get image size from the rle mask ("size" is [height, width]).
            img_size = rle_mask.get("size")
            if img_size is None or len(img_size) != 2:
                raise ValueError("Invalid image size in RLE mask.")
            # Convert the bounding box to YOLO format.
            yolo_bbox = convert_bbox_to_yolo(bbox, img_size)
            # Format each value to 6 decimal places for consistency.
            yolo_line = "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                object_id, yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3]
            )
        except Exception as e:
            logging.error(
                f"Error computing YOLO bbox for object_id {object_id} in file {file_path}: {str(e)}"
            )
            # Skip this object if there is an error.
            continue
        
        yolo_lines.append(yolo_line)
    
    return yolo_lines


def save_yolo_file(yolo_lines: list, output_path: str) -> None:
    """
    Save the YOLO bounding box lines to a text file.

    Args:
        yolo_lines (list): List of YOLO formatted bounding box strings.
        output_path (str): Path to the output text file.
    """
    with open(output_path, "w") as f:
        for line in yolo_lines:
            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process downloaded mask JSON files and convert them into YOLO format bounding boxes."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing mask JSON files."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save YOLO bounding box text files."
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(input_dir):
        logging.error(f"Input directory {input_dir} does not exist.")
        exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in the input directory.
    mask_files = glob(os.path.join(input_dir, "*.json"))
    if not mask_files:
        logging.warning(f"No JSON mask files found in {input_dir}.")
        exit(0)
    
    logging.info(f"Found {len(mask_files)} mask files in {input_dir}.")
    
    for mask_file in tqdm(mask_files, desc="Processing mask files", unit="file"):
        try:
            yolo_lines = process_mask_file(mask_file)
            # Use the same base name as the input file but with a .txt extension.
            base_name = os.path.splitext(os.path.basename(mask_file))[0] + ".txt"
            output_path = os.path.join(output_dir, base_name)
            save_yolo_file(yolo_lines, output_path)
            logging.info(f"Processed file {mask_file} and saved YOLO boxes to {output_path}.")
        except Exception as e:
            logging.error(f"Failed to process file {mask_file}: {str(e)}")
    
    logging.info("Processing complete.")


if __name__ == "__main__":
    main()