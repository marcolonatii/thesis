#!/usr/bin/env python3
"""
Script to visualize YOLO bounding boxes on a video with colored boxes.

This script reads YOLO format text files from an input directory. Each text file represents
a frame and contains one or more lines in the following format:
    <object_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
where the coordinates are normalized (i.e., values between 0 and 1).
A black image of specified width and height is created for each frame, and bounding boxes 
are drawn on top of the black image. Each class (object_id) is visualized with a distinct color 
(up to 16 classes). The resulting frames are then compiled into a video.

Usage:
    python visualize_yolo_boxes.py --yolo-dir <YOLO_FILES_DIR> --output-video <OUTPUT_VIDEO_PATH> --img-width <WIDTH> --img-height <HEIGHT> [--fps <FPS>]

Requirements:
    pip install opencv-python numpy tqdm
"""

import argparse
import os
import glob
import cv2
import numpy as np
import logging
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO bounding boxes on a video with colored boxes for each class."
    )
    parser.add_argument(
        "--yolo-dir",
        required=True,
        help="Directory containing YOLO text files (one file per frame)."
    )
    parser.add_argument(
        "--output-video",
        required=True,
        help="Path to the output video file (e.g., output.mp4)."
    )
    parser.add_argument(
        "--img-width",
        type=int,
        required=True,
        help="Width of the output video frames."
    )
    parser.add_argument(
        "--img-height",
        type=int,
        required=True,
        help="Height of the output video frames."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second for the output video. Default is 25."
    )
    return parser.parse_args()

def get_class_color(object_id: int) -> tuple:
    """
    Get a color for a given class (object_id) from a predefined list of 16 colors.
    Colors are in BGR format as used by OpenCV.
    
    Args:
        object_id (int): The object/class id.
    
    Returns:
        tuple: A tuple representing the BGR color.
    """
    # Predefined list of 16 distinct colors (BGR format)
    colors = [
        (0, 0, 255),       # Red
        (0, 255, 0),       # Green
        (255, 0, 0),       # Blue
        (0, 255, 255),     # Yellow
        (255, 0, 255),     # Magenta
        (255, 255, 0),     # Cyan
        (128, 0, 0),       # Maroon
        (0, 128, 0),       # Dark Green
        (0, 0, 128),       # Navy
        (128, 128, 0),     # Olive
        (128, 0, 128),     # Purple
        (0, 128, 128),     # Teal
        (64, 0, 0),        # Dark Red
        (0, 64, 0),        # Darker Green
        (0, 0, 64),        # Dark Blue
        (64, 64, 64),      # Gray
    ]
    # Use modulo to cycle through colors if object_id exceeds 15
    return colors[object_id % len(colors)]

def read_yolo_file(file_path, img_width, img_height):
    """
    Read a YOLO file and convert the normalized bounding box values to absolute pixel values.
    
    Args:
        file_path (str): Path to the YOLO file.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        
    Returns:
        list of tuples: Each tuple is (object_id, x_min, y_min, x_max, y_max).
    """
    boxes = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        # Each line format: <object_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip lines that do not match the expected format
        try:
            object_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])
        except ValueError:
            continue  # skip invalid lines

        # Convert normalized coordinates to absolute values
        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height
        box_width = width_norm * img_width
        box_height = height_norm * img_height

        x_min = int(round(x_center - box_width / 2))
        y_min = int(round(y_center - box_height / 2))
        x_max = int(round(x_center + box_width / 2))
        y_max = int(round(y_center + box_height / 2))
        boxes.append((object_id, x_min, y_min, x_max, y_max))
    return boxes

def create_frame(img_width, img_height, boxes):
    """
    Create a black image and draw colored bounding boxes.
    
    Args:
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        boxes (list): List of bounding boxes (object_id, x_min, y_min, x_max, y_max).
        
    Returns:
        numpy.ndarray: The image with drawn boxes.
    """
    # Create a black image (3-channel for color)
    frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    # Draw each box with its corresponding class color and a thickness of 2 pixels
    for box in boxes:
        object_id, x_min, y_min, x_max, y_max = box
        color = get_class_color(object_id)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
    return frame

def main():
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    yolo_dir = args.yolo_dir
    output_video_path = args.output_video
    img_width = args.img_width
    img_height = args.img_height
    fps = args.fps

    # Get list of YOLO files, assuming they end with .txt
    yolo_files = sorted(glob.glob(os.path.join(yolo_dir, "*.txt")))
    if not yolo_files:
        logging.error(f"No YOLO files found in directory: {yolo_dir}")
        exit(1)
    
    logging.info(f"Found {len(yolo_files)} YOLO files in {yolo_dir}.")
    
    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (img_width, img_height))
    
    for yolo_file in tqdm(yolo_files, desc="Processing frames", unit="frame"):
        boxes = read_yolo_file(yolo_file, img_width, img_height)
        frame = create_frame(img_width, img_height, boxes)
        video_writer.write(frame)
    
    video_writer.release()
    logging.info(f"Video saved to {output_video_path}.")

if __name__ == "__main__":
    main()