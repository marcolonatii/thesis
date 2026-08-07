#!/usr/bin/env python3
"""
extract_frames.py
=================
Extract frames from an MP4 video into a zero-padded JPEG sequence
compatible with the VOS inference pipeline.

Output structure:
    <out_dir>/<video_name>/00000.jpg
    <out_dir>/<video_name>/00001.jpg
    ...

Usage:
    python extract_frames.py --video /path/to/video.mp4 --out_dir /path/to/frames
    python extract_frames.py --video /path/to/video.mp4 --out_dir /path/to/frames --fps 20
"""

import argparse
import sys
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from a video file.")
    p.add_argument("--video",   required=True, help="Path to the input video file.")
    p.add_argument("--out_dir", required=True, help="Root output directory. Frames go into <out_dir>/<video_stem>/.")
    p.add_argument("--fps",     type=float, default=None,
                   help="Target FPS to extract. None = extract every frame (default).")
    p.add_argument("--ext",     default="jpg", choices=["jpg", "png"],
                   help="Output image format (default: jpg).")
    p.add_argument("--quality", type=int, default=95,
                   help="JPEG quality 0-100 (default: 95). Ignored for PNG.")
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[error] Video not found: {video_path}")
        sys.exit(1)

    out_root = Path(args.out_dir) / video_path.stem
    out_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[error] Cannot open video: {video_path}")
        sys.exit(1)

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] Video: {video_path.name}")
    print(f"[info] Native FPS: {native_fps:.2f}  |  Total frames: {total_frames}")

    if args.fps is not None and args.fps < native_fps:
        frame_interval = round(native_fps / args.fps)
        effective_fps = native_fps / frame_interval
        print(f"[info] Extracting every {frame_interval} frame(s) → ~{effective_fps:.2f} FPS")
    else:
        frame_interval = 1
        print(f"[info] Extracting every frame ({native_fps:.2f} FPS)")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality] if args.ext == "jpg" else []

    saved = 0
    native_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if native_idx % frame_interval == 0:
            out_path = out_root / f"{saved:05d}.{args.ext}"
            cv2.imwrite(str(out_path), frame, encode_params)
            saved += 1
        native_idx += 1

    cap.release()
    print(f"[done] {saved} frames saved → {out_root}")


if __name__ == "__main__":
    main()
