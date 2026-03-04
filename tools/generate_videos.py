import os
import cv2
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Create overlay videos from frames and binary masks.")
    parser.add_argument("--frames", required=True, help="Path to frames root folder")
    parser.add_argument("--masks", required=True, help="Path to predictions root folder")
    parser.add_argument("--output", required=True, help="Output folder for videos")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (default=20)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask transparency (0-1)")
    return parser.parse_args()


def list_sorted_files(folder):
    files = [f for f in os.listdir(folder) if not f.startswith(".")]
    files = [f for f in files if os.path.isfile(os.path.join(folder, f))]
    return sorted(files)


def ensure_same_size(frame, mask):
    h, w = frame.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def overlay_binary_mask(frame_bgr, mask_gray, alpha):
    m = mask_gray > 0  # foreground

    overlay_color = np.zeros_like(frame_bgr)
    overlay_color[:, :, 2] = 255  # red mask

    # Blend full image once, then select blended pixels by mask. This avoids
    # passing empty arrays to OpenCV when a mask selects no pixels.
    blended = cv2.addWeighted(frame_bgr, 1.0 - float(alpha), overlay_color, float(alpha), 0)
    mask3 = m[:, :, None]
    out = np.where(mask3, blended, frame_bgr)
    return out


def make_video_for_video_folder(video_name, frames_root, masks_root, out_root, fps, alpha):
    frames_dir = os.path.join(frames_root, video_name)
    masks_dir = os.path.join(masks_root, video_name)

    if not os.path.isdir(masks_dir):
        print(f"[SKIP] Missing masks folder for '{video_name}'")
        return

    frame_files = list_sorted_files(frames_dir)
    mask_files = list_sorted_files(masks_dir)

    if not frame_files or not mask_files:
        print(f"[SKIP] Empty frames or masks for '{video_name}'")
        return

    mask_map = {os.path.splitext(f)[0]: f for f in mask_files}

    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    if first_frame is None:
        print(f"[SKIP] Cannot read first frame in '{video_name}'")
        return

    h, w = first_frame.shape[:2]
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"{video_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    written = 0
    missing_masks = 0

    for ff in frame_files:
        stem = os.path.splitext(ff)[0]
        mf = mask_map.get(stem)

        if mf is None:
            missing_masks += 1
            continue

        frame = cv2.imread(os.path.join(frames_dir, ff))
        mask = cv2.imread(os.path.join(masks_dir, mf), cv2.IMREAD_GRAYSCALE)

        if frame is None or mask is None:
            continue

        mask = ensure_same_size(frame, mask)
        out = overlay_binary_mask(frame, mask, alpha)

        writer.write(out)
        written += 1

    writer.release()

    if written == 0:
        if os.path.exists(out_path):
            os.remove(out_path)
        print(f"[SKIP] '{video_name}' wrote 0 frames")
    else:
        print(f"[OK] '{video_name}': {written} frames -> {out_path} (missing {missing_masks})")


def main():
    args = parse_args()

    if not os.path.isdir(args.frames):
        raise FileNotFoundError(f"Frames folder not found: {args.frames}")
    if not os.path.isdir(args.masks):
        raise FileNotFoundError(f"Masks folder not found: {args.masks}")

    video_folders = sorted(
        d for d in os.listdir(args.frames)
        if os.path.isdir(os.path.join(args.frames, d)) and not d.startswith(".")
    )

    if not video_folders:
        print("No subfolders found in frames directory.")
        return

    for video_name in video_folders:
        make_video_for_video_folder(
            video_name,
            args.frames,
            args.masks,
            args.output,
            args.fps,
            args.alpha,
        )


if __name__ == "__main__":
    main()
