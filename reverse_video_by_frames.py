#!/usr/bin/env python3
"""
reverse_video_by_frames.py

Simple script:
- Reads a video file frame by frame
- Reverses frame order
- Writes final reversed video

Usage:
    python reverse_video_by_frames.py --input reconstructed_optimal.mp4 --out final_video.mp4
"""

import argparse
import cv2
from tqdm import tqdm

def reverse_video(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    print("Reading frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Loaded {len(frames)} frames. Reversing...")

    frames = frames[::-1]

    print(f"Writing reversed video to: {out_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for f in tqdm(frames, desc="Writing"):
        out.write(f)

    out.release()
    print("Done! Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--out", default="final_video.mp4", help="Output reversed video path")
    args = parser.parse_args()

    reverse_video(args.input, args.out)
