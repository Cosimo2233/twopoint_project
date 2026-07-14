#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import cv2


DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30.0
DEFAULT_DURATION_SECONDS = 120.0
DEFAULT_OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a short video from an OpenCV camera device.")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Requested capture height.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Requested capture and output FPS.")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path. Defaults to outputs/camera_YYYYmmdd_HHMMSS.mp4.",
    )
    return parser.parse_args()


def default_output_path(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"camera_{stamp}.mp4"


def open_capture(camera_index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open camera index {camera_index}.")
    return capture


def create_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    return writer


def record_video(args: argparse.Namespace) -> Path:
    if args.duration <= 0:
        raise ValueError("duration must be greater than 0")
    if args.fps <= 0:
        raise ValueError("fps must be greater than 0")

    output_path = Path(args.output) if args.output else default_output_path()
    capture = open_capture(args.camera, args.width, args.height, args.fps)

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    start = time.monotonic()

    try:
        while time.monotonic() - start < args.duration:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame from camera index {args.camera}.")

            if writer is None:
                height, width = frame.shape[:2]
                writer = create_writer(output_path, args.fps, (width, height))
                print(
                    f"Recording camera={args.camera}, size={width}x{height}, "
                    f"fps={args.fps:g}, duration={args.duration:g}s"
                )
                print(f"Saving to: {output_path}")

            writer.write(frame)
            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed = max(time.monotonic() - start, 1e-9)
    print(f"Saved {frame_count} frame(s) in {elapsed:.2f}s, actual fps={frame_count / elapsed:.2f}")
    return output_path


def main() -> None:
    record_video(parse_args())


if __name__ == "__main__":
    main()
