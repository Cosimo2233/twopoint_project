from __future__ import annotations

import argparse
from argparse import ArgumentParser
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from twopoint_project.vision.capture import CameraCapture
from twopoint_project.vision.inferencer import (
    DEFAULT_IMG_SIZE,
    DEFAULT_ONNX_PATH,
    DEFAULT_VISION_BACKEND,
    PointPrediction,
    VisionInferencer,
    build_vision_inferencer as build_configured_vision_inferencer,
)
from twopoint_project.vision.send import UnixJsonLineSender


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def parse_args() -> argparse.Namespace:
    load_dotenv()

    parser = ArgumentParser(description="Run the two-point vision worker.")
    parser.add_argument("--onnx", default=env_str("TWOPOINT_ONNX_PATH", DEFAULT_ONNX_PATH))
    parser.add_argument("--vision-backend", default=env_str("TWOPOINT_VISION_BACKEND", DEFAULT_VISION_BACKEND))
    parser.add_argument("--camera", type=int, default=env_int("TWOPOINT_CAMERA_INDEX", 0))
    parser.add_argument("--camera-width", type=int, default=env_int("TWOPOINT_CAMERA_WIDTH", 640))
    parser.add_argument("--camera-height", type=int, default=env_int("TWOPOINT_CAMERA_HEIGHT", 480))
    parser.add_argument("--camera-fps", type=int, default=env_int("TWOPOINT_CAMERA_FPS", 30))
    parser.add_argument("--socket", default=env_str("TWOPOINT_SOCKET_PATH", "/tmp/twopoint_vision.sock"))
    parser.add_argument(
        "--reconnect-interval",
        type=float,
        default=env_float("TWOPOINT_RECONNECT_INTERVAL", 1.0),
    )
    parser.add_argument("--conf-threshold", type=float, default=env_float("TWOPOINT_CONF_THRESHOLD", 0.5))
    parser.add_argument("--fps-limit", type=float, default=env_float("TWOPOINT_FPS_LIMIT", 10.0))
    parser.add_argument("--img-size", type=int, default=env_int("TWOPOINT_IMG_SIZE", DEFAULT_IMG_SIZE))
    return parser.parse_args()


def build_vision_inferencer(args: argparse.Namespace) -> VisionInferencer:
    return build_configured_vision_inferencer(
        backend=args.vision_backend,
        onnx_path=args.onnx,
        img_size=args.img_size,
    )


def build_message(
    frame_id: int,
    timestamp: float,
    image_width: int,
    image_height: int,
    points: list[PointPrediction],
    conf_threshold: float,
) -> dict[str, Any]:
    valid_points = [point for point in points if point["confidence"] >= conf_threshold]
    valid = len(valid_points) == len(points) == 2
    reason = None if valid else "low_confidence"

    return {
        "type": "vision_points",
        "frame_id": frame_id,
        "timestamp": timestamp,
        "image_width": image_width,
        "image_height": image_height,
        "points": valid_points if valid else [],
        "valid": valid,
        "reason": reason,
    }


def sleep_for_fps_limit(loop_started_at: float, fps_limit: float) -> None:
    if fps_limit <= 0:
        return
    min_interval = 1.0 / fps_limit
    elapsed = time.monotonic() - loop_started_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)


def run_worker(args: argparse.Namespace) -> None:
    inferencer = build_vision_inferencer(args)
    with CameraCapture(
        camera_index=args.camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    ) as capture, UnixJsonLineSender(
        socket_path=args.socket,
        reconnect_interval=args.reconnect_interval,
    ) as sender:
        print(f"vision_worker: camera={args.camera}")
        print(f"vision_worker: socket={args.socket}")
        print(f"vision_worker: backend={args.vision_backend}")
        print(f"vision_worker: providers={inferencer.providers}")

        while True:
            loop_started_at = time.monotonic()
            captured = capture.read_frame()
            points = inferencer.predict(captured.frame_bgr)
            message = build_message(
                frame_id=captured.frame_id,
                timestamp=captured.timestamp,
                image_width=captured.frame_bgr.shape[1],
                image_height=captured.frame_bgr.shape[0],
                points=points,
                conf_threshold=args.conf_threshold,
            )
            sender.send(message)
            sleep_for_fps_limit(loop_started_at, args.fps_limit)


def main() -> None:
    run_worker(parse_args())


if __name__ == "__main__":
    main()
