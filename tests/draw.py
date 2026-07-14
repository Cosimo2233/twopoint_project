#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from twopoint_project.contrl.center_then_flash import (
    center_target,
    env_bool,
    env_float,
    env_int,
    env_str,
    load_dotenv,
    open_camera_capture,
)
from twopoint_project.contrl.target_center_servo import AimUpdate, PointPrediction, TargetCenterServo
from twopoint_project.f32c.gimbal import (
    DEFAULT_BAUDRATE,
    DEFAULT_COMMAND_INTERVAL,
    DEFAULT_ENABLE_SETTLE_DELAY,
    DEFAULT_SERIAL_PORT,
    DEFAULT_SPEED_RPM,
    DEFAULT_STARTUP_DELAY,
    DEFAULT_X_ID,
    DEFAULT_Y_ID,
    open_serial_gimbal,
)
from twopoint_project.vision.inferencer import (
    DEFAULT_IMG_SIZE,
    DEFAULT_ONNX_PATH,
    DEFAULT_VISION_BACKEND,
    build_vision_inferencer,
)
from twopoint_project.vision.pipeline import VisionProducer


DEFAULT_OUTPUT_DIR = Path("outputs")


class NoopGimbal:
    def initialize(self) -> None:
        pass

    def move_by(self, x_delta_deg: float, y_delta_deg: float) -> None:
        pass

    def disable(self) -> None:
        pass


class AimVideoRecorder:
    def __init__(
        self,
        output_path: Path,
        fps: float,
        conf_threshold: float,
    ) -> None:
        self.output_path = output_path
        self.fps = fps
        self.conf_threshold = conf_threshold
        self.writer: cv2.VideoWriter | None = None
        self.frame_count = 0

    def write(self, captured: Any, points: list[PointPrediction], update: AimUpdate) -> None:
        annotated = draw_aim_frame(captured.frame_bgr, points, update, self.conf_threshold)
        if self.writer is None:
            height, width = annotated.shape[:2]
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.output_path}")
            print(f"Saving annotated aiming video to: {self.output_path}")

        self.writer.write(annotated)
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Record annotated target-centering video.")
    parser.add_argument("--output", default=None, help="Output video path. Defaults to outputs/aim_YYYYmmdd_HHMMSS.mp4.")
    parser.add_argument("--camera", type=int, default=env_int("TWOPOINT_CAMERA_INDEX", 0))
    parser.add_argument("--camera-width", type=int, default=env_int("TWOPOINT_CAMERA_WIDTH", 640))
    parser.add_argument("--camera-height", type=int, default=env_int("TWOPOINT_CAMERA_HEIGHT", 480))
    parser.add_argument("--camera-fps", type=int, default=env_int("TWOPOINT_CAMERA_FPS", 30))
    parser.add_argument("--vision-backend", default=env_str("TWOPOINT_VISION_BACKEND", DEFAULT_VISION_BACKEND))
    parser.add_argument("--onnx", default=env_str("TWOPOINT_ONNX_PATH", DEFAULT_ONNX_PATH))
    parser.add_argument("--img-size", type=int, default=env_int("TWOPOINT_IMG_SIZE", DEFAULT_IMG_SIZE))

    parser.add_argument("--port", default=env_str("F32C_SERIAL_PORT", DEFAULT_SERIAL_PORT))
    parser.add_argument("--baudrate", type=int, default=env_int("F32C_BAUDRATE", DEFAULT_BAUDRATE))
    parser.add_argument("--x-id", type=int, default=env_int("F32C_X_ID", DEFAULT_X_ID))
    parser.add_argument("--y-id", type=int, default=env_int("F32C_Y_ID", DEFAULT_Y_ID))
    parser.add_argument("--speed-rpm", type=int, default=env_int("F32C_SPEED_RPM", DEFAULT_SPEED_RPM))
    parser.add_argument("--init-zero", action=argparse.BooleanOptionalAction, default=env_bool("F32C_INIT_ZERO", True))
    parser.add_argument("--startup-delay", type=float, default=env_float("F32C_STARTUP_DELAY", DEFAULT_STARTUP_DELAY))
    parser.add_argument(
        "--command-interval",
        type=float,
        default=env_float("F32C_COMMAND_INTERVAL", DEFAULT_COMMAND_INTERVAL),
    )
    parser.add_argument(
        "--enable-settle-delay",
        type=float,
        default=env_float("F32C_ENABLE_SETTLE_DELAY", DEFAULT_ENABLE_SETTLE_DELAY),
    )
    parser.add_argument("--debug-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not open or move the gimbal; only record detection.")

    parser.add_argument("--conf-threshold", type=float, default=env_float("CENTER_CONF_THRESHOLD", 0.5))
    parser.add_argument("--center-x", type=float, default=env_float("CENTER_TARGET_X", 0.5))
    parser.add_argument("--center-y", type=float, default=env_float("CENTER_TARGET_Y", 0.5))
    parser.add_argument("--x-gain-deg", type=float, default=env_float("CENTER_X_GAIN_DEG", 8.0))
    parser.add_argument("--y-gain-deg", type=float, default=env_float("CENTER_Y_GAIN_DEG", -8.0))
    parser.add_argument("--max-step-deg", type=float, default=env_float("CENTER_MAX_STEP_DEG", 1.0))
    parser.add_argument("--deadband", type=float, default=env_float("CENTER_DEADBAND", 0.006))
    parser.add_argument("--loop-hz", type=float, default=env_float("CENTER_LOOP_HZ", 15.0))
    parser.add_argument("--timeout", type=float, default=env_float("CENTER_TIMEOUT", 10.0))
    parser.add_argument("--stable-frames", type=int, default=env_int("CENTER_STABLE_FRAMES", 3))
    parser.add_argument("--output-fps", type=float, default=env_float("DRAW_OUTPUT_FPS", 15.0))
    return parser.parse_args()


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"aim_{stamp}.mp4"


def normalized_to_pixel(point: PointPrediction, width: int, height: int) -> tuple[int, int]:
    x = int(round(point["x"] * max(width - 1, 1)))
    y = int(round(point["y"] * max(height - 1, 1)))
    return x, y


def draw_text(image, text: str, origin: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_aim_frame(
    frame_bgr,
    points: list[PointPrediction],
    update: AimUpdate,
    conf_threshold: float,
):
    image = frame_bgr.copy()
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    cv2.line(image, (center[0] - 18, center[1]), (center[0] + 18, center[1]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (center[0], center[1] - 18), (center[0], center[1] + 18), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(image, center, 22, (255, 255, 255), 1, cv2.LINE_AA)
    draw_text(image, "image center", (center[0] + 8, center[1] - 10))

    colors = {
        "target_center": (0, 220, 0),
        "laser_point": (0, 0, 255),
    }
    for point in points:
        label = str(point["label"])
        color = colors.get(label, (0, 200, 255))
        x, y = normalized_to_pixel(point, width, height)
        valid = point["confidence"] >= conf_threshold
        radius = 7 if valid else 4
        thickness = -1 if valid else 1
        cv2.circle(image, (x, y), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(image, (x, y), radius + 3, (255, 255, 255), 1, cv2.LINE_AA)
        draw_text(image, f"{label} conf={point['confidence']:.2f}", (x + 10, max(y - 8, 18)), color)
        if label == "target_center" and valid:
            cv2.arrowedLine(image, center, (x, y), color, 2, cv2.LINE_AA, tipLength=0.12)

    if update.step is not None:
        draw_text(
            image,
            "err=({:+.4f},{:+.4f}) step=({:+.3f},{:+.3f}) settled={}".format(
                update.step.error.x,
                update.step.error.y,
                update.step.x_delta_deg,
                update.step.y_delta_deg,
                update.settled,
            ),
            (14, 26),
            (0, 255, 255),
        )
    else:
        draw_text(image, f"invalid: {update.reason}", (14, 26), (0, 0, 255))
    return image


def run(args: argparse.Namespace) -> bool:
    if args.timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    if args.stable_frames <= 0:
        raise ValueError("stable-frames must be greater than 0")
    if args.output_fps <= 0:
        raise ValueError("output-fps must be greater than 0")

    output_path = Path(args.output) if args.output else default_output_path()
    inferencer = build_vision_inferencer(
        backend=args.vision_backend,
        onnx_path=args.onnx,
        img_size=args.img_size,
    )
    servo = TargetCenterServo(
        center_x=args.center_x,
        center_y=args.center_y,
        x_gain_deg=args.x_gain_deg,
        y_gain_deg=args.y_gain_deg,
        max_step_deg=args.max_step_deg,
        deadband=args.deadband,
    )
    recorder = AimVideoRecorder(output_path, fps=args.output_fps, conf_threshold=args.conf_threshold)
    gimbal_context = (
        nullcontext(NoopGimbal())
        if args.dry_run
        else open_serial_gimbal(
            port=args.port,
            baudrate=args.baudrate,
            x_id=args.x_id,
            y_id=args.y_id,
            speed_rpm=args.speed_rpm,
            init_zero=args.init_zero,
            startup_delay=args.startup_delay,
            command_interval=args.command_interval,
            enable_settle_delay=args.enable_settle_delay,
            debug_frames=args.debug_frames,
        )
    )

    with open_camera_capture(
        camera_index=args.camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    ) as capture, gimbal_context as gimbal, VisionProducer(
        capture=capture,
        inferencer=inferencer,
    ) as vision:
        print(f"draw: backend={args.vision_backend}")
        print(f"draw: providers={inferencer.providers}")
        print(f"draw: dry_run={args.dry_run}")
        if not args.dry_run:
            gimbal.initialize()
        try:
            return center_target(
                vision=vision,
                gimbal=gimbal,
                servo=servo,
                conf_threshold=args.conf_threshold,
                loop_hz=args.loop_hz,
                timeout=args.timeout,
                stable_frames=args.stable_frames,
                on_frame=recorder.write,
            )
        finally:
            recorder.close()
            if not args.dry_run:
                gimbal.disable()
            print(f"draw: saved {recorder.frame_count} frame(s) to {output_path}")


def main() -> None:
    raise SystemExit(0 if run(parse_args()) else 1)


if __name__ == "__main__":
    main()
