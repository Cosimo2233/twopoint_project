from __future__ import annotations

import argparse
from argparse import ArgumentParser
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv() -> bool:
        return False

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from twopoint_project.contrl.target_center_servo import (
    AimUpdate,
    PointPrediction,
    TargetCenterServo,
    sleep_for_loop_rate,
)
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
from twopoint_project.flash import open_laser_pointer
from twopoint_project.vision.inferencer import (
    DEFAULT_IMG_SIZE,
    DEFAULT_ONNX_PATH,
    DEFAULT_VISION_BACKEND,
    build_vision_inferencer,
)
from twopoint_project.vision.pipeline import VisionProducer


CenterFrameCallback = Callable[[Any, list[PointPrediction], AimUpdate], None]


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def open_camera_capture(
    *,
    camera_index: int,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> object:
    from twopoint_project.vision.capture import CameraCapture

    return CameraCapture(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
    )


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = ArgumentParser(
        description=(
            "Center the target paper in the camera view, then turn the laser on "
            "for a fixed duration before safely shutting down."
        )
    )
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

    parser.add_argument("--conf-threshold", type=float, default=env_float("CENTER_CONF_THRESHOLD", 0.5))
    parser.add_argument("--center-x", type=float, default=env_float("CENTER_TARGET_X", 0.5))
    parser.add_argument("--center-y", type=float, default=env_float("CENTER_TARGET_Y", 0.5))
    parser.add_argument("--x-gain-deg", type=float, default=env_float("CENTER_X_GAIN_DEG", 8.0))
    parser.add_argument("--y-gain-deg", type=float, default=env_float("CENTER_Y_GAIN_DEG", -8.0))
    parser.add_argument("--max-step-deg", type=float, default=env_float("CENTER_MAX_STEP_DEG", 1.0))
    parser.add_argument("--deadband", type=float, default=env_float("CENTER_DEADBAND", 0.006))
    parser.add_argument("--loop-hz", type=float, default=env_float("CENTER_LOOP_HZ", 15.0))
    parser.add_argument("--center-timeout", type=float, default=env_float("CENTER_TIMEOUT", 10.0))
    parser.add_argument("--stable-frames", type=int, default=env_int("CENTER_STABLE_FRAMES", 3))
    parser.add_argument("--laser-hold-seconds", type=float, default=env_float("LASER_HOLD_SECONDS", 5.0))
    return parser.parse_args()


def center_target(
    *,
    vision: VisionProducer,
    gimbal: object,
    servo: TargetCenterServo,
    conf_threshold: float,
    loop_hz: float,
    timeout: float,
    stable_frames: int,
    on_frame: CenterFrameCallback | None = None,
) -> bool:
    deadline = time.monotonic() + timeout
    settled_frames = 0

    while time.monotonic() < deadline:
        loop_started_at = time.monotonic()
        read_timeout = min(max(deadline - time.monotonic(), 0.0), 1.0)
        try:
            vision_frame = vision.read_latest(timeout=read_timeout)
        except TimeoutError:
            continue

        captured = vision_frame.captured
        points = vision_frame.points
        update = servo.update(gimbal, points, conf_threshold=conf_threshold)
        if on_frame is not None:
            on_frame(captured, points, update)

        if update.settled:
            settled_frames += 1
        else:
            settled_frames = 0

        if update.step is not None:
            print(
                "center: frame={frame_id} err=({err_x:+.4f},{err_y:+.4f}) "
                "step=({step_x:+.3f},{step_y:+.3f}) settled={settled}".format(
                    frame_id=captured.frame_id,
                    err_x=update.step.error.x,
                    err_y=update.step.error.y,
                    step_x=update.step.x_delta_deg,
                    step_y=update.step.y_delta_deg,
                    settled=update.settled,
                )
            )
        else:
            print(f"center: frame={captured.frame_id} target_center invalid: {update.reason}")

        if settled_frames >= stable_frames:
            print(f"center: settled for {settled_frames} frame(s)")
            return True

        sleep_for_loop_rate(loop_started_at, loop_hz)

    print(f"center: timeout after {timeout:.2f}s")
    return False


def run(args: argparse.Namespace) -> bool:
    if args.center_timeout < 0:
        raise ValueError("center-timeout must be non-negative")
    if args.laser_hold_seconds < 0:
        raise ValueError("laser-hold-seconds must be non-negative")
    if args.stable_frames <= 0:
        raise ValueError("stable-frames must be greater than 0")

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

    with open_camera_capture(
        camera_index=args.camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    ) as capture, open_serial_gimbal(
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
    ) as gimbal, VisionProducer(
        capture=capture,
        inferencer=inferencer,
    ) as vision, open_laser_pointer(initial_on=False) as laser:
        print("task: center_then_flash")
        print(f"task: backend={args.vision_backend}")
        print(f"task: providers={inferencer.providers}")
        print(f"task: center_timeout={args.center_timeout:.2f}s")
        print(f"task: laser_hold_seconds={args.laser_hold_seconds:.2f}s")

        laser.off()
        gimbal.initialize()

        try:
            centered = center_target(
                vision=vision,
                gimbal=gimbal,
                servo=servo,
                conf_threshold=args.conf_threshold,
                loop_hz=args.loop_hz,
                timeout=args.center_timeout,
                stable_frames=args.stable_frames,
            )

            print("laser: on")
            laser.on()
            time.sleep(args.laser_hold_seconds)
            print("laser: off")
            return centered
        finally:
            laser.off()
            gimbal.disable()


def main() -> None:
    raise SystemExit(0 if run(parse_args()) else 1)


if __name__ == "__main__":
    main()
