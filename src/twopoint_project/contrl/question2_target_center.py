from __future__ import annotations

import argparse
from argparse import ArgumentParser
import os
from pathlib import Path
import sys
import time

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv() -> bool:
        return False

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from twopoint_project.contrl.target_center_servo import (
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
from twopoint_project.vision.inferencer import (
    DEFAULT_IMG_SIZE,
    DEFAULT_ONNX_PATH,
    DEFAULT_VISION_BACKEND,
    build_vision_inferencer,
)


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = ArgumentParser(
        description=(
            "Question 2 target-center servo: keep target_center in view and move "
            "the camera center toward it. This mode does not search for a missing target."
        )
    )
    parser.add_argument("--onnx", default=env_str("TWOPOINT_ONNX_PATH", DEFAULT_ONNX_PATH))
    parser.add_argument("--vision-backend", default=env_str("TWOPOINT_VISION_BACKEND", DEFAULT_VISION_BACKEND))
    parser.add_argument("--img-size", type=int, default=env_int("TWOPOINT_IMG_SIZE", DEFAULT_IMG_SIZE))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)

    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--x-id", type=int, default=DEFAULT_X_ID)
    parser.add_argument("--y-id", type=int, default=DEFAULT_Y_ID)
    parser.add_argument("--speed-rpm", type=int, default=DEFAULT_SPEED_RPM)
    parser.add_argument("--startup-delay", type=float, default=DEFAULT_STARTUP_DELAY)
    parser.add_argument("--command-interval", type=float, default=DEFAULT_COMMAND_INTERVAL)
    parser.add_argument("--enable-settle-delay", type=float, default=DEFAULT_ENABLE_SETTLE_DELAY)
    parser.add_argument("--no-zero", action="store_true")
    parser.add_argument("--debug-frames", action="store_true")

    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--center-x", type=float, default=0.5)
    parser.add_argument("--center-y", type=float, default=0.5)
    parser.add_argument("--x-gain-deg", type=float, default=8.0)
    parser.add_argument("--y-gain-deg", type=float, default=-8.0)
    parser.add_argument("--max-step-deg", type=float, default=1.0)
    parser.add_argument("--deadband", type=float, default=0.006)
    parser.add_argument("--loop-hz", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--stable-frames", type=int, default=3)
    return parser.parse_args()


def run(args: argparse.Namespace) -> bool:
    from twopoint_project.vision.capture import CameraCapture

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

    deadline = time.monotonic() + args.timeout
    settled_frames = 0

    with CameraCapture(
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
        init_zero=not args.no_zero,
        startup_delay=args.startup_delay,
        command_interval=args.command_interval,
        enable_settle_delay=args.enable_settle_delay,
        debug_frames=args.debug_frames,
    ) as gimbal:
        print(f"question2: backend={args.vision_backend}")
        print(f"question2: providers={inferencer.providers}")
        print(f"question2: target-only mode, timeout={args.timeout:.2f}s")
        gimbal.initialize()

        while time.monotonic() < deadline:
            loop_started_at = time.monotonic()
            captured = capture.read_frame()
            points = inferencer.predict(captured.frame_bgr)
            update = servo.update(gimbal, points, conf_threshold=args.conf_threshold)

            if update.settled:
                settled_frames += 1
            else:
                settled_frames = 0

            if update.step is not None:
                print(
                    "frame={frame_id} err=({err_x:+.4f},{err_y:+.4f}) "
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
                print(f"frame={captured.frame_id} target_center invalid: {update.reason}")

            if settled_frames >= args.stable_frames:
                print(f"question2: settled for {settled_frames} frame(s)")
                return True

            sleep_for_loop_rate(loop_started_at, args.loop_hz)

    print("question2: timeout before stable centering")
    return False


def main() -> None:
    raise SystemExit(0 if run(parse_args()) else 1)


if __name__ == "__main__":
    main()
