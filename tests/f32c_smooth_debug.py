#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv() -> bool:
        return False

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


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Smoothly move both F32C gimbal motors with small continuous sine trajectories. "
            "The script initializes the gimbal and disables it on exit by default."
        )
    )
    parser.add_argument("--port", default=env_str("F32C_SERIAL_PORT", DEFAULT_SERIAL_PORT))
    parser.add_argument("--baudrate", type=int, default=env_int("F32C_BAUDRATE", DEFAULT_BAUDRATE))
    parser.add_argument("--x-id", type=int, default=env_int("F32C_X_ID", DEFAULT_X_ID))
    parser.add_argument("--y-id", type=int, default=env_int("F32C_Y_ID", DEFAULT_Y_ID))
    parser.add_argument("--speed-rpm", type=int, default=env_int("F32C_SPEED_RPM", DEFAULT_SPEED_RPM))
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
    parser.add_argument("--debug-frames", action="store_true", help="Print outgoing F32C frames as hex.")
    parser.add_argument("--max-angle", type=float, default=1.0, help="Default maximum absolute angle in degrees.")
    parser.add_argument("--x-max-angle", type=float, default=None, help="Override X-axis max angle in degrees.")
    parser.add_argument("--y-max-angle", type=float, default=None, help="Override Y-axis max angle in degrees.")
    parser.add_argument("--frequency", type=float, default=0.2, help="Sine frequency in Hz.")
    parser.add_argument("--rate-hz", type=float, default=30.0, help="Command update rate in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Motion duration in seconds.")
    parser.add_argument("--phase-deg", type=float, default=90.0, help="Y-axis phase offset relative to X.")
    parser.add_argument("--center-x", type=float, default=0.0, help="X-axis center angle in degrees.")
    parser.add_argument("--center-y", type=float, default=0.0, help="Y-axis center angle in degrees.")
    parser.add_argument("--return-center", action="store_true", help="Move to center before disabling.")
    parser.add_argument("--keep-enabled", action="store_true", help="Do not disable motors when the script exits.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.duration <= 0:
        raise ValueError("duration must be greater than 0")
    if args.frequency <= 0:
        raise ValueError("frequency must be greater than 0")
    if args.rate_hz <= 0:
        raise ValueError("rate-hz must be greater than 0")
    if args.max_angle < 0:
        raise ValueError("max-angle must be non-negative")
    if args.x_max_angle is not None and args.x_max_angle < 0:
        raise ValueError("x-max-angle must be non-negative")
    if args.y_max_angle is not None and args.y_max_angle < 0:
        raise ValueError("y-max-angle must be non-negative")


def open_gimbal_from_args(args: argparse.Namespace):
    return open_serial_gimbal(
        port=args.port,
        baudrate=args.baudrate,
        x_id=args.x_id,
        y_id=args.y_id,
        speed_rpm=args.speed_rpm,
        startup_delay=args.startup_delay,
        command_interval=args.command_interval,
        enable_settle_delay=args.enable_settle_delay,
        debug_frames=args.debug_frames,
    )


def sleep_until(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def run_motion(args: argparse.Namespace) -> None:
    x_amplitude = args.max_angle if args.x_max_angle is None else args.x_max_angle
    y_amplitude = args.max_angle if args.y_max_angle is None else args.y_max_angle
    phase_rad = math.radians(args.phase_deg)
    interval = 1.0 / args.rate_hz

    with open_gimbal_from_args(args) as gimbal:
        print("action: initialize")
        gimbal.initialize()
        started_at = time.monotonic()
        next_tick = started_at
        samples = 0

        try:
            print(
                "action: smooth sine motion "
                f"x_amp={x_amplitude:g} deg, y_amp={y_amplitude:g} deg, "
                f"frequency={args.frequency:g} Hz, rate={args.rate_hz:g} Hz, duration={args.duration:g}s"
            )
            while True:
                now = time.monotonic()
                elapsed = now - started_at
                if elapsed >= args.duration:
                    break

                angle = 2.0 * math.pi * args.frequency * elapsed
                x_angle = args.center_x + x_amplitude * math.sin(angle)
                y_angle = args.center_y + y_amplitude * math.sin(angle + phase_rad)
                gimbal.move_to(x_angle, y_angle)
                samples += 1

                next_tick += interval
                sleep_until(next_tick)
        except KeyboardInterrupt:
            print("\ninterrupted: stopping motion")
        finally:
            if args.return_center:
                print(f"cleanup: return center x={args.center_x:g} deg, y={args.center_y:g} deg")
                gimbal.move_to(args.center_x, args.center_y)
                time.sleep(min(0.5, max(interval, 0.0)))
            if args.keep_enabled:
                print("cleanup: leaving motors enabled")
            else:
                print("cleanup: disabling motors")
                gimbal.disable()
            print(f"sent {samples} smooth-motion sample(s)")


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_motion(args)


if __name__ == "__main__":
    main()
