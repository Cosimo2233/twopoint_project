#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


def add_common_options(parser: argparse.ArgumentParser) -> None:
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


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Manual F32C gimbal hardware debug script.")
    add_common_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("enable", help="Enable both motors only and leave them enabled.")
    subparsers.add_parser("disable", help="Disable both motors only.")
    init = subparsers.add_parser("init", help="Enable, set passthrough mode, set speed, then disable by default.")
    init.add_argument("--keep-enabled", action="store_true", help="Do not disable motors after initializing.")

    move_by = subparsers.add_parser("move-by", help="Initialize, move by fixed angle offsets, then disable by default.")
    move_by.add_argument("--x", type=float, default=0.0, help="X-axis relative angle in degrees.")
    move_by.add_argument("--y", type=float, default=0.0, help="Y-axis relative angle in degrees.")
    move_by.add_argument("--hold-seconds", type=float, default=1.0, help="Delay after moving before cleanup.")
    move_by.add_argument("--keep-enabled", action="store_true", help="Do not disable motors after moving.")

    move_to = subparsers.add_parser("move-to", help="Initialize, move to absolute angles, then disable by default.")
    move_to.add_argument("--x", type=float, default=0.0, help="X-axis absolute angle in degrees.")
    move_to.add_argument("--y", type=float, default=0.0, help="Y-axis absolute angle in degrees.")
    move_to.add_argument("--hold-seconds", type=float, default=1.0, help="Delay after moving before cleanup.")
    move_to.add_argument("--keep-enabled", action="store_true", help="Do not disable motors after moving.")

    pulse = subparsers.add_parser("pulse", help="Initialize, move by an offset, move back, then disable.")
    pulse.add_argument("--x", type=float, default=1.0, help="X-axis pulse angle in degrees.")
    pulse.add_argument("--y", type=float, default=0.0, help="Y-axis pulse angle in degrees.")
    pulse.add_argument("--hold-seconds", type=float, default=1.0, help="Delay at each position.")
    pulse.add_argument("--keep-enabled", action="store_true", help="Do not disable motors after the pulse.")

    return parser.parse_args()


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


def cleanup(gimbal: object, *, keep_enabled: bool) -> None:
    if keep_enabled:
        print("cleanup: leaving motors enabled")
        return
    print("cleanup: disabling motors")
    gimbal.disable()


def main() -> None:
    args = parse_args()
    with open_gimbal_from_args(args) as gimbal:
        if args.command == "enable":
            print("action: enable")
            gimbal.enable()
            return

        if args.command == "disable":
            print("action: disable")
            gimbal.disable()
            return

        if args.command == "init":
            print("action: init")
            gimbal.initialize()
            cleanup(gimbal, keep_enabled=args.keep_enabled)
            return

        keep_enabled = bool(getattr(args, "keep_enabled", False))
        try:
            print("action: initialize")
            gimbal.initialize()

            if args.command == "move-by":
                print(f"action: move-by x={args.x:g} deg, y={args.y:g} deg")
                gimbal.move_by(args.x, args.y)
                time.sleep(args.hold_seconds)
            elif args.command == "move-to":
                print(f"action: move-to x={args.x:g} deg, y={args.y:g} deg")
                gimbal.move_to(args.x, args.y)
                time.sleep(args.hold_seconds)
            elif args.command == "pulse":
                print(f"action: pulse x={args.x:g} deg, y={args.y:g} deg")
                gimbal.move_by(args.x, args.y)
                time.sleep(args.hold_seconds)
                gimbal.move_by(-args.x, -args.y)
                time.sleep(args.hold_seconds)
        finally:
            cleanup(gimbal, keep_enabled=keep_enabled)


if __name__ == "__main__":
    main()
