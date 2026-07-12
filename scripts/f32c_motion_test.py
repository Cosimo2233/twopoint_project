from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from twopoint_project.f32c.gimbal import (
    DEFAULT_BAUDRATE,
    DEFAULT_COMMAND_INTERVAL,
    DEFAULT_ENABLE_SETTLE_DELAY,
    DEFAULT_SERIAL_PORT,
    DEFAULT_STARTUP_DELAY,
    DEFAULT_X_ID,
    DEFAULT_Y_ID,
    A7A_UART_PORT_NAME,
    A7A_UART_RX_PIN,
    A7A_UART_TX_PIN,
    open_serial_gimbal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small visible F32C gimbal motion test. "
            f"Default port targets A7A {A7A_UART_PORT_NAME}: "
            f"TX pin {A7A_UART_TX_PIN}, RX pin {A7A_UART_RX_PIN}."
        )
    )
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--x-id", type=int, default=DEFAULT_X_ID)
    parser.add_argument("--y-id", type=int, default=DEFAULT_Y_ID)
    parser.add_argument("--speed-rpm", type=int, default=50)
    parser.add_argument("--angle", type=float, default=10.0, help="Test offset angle in degrees.")
    parser.add_argument("--hold", type=float, default=1.0, help="Seconds to hold each test position.")
    parser.add_argument("--startup-delay", type=float, default=DEFAULT_STARTUP_DELAY)
    parser.add_argument("--command-interval", type=float, default=DEFAULT_COMMAND_INTERVAL)
    parser.add_argument("--enable-settle-delay", type=float, default=DEFAULT_ENABLE_SETTLE_DELAY)
    parser.add_argument(
        "--axis",
        choices=("both", "x", "y"),
        default="both",
        help="Which axis to move during the test.",
    )
    parser.add_argument(
        "--no-zero",
        action="store_true",
        help="Do not send the multi-turn zero command during initialization.",
    )
    parser.add_argument(
        "--disable-at-end",
        action="store_true",
        help="Disable both motors after returning to zero.",
    )
    parser.add_argument(
        "--enable-only",
        action="store_true",
        help="Only send enable frames, then wait for --hold seconds.",
    )
    parser.add_argument("--debug-frames", action="store_true", help="Print outgoing F32C frames as hex.")
    return parser.parse_args()


def axis_offsets(axis: str, angle: float) -> tuple[float, float]:
    if axis == "x":
        return angle, 0.0
    if axis == "y":
        return 0.0, angle
    return angle, angle


def main() -> None:
    args = parse_args()
    x_angle, y_angle = axis_offsets(args.axis, args.angle)

    print(f"Opening serial port {args.port} at {args.baudrate} baud")
    print(f"Motor IDs: x={args.x_id}, y={args.y_id}; speed={args.speed_rpm} RPM")
    print(f"Motion: axis={args.axis}, offset={args.angle} deg, hold={args.hold}s")

    with open_serial_gimbal(
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
        if args.enable_only:
            print("Enable only")
            gimbal.enable()
            time.sleep(args.hold)
            return

        print("Initializing: enable, multi-turn passthrough, speed, zero")
        gimbal.initialize()
        time.sleep(0.2)

        print(f"Move by: x={x_angle} deg, y={y_angle} deg")
        gimbal.move_by(x_angle, y_angle)
        time.sleep(args.hold)

        print("Return to zero")
        gimbal.move_to(0.0, 0.0)
        time.sleep(args.hold)

        if args.disable_at_end:
            print("Disable motors")
            gimbal.disable()

    print("Done")


if __name__ == "__main__":
    main()
