from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from twopoint_project.f32c.gimbal import (
    DEFAULT_BAUDRATE,
    DEFAULT_COMMAND_INTERVAL,
    DEFAULT_SERIAL_PORT,
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
            "Disable F32C gimbal motors without initializing or moving them. "
            f"Default port targets A7A {A7A_UART_PORT_NAME}: "
            f"TX pin {A7A_UART_TX_PIN}, RX pin {A7A_UART_RX_PIN}."
        )
    )
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--x-id", type=int, default=DEFAULT_X_ID)
    parser.add_argument("--y-id", type=int, default=DEFAULT_Y_ID)
    parser.add_argument("--command-interval", type=float, default=DEFAULT_COMMAND_INTERVAL)
    parser.add_argument("--debug-frames", action="store_true", help="Print outgoing F32C frames as hex.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Opening serial port {args.port} at {args.baudrate} baud")
    print(f"Disabling motors: x={args.x_id}, y={args.y_id}")

    with open_serial_gimbal(
        port=args.port,
        baudrate=args.baudrate,
        x_id=args.x_id,
        y_id=args.y_id,
        command_interval=args.command_interval,
        debug_frames=args.debug_frames,
    ) as gimbal:
        gimbal.disable()

    print("Done")


if __name__ == "__main__":
    main()
