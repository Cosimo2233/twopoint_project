from __future__ import annotations

import argparse
import os

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
    A7A_UART_PORT_NAME,
    A7A_UART_RX_PIN,
    A7A_UART_TX_PIN,
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
            "Control an F32C two-axis gimbal. "
            f"Default port targets A7A {A7A_UART_PORT_NAME}: "
            f"TX pin {A7A_UART_TX_PIN}, RX pin {A7A_UART_RX_PIN}."
        )
    )
    parser.add_argument("--port", default=env_str("F32C_SERIAL_PORT", DEFAULT_SERIAL_PORT))
    parser.add_argument("--baudrate", type=int, default=env_int("F32C_BAUDRATE", DEFAULT_BAUDRATE))
    parser.add_argument("--x-id", type=int, default=env_int("F32C_X_ID", DEFAULT_X_ID))
    parser.add_argument("--y-id", type=int, default=env_int("F32C_Y_ID", DEFAULT_Y_ID))
    parser.add_argument("--speed-rpm", type=int, default=env_int("F32C_SPEED_RPM", DEFAULT_SPEED_RPM))
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=env_float("F32C_STARTUP_DELAY", DEFAULT_STARTUP_DELAY),
    )
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

    subparsers = parser.add_subparsers(dest="command", required=True)
    enable = subparsers.add_parser("enable", help="Enable both gimbal motors only.")
    add_command_debug_option(enable)
    init = subparsers.add_parser("init", help="Initialize the gimbal without zeroing.")
    add_command_debug_option(init)
    disable = subparsers.add_parser("disable", help="Disable both gimbal motors.")
    add_command_debug_option(disable)

    move_by = subparsers.add_parser("move-by", help="Move by angular offsets in degrees.")
    add_command_debug_option(move_by)
    move_by.add_argument("--x", type=float, required=True)
    move_by.add_argument("--y", type=float, required=True)

    move_to = subparsers.add_parser("move-to", help="Move to absolute multi-turn angles in degrees.")
    add_command_debug_option(move_to)
    move_to.add_argument("--x", type=float, required=True)
    move_to.add_argument("--y", type=float, required=True)

    return parser.parse_args()


def add_command_debug_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--debug-frames",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print outgoing F32C frames as hex.",
    )


def build_gimbal(args: argparse.Namespace):
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


def run(args: argparse.Namespace) -> None:
    with build_gimbal(args) as gimbal:
        if args.command == "enable":
            gimbal.enable()
            return

        if args.command == "disable":
            gimbal.disable()
            return

        gimbal.initialize()
        if args.command == "move-by":
            gimbal.move_by(args.x, args.y)
        elif args.command == "move-to":
            gimbal.move_to(args.x, args.y)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
