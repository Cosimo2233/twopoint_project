from __future__ import annotations

from typing import Any

from twopoint_project.f32c.gimbal import open_serial_gimbal


def build_gimbal(args: Any):
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


def run(args: Any) -> None:
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
