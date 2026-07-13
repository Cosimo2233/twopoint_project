from __future__ import annotations

import argparse
import time

from twopoint_project.flash import open_laser_pointer


DEFAULT_INTERVAL_SECONDS = 1.0
DEFAULT_ON_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run laser pointer hardware tests.")
    subparsers = parser.add_subparsers(dest="command")

    blink = subparsers.add_parser("blink", help="Blink the laser pointer.")
    blink.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    blink.add_argument("--frequency", type=float, default=None)
    blink.add_argument("--on-seconds", type=float, default=DEFAULT_ON_SECONDS)
    blink.add_argument("--count", type=int, default=None)

    on = subparsers.add_parser("on", help="Keep the laser on, or turn it on for a fixed duration.")
    on.add_argument("--seconds", type=float, default=None)

    subparsers.add_parser("off", help="Turn the laser off once.")

    parser.set_defaults(
        command="blink",
        interval=DEFAULT_INTERVAL_SECONDS,
        frequency=None,
        on_seconds=DEFAULT_ON_SECONDS,
        count=None,
    )
    return parser.parse_args()


def blink_laser(
    *,
    interval: float = DEFAULT_INTERVAL_SECONDS,
    on_seconds: float = DEFAULT_ON_SECONDS,
    count: int | None = None,
) -> None:
    if interval <= 0:
        raise ValueError("interval must be greater than 0")
    if on_seconds <= 0:
        raise ValueError("on-seconds must be greater than 0")
    if on_seconds > interval:
        raise ValueError("on-seconds must not be greater than interval")
    if count is not None and count <= 0:
        raise ValueError("count must be greater than 0")

    off_seconds = interval - on_seconds
    cycles = 0
    with open_laser_pointer() as laser:
        try:
            while count is None or cycles < count:
                laser.on()
                time.sleep(on_seconds)
                laser.off()
                cycles += 1
                if count is None or cycles < count:
                    time.sleep(off_seconds)
        finally:
            laser.off()


def hold_laser_on(seconds: float | None = None) -> None:
    if seconds is not None and seconds < 0:
        raise ValueError("seconds must be non-negative")

    with open_laser_pointer() as laser:
        try:
            laser.on()
            if seconds is None:
                while True:
                    time.sleep(1)
            else:
                time.sleep(seconds)
        finally:
            laser.off()


def turn_laser_off() -> None:
    with open_laser_pointer() as laser:
        laser.off()


def main() -> None:
    args = parse_args()

    if args.command == "off":
        turn_laser_off()
        return

    if args.command == "on":
        hold_laser_on(args.seconds)
        return

    if args.frequency is not None and args.frequency <= 0:
        raise ValueError("frequency must be greater than 0")

    interval = 1.0 / args.frequency if args.frequency is not None else args.interval
    blink_laser(interval=interval, on_seconds=args.on_seconds, count=args.count)


if __name__ == "__main__":
    main()
