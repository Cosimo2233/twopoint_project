#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from twopoint_project.f32c.gimbal import DEFAULT_BAUDRATE


DEFAULT_USB_SERIAL_PORT = "/dev/serial/by-id/usb-Horco_Horco_CMSIS-DAP_507874186800-if02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print received serial data to the terminal.")
    parser.add_argument("--port", default=DEFAULT_USB_SERIAL_PORT, help="Serial port path.")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Serial baud rate.")
    parser.add_argument("--timeout", type=float, default=0.1, help="Read timeout in seconds.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Maximum bytes to read at once.")
    parser.add_argument(
        "--format",
        choices=("text", "hex", "both"),
        default="both",
        help="How to print received bytes.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Prefix each received chunk with local time.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text decoding used by --format text/both.",
    )
    return parser.parse_args()


def prefix(timestamp: bool) -> str:
    if not timestamp:
        return ""
    return f"[{time.strftime('%H:%M:%S')}] "


def format_chunk(chunk: bytes, output_format: str, encoding: str) -> str:
    text = chunk.decode(encoding, errors="replace")
    hex_text = chunk.hex(" ").upper()
    if output_format == "text":
        return text
    if output_format == "hex":
        return hex_text
    return f"{text.rstrip()} | HEX: {hex_text}"


def main() -> None:
    args = parse_args()

    try:
        import serial
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyserial is required. Run: poetry install") from exc

    print(f"Listening on {args.port} at {args.baudrate} baud. Press Ctrl+C to stop.")
    with serial.Serial(
        port=args.port,
        baudrate=args.baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=args.timeout,
    ) as serial_port:
        try:
            while True:
                chunk = serial_port.read(args.chunk_size)
                if not chunk:
                    continue
                print(f"{prefix(args.timestamp)}{format_chunk(chunk, args.format, args.encoding)}", flush=True)
        except KeyboardInterrupt:
            print("\nStopped")


if __name__ == "__main__":
    main()
