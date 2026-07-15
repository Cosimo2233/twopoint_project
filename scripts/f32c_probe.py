from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from twopoint_project.f32c.gimbal import DEFAULT_BAUDRATE, DEFAULT_SERIAL_PORT
from twopoint_project.f32c.protocol import FeedbackType, build_request_feedback, checksum


FEEDBACK_NAMES = {
    FeedbackType.SPEED: "speed_rpm",
    FeedbackType.MULTI_TURN_ANGLE: "multi_turn_angle_deg",
    FeedbackType.MECHANICAL_ANGLE: "mechanical_angle_deg",
    FeedbackType.ACCELERATION: "acceleration",
    FeedbackType.BUS_VOLTAGE: "bus_voltage_v",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe F32C motors without moving them.")
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--ids", type=int, nargs="+", default=[1, 2], help="Motor IDs to query.")
    parser.add_argument("--scan", type=int, default=0, help="Scan IDs from 1 to this value.")
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--gap", type=float, default=0.03)
    return parser.parse_args()


def read_response(serial_port, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    response = bytearray()
    while time.monotonic() < deadline:
        chunk = serial_port.read(64)
        if chunk:
            response.extend(chunk)
            if response.endswith(b"\x7B"):
                break
        else:
            time.sleep(0.005)
    return bytes(response)


def decode_response(response: bytes) -> str:
    if len(response) < 9:
        return "short/unknown response"
    if response[0] != 0x7A or response[-1] != 0x7B:
        return "bad frame markers"
    expected = checksum(response[:-2])
    actual = response[-2]
    if expected != actual:
        return f"bad checksum expected={expected:02X} actual={actual:02X}"

    feedback_type = response[2]
    value = int.from_bytes(response[3:7], "big", signed=True)
    try:
        name = FEEDBACK_NAMES[FeedbackType(feedback_type)]
    except ValueError:
        return f"type=0x{feedback_type:02X}, raw_value={value}"

    if feedback_type in {FeedbackType.MULTI_TURN_ANGLE, FeedbackType.MECHANICAL_ANGLE}:
        return f"{name}={value / 10:.1f}"
    if feedback_type == FeedbackType.BUS_VOLTAGE:
        return f"{name}={value / 100:.2f}"
    return f"{name}={value}"


def main() -> None:
    args = parse_args()
    ids = list(range(1, args.scan + 1)) if args.scan > 0 else args.ids

    import serial

    with serial.Serial(
        port=args.port,
        baudrate=args.baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.02,
        write_timeout=1.0,
    ) as serial_port:
        print(f"Probing {args.port} at {args.baudrate} baud; ids={ids}")
        for motor_id in ids:
            for feedback_type in (
                FeedbackType.BUS_VOLTAGE,
                FeedbackType.MECHANICAL_ANGLE,
                FeedbackType.MULTI_TURN_ANGLE,
            ):
                request = build_request_feedback(motor_id, feedback_type)
                serial_port.reset_input_buffer()
                serial_port.write(request)
                print(f"ID {motor_id} TX {feedback_type.name}: {request.hex(' ').upper()}")
                response = read_response(serial_port, args.timeout)
                if response:
                    print(f"ID {motor_id} RX: {response.hex(' ').upper()} ({decode_response(response)})")
                else:
                    print(f"ID {motor_id} RX: <none>")
                time.sleep(args.gap)


if __name__ == "__main__":
    main()
