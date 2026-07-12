from __future__ import annotations

import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from twopoint_project.f32c import cli
from twopoint_project.f32c import protocol
from twopoint_project.f32c.gimbal import F32CGimbal
from twopoint_project.f32c.motor import F32CMotor


class FakeSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class F32CMotorTest(unittest.TestCase):
    def test_move_by_accumulates_target_angle(self) -> None:
        serial_port = FakeSerial()
        motor = F32CMotor(serial_port, 1, command_interval=0)

        motor.move_by_angle(10)
        motor.move_by_angle(5)

        self.assertEqual(motor.target_angle_deg, 15)
        self.assertEqual(serial_port.writes[0], protocol.build_multi_turn_angle(1, 10))
        self.assertEqual(serial_port.writes[1], protocol.build_multi_turn_angle(1, 15))

    def test_zero_multi_turn_angle_resets_local_target(self) -> None:
        serial_port = FakeSerial()
        motor = F32CMotor(serial_port, 1, command_interval=0)

        motor.move_by_angle(10)
        motor.zero_multi_turn_angle()

        self.assertEqual(motor.target_angle_deg, 0)
        self.assertEqual(serial_port.writes[-1], protocol.build_zero_multi_turn_angle(1))


class F32CGimbalTest(unittest.TestCase):
    def test_initialize_order(self) -> None:
        serial_port = FakeSerial()
        gimbal = F32CGimbal(
            serial_port,
            speed_rpm=100,
            init_zero=True,
            startup_delay=0,
            command_interval=0,
            enable_settle_delay=0,
        )

        gimbal.initialize()

        self.assertEqual(
            serial_port.writes,
            [
                protocol.build_enable(1),
                protocol.build_enable(2),
                protocol.build_set_multi_turn_passthrough(1),
                protocol.build_set_multi_turn_passthrough(2),
                protocol.build_set_speed(1, 100),
                protocol.build_set_speed(2, 100),
                protocol.build_zero_multi_turn_angle(1),
                protocol.build_zero_multi_turn_angle(2),
            ],
        )

    def test_move_by_sends_both_axes(self) -> None:
        serial_port = FakeSerial()
        gimbal = F32CGimbal(serial_port, startup_delay=0, command_interval=0)

        gimbal.move_by(1.0, -0.5)

        self.assertEqual(serial_port.writes[0], protocol.build_multi_turn_angle(1, 1.0))
        self.assertEqual(serial_port.writes[1], protocol.build_multi_turn_angle(2, -0.5))


class F32CCliTest(unittest.TestCase):
    def test_debug_frames_can_follow_subcommand(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["cli.py", "--port", "/dev/ttyAS4", "disable", "--debug-frames"],
        ):
            args = cli.parse_args()

        self.assertEqual(args.command, "disable")
        self.assertEqual(args.port, "/dev/ttyAS4")
        self.assertTrue(args.debug_frames)

    def test_help_displays(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "twopoint_project.f32c.cli", "--help"],
            cwd=repo_root,
            env={"PYTHONPATH": str(repo_root / "src")},
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("move-by", result.stdout)

    def test_move_by_opens_gimbal_and_sends_offsets(self) -> None:
        serial_port = FakeSerial()
        gimbal = F32CGimbal(serial_port, startup_delay=0, command_interval=0)
        args = Namespace(command="move-by", x=1.0, y=-0.5)

        with patch("twopoint_project.f32c.cli.build_gimbal", return_value=gimbal) as build_gimbal:
            cli.run(args)

        build_gimbal.assert_called_once_with(args)
        self.assertEqual(serial_port.writes[-2], protocol.build_multi_turn_angle(1, 1.0))
        self.assertEqual(serial_port.writes[-1], protocol.build_multi_turn_angle(2, -0.5))


if __name__ == "__main__":
    unittest.main()
