from __future__ import annotations

import time
from typing import Protocol

from twopoint_project.f32c import protocol


class SerialLike(Protocol):
    def write(self, data: bytes) -> int | None:
        ...


class F32CMotor:
    def __init__(
        self,
        serial_port: SerialLike,
        motor_id: int,
        *,
        command_interval: float = 0.001,
        debug_frames: bool = False,
    ) -> None:
        self.serial_port = serial_port
        self.motor_id = motor_id
        self.command_interval = command_interval
        self.debug_frames = debug_frames
        self.target_angle_deg = 0.0

    def enable(self) -> None:
        self._send(protocol.build_enable(self.motor_id))

    def disable(self) -> None:
        self._send(protocol.build_disable(self.motor_id))

    def set_multi_turn_passthrough(self) -> None:
        self._send(protocol.build_set_multi_turn_passthrough(self.motor_id))

    def set_speed_rpm(self, rpm: int) -> None:
        self._send(protocol.build_set_speed(self.motor_id, abs(int(rpm))))

    def zero_multi_turn_angle(self) -> None:
        self._send(protocol.build_zero_multi_turn_angle(self.motor_id))
        self.target_angle_deg = 0.0

    def move_to_angle(self, angle_deg: float) -> None:
        self.target_angle_deg = float(angle_deg)
        self._send(protocol.build_multi_turn_angle(self.motor_id, self.target_angle_deg))

    def move_by_angle(self, delta_deg: float) -> None:
        self.move_to_angle(self.target_angle_deg + float(delta_deg))

    def _send(self, frame: bytes) -> None:
        if self.debug_frames:
            print(f"F32C[{self.motor_id}] -> {frame.hex(' ').upper()}")
        self.serial_port.write(frame)
        if self.command_interval > 0:
            time.sleep(self.command_interval)
