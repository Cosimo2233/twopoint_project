from __future__ import annotations

import time
from types import TracebackType
from typing import Type

from twopoint_project.f32c.motor import F32CMotor, SerialLike


DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200
DEFAULT_X_ID = 1
DEFAULT_Y_ID = 2
DEFAULT_SPEED_RPM = 100
DEFAULT_STARTUP_DELAY = 0.3
DEFAULT_COMMAND_INTERVAL = 0.001


class F32CGimbal:
    def __init__(
        self,
        serial_port: SerialLike,
        *,
        x_id: int = DEFAULT_X_ID,
        y_id: int = DEFAULT_Y_ID,
        speed_rpm: int = DEFAULT_SPEED_RPM,
        init_zero: bool = True,
        startup_delay: float = DEFAULT_STARTUP_DELAY,
        command_interval: float = DEFAULT_COMMAND_INTERVAL,
    ) -> None:
        self.serial_port = serial_port
        self.x = F32CMotor(serial_port, x_id, command_interval=command_interval)
        self.y = F32CMotor(serial_port, y_id, command_interval=command_interval)
        self.speed_rpm = speed_rpm
        self.init_zero = init_zero
        self.startup_delay = startup_delay

    def __enter__(self) -> F32CGimbal:
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def initialize(self) -> None:
        if self.startup_delay > 0:
            time.sleep(self.startup_delay)
        self.x.enable()
        self.y.enable()
        self.x.set_multi_turn_passthrough()
        self.y.set_multi_turn_passthrough()
        self.x.set_speed_rpm(self.speed_rpm)
        self.y.set_speed_rpm(self.speed_rpm)
        if self.init_zero:
            self.x.zero_multi_turn_angle()
            self.y.zero_multi_turn_angle()
        else:
            self.x.target_angle_deg = 0.0
            self.y.target_angle_deg = 0.0

    def move_by(self, x_delta_deg: float, y_delta_deg: float) -> None:
        self.x.move_by_angle(x_delta_deg)
        self.y.move_by_angle(y_delta_deg)

    def move_to(self, x_angle_deg: float, y_angle_deg: float) -> None:
        self.x.move_to_angle(x_angle_deg)
        self.y.move_to_angle(y_angle_deg)

    def disable(self) -> None:
        self.x.disable()
        self.y.disable()

    def close(self) -> None:
        close = getattr(self.serial_port, "close", None)
        if close is not None:
            close()


def open_serial_gimbal(
    *,
    port: str = DEFAULT_SERIAL_PORT,
    baudrate: int = DEFAULT_BAUDRATE,
    x_id: int = DEFAULT_X_ID,
    y_id: int = DEFAULT_Y_ID,
    speed_rpm: int = DEFAULT_SPEED_RPM,
    init_zero: bool = True,
    startup_delay: float = DEFAULT_STARTUP_DELAY,
    command_interval: float = DEFAULT_COMMAND_INTERVAL,
) -> F32CGimbal:
    import serial

    serial_port = serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.1,
        write_timeout=1.0,
    )
    return F32CGimbal(
        serial_port,
        x_id=x_id,
        y_id=y_id,
        speed_rpm=speed_rpm,
        init_zero=init_zero,
        startup_delay=startup_delay,
        command_interval=command_interval,
    )
