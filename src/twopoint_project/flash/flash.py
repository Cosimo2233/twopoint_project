from __future__ import annotations

import errno
import time
from types import TracebackType
from typing import Callable, Protocol, Type


A7A_LASER_GPIO_CHIP = "/dev/gpiochip0"
# Cubie A7A physical pin 8 maps to PB9: 9 + 32 * B(1) = 41.
A7A_LASER_GPIO_LINE = 41
A7A_LASER_ACTIVE_LOW = False
DEFAULT_PULSE_SECONDS = 0.1


class GPIOLine(Protocol):
    def read(self) -> bool:
        ...

    def write(self, value: bool) -> None:
        ...

    def close(self) -> None:
        ...


GPIOFactory = Callable[..., GPIOLine]


class LaserPointer:
    def __init__(self, gpio: GPIOLine, *, active_low: bool = A7A_LASER_ACTIVE_LOW) -> None:
        self.gpio = gpio
        self.active_low = active_low
        self._closed = False
        self._enabled = self._logical_from_level(self.gpio.read())

    def __enter__(self) -> LaserPointer:
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def on(self) -> None:
        self.set_enabled(True)

    def off(self) -> None:
        self.set_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        self._ensure_open()
        self.gpio.write(self._level_for_logical(enabled))
        self._enabled = bool(enabled)

    def pulse(self, seconds: float = DEFAULT_PULSE_SECONDS) -> None:
        if seconds < 0:
            raise ValueError("pulse duration must be non-negative")

        self.on()
        try:
            time.sleep(seconds)
        finally:
            self.off()

    def read_enabled(self) -> bool:
        self._ensure_open()
        self._enabled = self._logical_from_level(self.gpio.read())
        return self._enabled

    def close(self, *, turn_off: bool = True) -> None:
        if self._closed:
            return

        try:
            if turn_off:
                self.off()
        finally:
            self.gpio.close()
            self._closed = True

    def _level_for_logical(self, enabled: bool) -> bool:
        return not enabled if self.active_low else enabled

    def _logical_from_level(self, level: bool) -> bool:
        return not level if self.active_low else level

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("laser pointer GPIO is already closed")


def open_laser_pointer(
    *,
    chip: str = A7A_LASER_GPIO_CHIP,
    line: int = A7A_LASER_GPIO_LINE,
    active_low: bool = A7A_LASER_ACTIVE_LOW,
    initial_on: bool = False,
    label: str = "twopoint-laser",
    gpio_factory: GPIOFactory | None = None,
) -> LaserPointer:
    if gpio_factory is None:
        try:
            from periphery import GPIO, GPIOError
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "python-periphery is required for GPIO laser control. "
                "Install project dependencies with: poetry install"
            ) from exc

        gpio_factory = GPIO
    else:
        GPIOError = OSError

    initial_level = not initial_on if active_low else initial_on
    initial_direction = "high" if initial_level else "low"
    try:
        gpio = gpio_factory(chip, line, initial_direction, label=label)
    except GPIOError as exc:
        if getattr(exc, "errno", None) == errno.EBUSY:
            raise RuntimeError(
                f"GPIO {chip} line {line} is busy. Another process may still be holding "
                "the laser pin; check with: ps -ef | grep tests/test_flash.py"
            ) from exc
        raise

    return LaserPointer(gpio, active_low=active_low)
