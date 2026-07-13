from __future__ import annotations

import errno
import unittest
from unittest.mock import patch

from twopoint_project.flash.flash import (
    A7A_LASER_GPIO_CHIP,
    A7A_LASER_GPIO_LINE,
    LaserPointer,
    open_laser_pointer,
)


class FakeGPIO:
    def __init__(self, initial_level: bool = True) -> None:
        self.values = [initial_level]
        self.closed = False

    def read(self) -> bool:
        return self.values[-1]

    def write(self, value: bool) -> None:
        self.values.append(value)

    def close(self) -> None:
        self.closed = True


class FakeGPIOFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.gpio: FakeGPIO | None = None

    def __call__(self, *args: object, **kwargs: object) -> FakeGPIO:
        self.calls.append((args, kwargs))
        direction = args[2]
        initial_level = direction == "high"
        self.gpio = FakeGPIO(initial_level)
        return self.gpio


class LaserPointerTest(unittest.TestCase):
    def test_active_low_turns_on_with_low_level(self) -> None:
        gpio = FakeGPIO(initial_level=True)
        laser = LaserPointer(gpio, active_low=True)

        laser.on()
        laser.off()

        self.assertEqual(gpio.values, [True, False, True])
        self.assertFalse(laser.enabled)

    def test_pulse_always_turns_off(self) -> None:
        gpio = FakeGPIO(initial_level=True)
        laser = LaserPointer(gpio, active_low=True)

        with patch("twopoint_project.flash.flash.time.sleep") as sleep:
            laser.pulse(0.25)

        sleep.assert_called_once_with(0.25)
        self.assertEqual(gpio.values[-2:], [False, True])
        self.assertFalse(laser.enabled)

    def test_close_turns_off_before_releasing_gpio(self) -> None:
        gpio = FakeGPIO(initial_level=True)
        laser = LaserPointer(gpio, active_low=True)

        laser.on()
        laser.close()

        self.assertEqual(gpio.values[-1], True)
        self.assertTrue(gpio.closed)

    def test_open_laser_pointer_uses_a7a_pin3_safely_off_by_default(self) -> None:
        factory = FakeGPIOFactory()

        laser = open_laser_pointer(gpio_factory=factory)

        self.assertEqual(
            factory.calls,
            [((A7A_LASER_GPIO_CHIP, A7A_LASER_GPIO_LINE, "high"), {"label": "twopoint-laser"})],
        )
        self.assertFalse(laser.enabled)

    def test_open_laser_pointer_can_start_on(self) -> None:
        factory = FakeGPIOFactory()

        laser = open_laser_pointer(initial_on=True, gpio_factory=factory)

        self.assertEqual(factory.calls[0][0], (A7A_LASER_GPIO_CHIP, A7A_LASER_GPIO_LINE, "low"))
        self.assertTrue(laser.enabled)

    def test_open_laser_pointer_explains_busy_gpio(self) -> None:
        def busy_factory(*args: object, **kwargs: object) -> FakeGPIO:
            raise OSError(errno.EBUSY, "Device or resource busy")

        with self.assertRaisesRegex(RuntimeError, "line 311 is busy"):
            open_laser_pointer(gpio_factory=busy_factory)


if __name__ == "__main__":
    unittest.main()
