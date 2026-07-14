from __future__ import annotations

import unittest

from twopoint_project.contrl.target_center_servo import (
    DEFAULT_CONTROL_CONF_THRESHOLD,
    PIDAxisGains,
    TargetCenterObservation,
    TargetCenterServo,
    control_conf_threshold,
    select_target_center,
    validate_conf_threshold,
)


class FakeGimbal:
    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []

    def move_by(self, x_delta_deg: float, y_delta_deg: float) -> None:
        self.moves.append((x_delta_deg, y_delta_deg))


class TargetCenterServoTest(unittest.TestCase):
    def test_select_target_center_ignores_laser_point(self) -> None:
        point = select_target_center(
            [
                {"label": "laser_point", "x": 0.1, "y": 0.2, "confidence": 0.99},
                {"label": "target_center", "x": 0.6, "y": 0.4, "confidence": 0.8},
            ],
            conf_threshold=0.5,
        )

        self.assertIsNotNone(point)
        assert point is not None
        self.assertEqual(point.x, 0.6)
        self.assertEqual(point.y, 0.4)
        self.assertEqual(point.confidence, 0.8)

    def test_select_target_center_returns_none_below_threshold(self) -> None:
        point = select_target_center(
            [{"label": "target_center", "x": 0.6, "y": 0.4, "confidence": 0.4}],
            conf_threshold=0.5,
        )

        self.assertIsNone(point)

    def test_control_conf_threshold_returns_default(self) -> None:
        self.assertEqual(control_conf_threshold(), DEFAULT_CONTROL_CONF_THRESHOLD)

    def test_validate_conf_threshold_accepts_zero_to_one(self) -> None:
        self.assertEqual(validate_conf_threshold(0.0), 0.0)
        self.assertEqual(validate_conf_threshold(1.0), 1.0)

    def test_validate_conf_threshold_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            validate_conf_threshold(-0.1)
        with self.assertRaises(ValueError):
            validate_conf_threshold(1.1)

    def test_compute_step_uses_center_error_and_clamps(self) -> None:
        servo = TargetCenterServo(
            x_gain_deg=10.0,
            y_gain_deg=-10.0,
            max_step_deg=1.0,
            deadband=0.0,
        )

        step = servo.compute_step(TargetCenterObservation(x=0.7, y=0.4, confidence=1.0))

        self.assertAlmostEqual(step.error.x, 0.2)
        self.assertAlmostEqual(step.error.y, -0.1)
        self.assertAlmostEqual(step.x_delta_deg, 1.0)
        self.assertAlmostEqual(step.y_delta_deg, 1.0)
        self.assertFalse(step.settled)

    def test_update_does_not_move_inside_deadband(self) -> None:
        servo = TargetCenterServo(deadband=0.01, conf_threshold=0.5)
        gimbal = FakeGimbal()

        update = servo.update(
            gimbal,
            [{"label": "target_center", "x": 0.505, "y": 0.495, "confidence": 0.9}],
        )

        self.assertTrue(update.valid)
        self.assertTrue(update.settled)
        self.assertFalse(update.moved)
        self.assertEqual(gimbal.moves, [])

    def test_update_moves_when_target_is_valid_and_outside_deadband(self) -> None:
        servo = TargetCenterServo(
            x_gain_deg=10.0,
            y_gain_deg=-10.0,
            max_step_deg=2.0,
            deadband=0.01,
            conf_threshold=0.5,
        )
        gimbal = FakeGimbal()

        update = servo.update(
            gimbal,
            [{"label": "target_center", "x": 0.6, "y": 0.45, "confidence": 0.9}],
        )

        self.assertTrue(update.valid)
        self.assertFalse(update.settled)
        self.assertTrue(update.moved)
        self.assertEqual(len(gimbal.moves), 1)
        self.assertAlmostEqual(gimbal.moves[0][0], 1.0)
        self.assertAlmostEqual(gimbal.moves[0][1], 0.5)

    def test_pid_integral_contributes_to_repeated_steps(self) -> None:
        servo = TargetCenterServo(
            x_pid=PIDAxisGains(kp=0.0, ki=10.0, kd=0.0, integral_limit=1.0, output_limit_deg=10.0),
            y_pid=PIDAxisGains(kp=0.0, ki=0.0, kd=0.0, output_limit_deg=10.0),
            deadband=0.0,
            conf_threshold=0.5,
        )

        first = servo.compute_step(TargetCenterObservation(x=0.6, y=0.5, confidence=1.0), now=1.0)
        second = servo.compute_step(TargetCenterObservation(x=0.6, y=0.5, confidence=1.0), now=1.5)

        self.assertAlmostEqual(first.x_delta_deg, 0.0)
        self.assertAlmostEqual(second.x_delta_deg, 0.5)
        self.assertIsNotNone(second.x_output)
        assert second.x_output is not None
        self.assertAlmostEqual(second.x_output.i, 0.5)


if __name__ == "__main__":
    unittest.main()
