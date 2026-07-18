from __future__ import annotations

from types import SimpleNamespace
from math import atan2, degrees
import unittest

from twopoint_project.config import TrackClosedLoopConfig
from twopoint_project.contrl.laser_alignment_servo import LaserAlignmentServo
from twopoint_project.contrl.target_center_servo import PIDAxisGains
from twopoint_project.f32c.gimbal import GimbalAngles
from twopoint_project.vision.pipeline import VisionResult


def make_vision(
    *,
    frame_id: int,
    captured_at_ns: int,
    target_x: float,
    target_y: float,
    laser_x: float,
    laser_y: float,
    distance_cm: float | None = 100.0,
    target_corners_normalized: tuple[tuple[float, float], ...] = (
        (0.2, 0.2),
        (0.8, 0.2),
        (0.8, 0.8),
        (0.2, 0.8),
    ),
) -> VisionResult:
    return VisionResult(
        camera_id="test",
        stream_generation=1,
        source_frame_id=frame_id,
        captured_at_monotonic_ns=captured_at_ns,
        inference_started_monotonic_ns=captured_at_ns + 1,
        inference_finished_monotonic_ns=captured_at_ns + 2,
        captured=SimpleNamespace(frame_id=frame_id, frame_bgr=object()),
        points=[
            {"label": "target_center", "x": target_x, "y": target_y, "confidence": 1.0},
            {"label": "laser_point", "x": laser_x, "y": laser_y, "confidence": 1.0},
        ],
        target_distance_cm=distance_cm,
        target_corners_normalized=target_corners_normalized,
    )


def make_config() -> TrackClosedLoopConfig:
    gains = PIDAxisGains(kp=0.5, output_limit_deg=2.0)
    return TrackClosedLoopConfig(
        max_vision_age_seconds=1.0,
        angle_deadband_deg=0.01,
        target_width_cm=29.7,
        target_height_cm=21.0,
        x_angle_scale=1.0,
        y_angle_scale=-1.0,
        max_visual_correction_deg=5.0,
        x_pid=gains,
        y_pid=gains,
    )


class LaserAlignmentServoTest(unittest.TestCase):
    def test_visual_target_uses_motor_angle_at_capture_time(self) -> None:
        servo = LaserAlignmentServo(
            config=make_config(),
            confidence_threshold=0.5,
            visual_deadband=0.006,
        )
        base = 1_000_000_000
        servo.record_angles(GimbalAngles(0.0, 2.0, base))
        servo.record_angles(GimbalAngles(2.0, 4.0, base + 100_000_000))
        current = GimbalAngles(2.0, 4.0, base + 100_000_000)

        update = servo.accept_vision(
            make_vision(
                frame_id=7,
                captured_at_ns=base + 50_000_000,
                target_x=0.6,
                target_y=0.4,
                laser_x=0.5,
                laser_y=0.5,
            ),
            current,
            now_monotonic_ns=base + 100_000_000,
        )

        self.assertTrue(update.valid)
        assert servo.target is not None
        expected_x = degrees(atan2(4.95, 100.0))
        expected_y = -degrees(atan2(-3.5, 100.0))
        self.assertAlmostEqual(servo.target.target_plane_error_x_cm, 4.95, places=5)
        self.assertAlmostEqual(servo.target.target_plane_error_y_cm, -3.5, places=5)
        self.assertAlmostEqual(servo.target.x_correction_deg, expected_x, places=5)
        self.assertAlmostEqual(servo.target.y_correction_deg, expected_y, places=5)
        self.assertAlmostEqual(servo.target.desired_x_deg, 1.0 + expected_x, places=5)
        self.assertAlmostEqual(servo.target.desired_y_deg, 3.0 + expected_y, places=5)

    def test_motor_pid_recomputes_remaining_angle_between_vision_frames(self) -> None:
        servo = LaserAlignmentServo(
            config=make_config(),
            confidence_threshold=0.5,
            visual_deadband=0.006,
        )
        base = 2_000_000_000
        initial = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(initial)
        servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            initial,
            now_monotonic_ns=base,
        )

        first = servo.compute_motor_update(initial, now=2.0)
        halfway = GimbalAngles(0.5, 0.0, base + 33_000_000)
        second = servo.compute_motor_update(halfway, now=2.033)

        target_angle = degrees(atan2(4.95, 100.0))
        self.assertAlmostEqual(first.x_error_deg, target_angle, places=5)
        self.assertAlmostEqual(first.x_delta_deg, target_angle * 0.5, places=5)
        self.assertAlmostEqual(first.x_command_deg or 0.0, target_angle * 0.5, places=5)
        self.assertAlmostEqual(second.x_error_deg, target_angle - 0.5, places=5)
        self.assertAlmostEqual(second.x_delta_deg, (target_angle - 0.5) * 0.5, places=5)
        self.assertAlmostEqual(
            second.x_command_deg or 0.0,
            0.5 + (target_angle - 0.5) * 0.5,
            places=5,
        )

    def test_same_target_plane_error_produces_smaller_angle_at_longer_distance(self) -> None:
        base = 2_500_000_000
        actual = GimbalAngles(0.0, 0.0, base)

        near_servo = LaserAlignmentServo(
            config=make_config(), confidence_threshold=0.5, visual_deadband=0.006
        )
        near_servo.record_angles(actual)
        near_servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
                distance_cm=50.0,
            ),
            actual,
            now_monotonic_ns=base,
        )

        far_servo = LaserAlignmentServo(
            config=make_config(), confidence_threshold=0.5, visual_deadband=0.006
        )
        far_servo.record_angles(actual)
        far_servo.accept_vision(
            make_vision(
                frame_id=2,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
                distance_cm=150.0,
            ),
            actual,
            now_monotonic_ns=base,
        )

        assert near_servo.target is not None and far_servo.target is not None
        self.assertGreater(
            abs(near_servo.target.x_correction_deg),
            abs(far_servo.target.x_correction_deg),
        )

    def test_invalid_distance_does_not_replace_existing_angle_target(self) -> None:
        servo = LaserAlignmentServo(
            config=make_config(),
            confidence_threshold=0.5,
            visual_deadband=0.006,
        )
        base = 3_000_000_000
        actual = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(actual)
        servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            actual,
            now_monotonic_ns=base,
        )
        original_target = servo.target

        invalid = servo.accept_vision(
            make_vision(
                frame_id=2,
                captured_at_ns=base + 10_000_000,
                target_x=0.4,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
                distance_cm=None,
            ),
            actual,
            now_monotonic_ns=base + 10_000_000,
        )

        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.reason, "laser_or_distance_unavailable")
        self.assertIs(servo.target, original_target)

    def test_missing_target_geometry_does_not_replace_existing_angle_target(self) -> None:
        servo = LaserAlignmentServo(
            config=make_config(), confidence_threshold=0.5, visual_deadband=0.006
        )
        base = 4_000_000_000
        actual = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(actual)
        valid = servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            actual,
            now_monotonic_ns=base,
        )
        self.assertTrue(valid.valid)
        original_target = servo.target

        invalid = servo.accept_vision(
            make_vision(
                frame_id=2,
                captured_at_ns=base,
                target_x=0.4,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
                target_corners_normalized=(),
            ),
            actual,
            now_monotonic_ns=base,
        )

        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.reason, "target_geometry_unavailable")
        self.assertIs(servo.target, original_target)


if __name__ == "__main__":
    unittest.main()
