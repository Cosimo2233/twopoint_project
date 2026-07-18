"""Two-rate laser alignment control using vision targets and motor feedback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite
from typing import Sequence

import cv2
import numpy as np

from twopoint_project.config import TrackClosedLoopConfig
from twopoint_project.contrl.target_center_servo import (
    AimUpdate,
    CenteringError,
    GimbalStep,
    PIDAxisState,
    PIDOutput,
    PointPrediction,
    TargetCenterObservation,
    clamp,
)
from twopoint_project.f32c.gimbal import GimbalAngles
from twopoint_project.vision.pipeline import VisionResult
from twopoint_project.vision3.laser_area_mapping import LASER_POINT_LABEL


TARGET_CENTER_LABEL = "target_center"


@dataclass(frozen=True)
class AngularTarget:
    source_frame_id: int
    captured_at_monotonic_ns: int
    visual_error_x: float
    visual_error_y: float
    target_plane_error_x_cm: float
    target_plane_error_y_cm: float
    target_distance_cm: float
    x_correction_deg: float
    y_correction_deg: float
    desired_x_deg: float
    desired_y_deg: float


@dataclass(frozen=True)
class MotorControlUpdate:
    target: AngularTarget | None
    actual: GimbalAngles
    x_error_deg: float
    y_error_deg: float
    x_command_deg: float | None
    y_command_deg: float | None
    x_delta_deg: float
    y_delta_deg: float
    settled: bool
    x_output: PIDOutput | None = None
    y_output: PIDOutput | None = None


class AngleHistory:
    def __init__(self, max_samples: int = 256) -> None:
        self._samples: deque[GimbalAngles] = deque(maxlen=max_samples)

    def append(self, angles: GimbalAngles) -> None:
        if self._samples and angles.sampled_at_monotonic_ns < self._samples[-1].sampled_at_monotonic_ns:
            raise ValueError("gimbal angle timestamps must be monotonic")
        self._samples.append(angles)

    def at(self, timestamp_ns: int, fallback: GimbalAngles) -> GimbalAngles:
        if not self._samples:
            return fallback
        if timestamp_ns <= self._samples[0].sampled_at_monotonic_ns:
            return self._samples[0]
        if timestamp_ns >= self._samples[-1].sampled_at_monotonic_ns:
            return self._samples[-1]

        previous = self._samples[0]
        for current in tuple(self._samples)[1:]:
            if current.sampled_at_monotonic_ns >= timestamp_ns:
                interval = current.sampled_at_monotonic_ns - previous.sampled_at_monotonic_ns
                ratio = 0.0 if interval <= 0 else (
                    (timestamp_ns - previous.sampled_at_monotonic_ns) / interval
                )
                return GimbalAngles(
                    x_deg=previous.x_deg + (current.x_deg - previous.x_deg) * ratio,
                    y_deg=previous.y_deg + (current.y_deg - previous.y_deg) * ratio,
                    sampled_at_monotonic_ns=timestamp_ns,
                    feedback_valid=previous.feedback_valid and current.feedback_valid,
                )
            previous = current
        return fallback


def select_point(
    points: Sequence[PointPrediction],
    label: str,
    confidence_threshold: float,
) -> PointPrediction | None:
    candidates = [
        point
        for point in points
        if point["label"] == label and point["confidence"] >= confidence_threshold
    ]
    return max(candidates, key=lambda point: point["confidence"], default=None)


def target_plane_error_cm(
    *,
    target_x: float,
    target_y: float,
    laser_x: float,
    laser_y: float,
    corners_normalized: Sequence[tuple[float, float]],
    target_width_cm: float,
    target_height_cm: float,
) -> tuple[float, float] | None:
    """Map normalized image points onto the physical target plane."""
    if len(corners_normalized) != 4:
        return None
    source = np.asarray(corners_normalized, dtype=np.float32)
    if source.shape != (4, 2) or not np.isfinite(source).all():
        return None
    if abs(float(cv2.contourArea(source))) <= 1e-9:
        return None
    destination = np.asarray(
        [
            (0.0, 0.0),
            (target_width_cm, 0.0),
            (target_width_cm, target_height_cm),
            (0.0, target_height_cm),
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    if not np.isfinite(transform).all():
        return None
    image_points = np.asarray(
        [[(target_x, target_y)], [(laser_x, laser_y)]],
        dtype=np.float32,
    )
    plane_points = cv2.perspectiveTransform(image_points, transform)
    if plane_points.shape != (2, 1, 2) or not np.isfinite(plane_points).all():
        return None
    target_plane = plane_points[0, 0]
    laser_plane = plane_points[1, 0]
    return (
        float(target_plane[0] - laser_plane[0]),
        float(target_plane[1] - laser_plane[1]),
    )


class LaserAlignmentServo:
    """Turn fresh target/laser observations into an encoder-feedback angle target."""

    def __init__(
        self,
        *,
        config: TrackClosedLoopConfig,
        confidence_threshold: float,
        visual_deadband: float,
    ) -> None:
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.visual_deadband = abs(visual_deadband)
        self.history = AngleHistory()
        self.target: AngularTarget | None = None
        self._x_state = PIDAxisState()
        self._y_state = PIDAxisState()

    def record_angles(self, angles: GimbalAngles) -> None:
        self.history.append(angles)

    def clear_target(self) -> None:
        self.target = None
        self._x_state.reset()
        self._y_state.reset()

    def accept_vision(
        self,
        vision: VisionResult,
        actual: GimbalAngles,
        *,
        now_monotonic_ns: int,
    ) -> AimUpdate:
        age_seconds = max(
            now_monotonic_ns - vision.captured_at_monotonic_ns,
            0,
        ) / 1_000_000_000.0
        if age_seconds > self.config.max_vision_age_seconds:
            return AimUpdate(False, False, False, "vision_result_stale", None, None)

        target_point = select_point(
            vision.points,
            TARGET_CENTER_LABEL,
            self.confidence_threshold,
        )
        laser_point = select_point(
            vision.points,
            LASER_POINT_LABEL,
            self.confidence_threshold,
        )
        if target_point is None:
            return AimUpdate(False, False, False, "target_center_low_confidence", None, None)
        if laser_point is None or vision.target_distance_cm is None:
            return AimUpdate(False, False, False, "laser_or_distance_unavailable", None, None)

        distance_cm = float(vision.target_distance_cm)
        if not isfinite(distance_cm) or distance_cm <= 0.0:
            return AimUpdate(False, False, False, "distance_invalid", None, None)

        error_x = float(target_point["x"] - laser_point["x"])
        error_y = float(target_point["y"] - laser_point["y"])
        plane_error = target_plane_error_cm(
            target_x=float(target_point["x"]),
            target_y=float(target_point["y"]),
            laser_x=float(laser_point["x"]),
            laser_y=float(laser_point["y"]),
            corners_normalized=vision.target_corners_normalized,
            target_width_cm=self.config.target_width_cm,
            target_height_cm=self.config.target_height_cm,
        )
        if plane_error is None:
            return AimUpdate(False, False, False, "target_geometry_unavailable", None, None)
        plane_error_x_cm, plane_error_y_cm = plane_error
        limit = abs(self.config.max_visual_correction_deg)
        correction_x = clamp(
            degrees(atan2(plane_error_x_cm, distance_cm)) * self.config.x_angle_scale,
            -limit,
            limit,
        )
        correction_y = clamp(
            degrees(atan2(plane_error_y_cm, distance_cm)) * self.config.y_angle_scale,
            -limit,
            limit,
        )
        capture_angles = self.history.at(vision.captured_at_monotonic_ns, actual)
        self.target = AngularTarget(
            source_frame_id=vision.source_frame_id,
            captured_at_monotonic_ns=vision.captured_at_monotonic_ns,
            visual_error_x=error_x,
            visual_error_y=error_y,
            target_plane_error_x_cm=plane_error_x_cm,
            target_plane_error_y_cm=plane_error_y_cm,
            target_distance_cm=distance_cm,
            x_correction_deg=correction_x,
            y_correction_deg=correction_y,
            desired_x_deg=capture_angles.x_deg + correction_x,
            desired_y_deg=capture_angles.y_deg + correction_y,
        )
        # A new vision frame changes the angle setpoint discontinuously. Resetting
        # the motor-loop history avoids derivative kick from that target change.
        self._x_state.reset()
        self._y_state.reset()

        visual_error = CenteringError(
            x=error_x,
            y=error_y,
            distance=hypot(error_x, error_y),
        )
        settled = abs(error_x) <= self.visual_deadband and abs(error_y) <= self.visual_deadband
        observation = TargetCenterObservation(
            x=float(target_point["x"]),
            y=float(target_point["y"]),
            confidence=float(target_point["confidence"]),
        )
        return AimUpdate(
            valid=True,
            moved=False,
            settled=settled,
            reason=None,
            target=observation,
            step=GimbalStep(
                x_delta_deg=correction_x,
                y_delta_deg=correction_y,
                error=visual_error,
                settled=settled,
            ),
        )

    @staticmethod
    def _bounded_delta(error: float, output: float) -> float:
        return clamp(output, min(0.0, error), max(0.0, error))

    def compute_motor_update(
        self,
        actual: GimbalAngles,
        *,
        now: float,
    ) -> MotorControlUpdate:
        target = self.target
        if target is None:
            return MotorControlUpdate(
                target=None,
                actual=actual,
                x_error_deg=0.0,
                y_error_deg=0.0,
                x_command_deg=None,
                y_command_deg=None,
                x_delta_deg=0.0,
                y_delta_deg=0.0,
                settled=False,
            )

        x_error = target.desired_x_deg - actual.x_deg
        y_error = target.desired_y_deg - actual.y_deg
        x_settled = abs(x_error) <= self.config.angle_deadband_deg
        y_settled = abs(y_error) <= self.config.angle_deadband_deg

        if x_settled:
            self._x_state.reset()
            x_output = None
            x_delta = 0.0
        else:
            x_output = self._x_state.update(x_error, self.config.x_pid, now)
            x_delta = self._bounded_delta(x_error, x_output.clamped)

        if y_settled:
            self._y_state.reset()
            y_output = None
            y_delta = 0.0
        else:
            y_output = self._y_state.update(y_error, self.config.y_pid, now)
            y_delta = self._bounded_delta(y_error, y_output.clamped)

        settled = x_settled and y_settled
        return MotorControlUpdate(
            target=target,
            actual=actual,
            x_error_deg=x_error,
            y_error_deg=y_error,
            x_command_deg=None if settled else actual.x_deg + x_delta,
            y_command_deg=None if settled else actual.y_deg + y_delta,
            x_delta_deg=x_delta,
            y_delta_deg=y_delta,
            settled=settled,
            x_output=x_output,
            y_output=y_output,
        )
