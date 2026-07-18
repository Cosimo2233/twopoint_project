"""Two-rate laser alignment control using vision targets and motor feedback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Sequence

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
    x_correction_deg: float
    y_correction_deg: float
    capture_x_deg: float
    capture_y_deg: float
    desired_x_deg: float
    desired_y_deg: float


@dataclass(frozen=True)
class MotorControlUpdate:
    target: AngularTarget | None
    actual: GimbalAngles
    x_error_deg: float
    y_error_deg: float
    x_cumulative_moved_deg: float
    y_cumulative_moved_deg: float
    x_command_deg: float | None
    y_command_deg: float | None
    x_delta_deg: float
    y_delta_deg: float
    settled: bool
    reason: str | None = None
    x_output: PIDOutput | None = None
    y_output: PIDOutput | None = None


class AngleHistory:
    def __init__(self, max_samples: int = 256) -> None:
        self._samples: deque[GimbalAngles] = deque(maxlen=max_samples)
        self._reject_before_first = False

    def append(self, angles: GimbalAngles) -> None:
        if self._samples and angles.sampled_at_monotonic_ns < self._samples[-1].sampled_at_monotonic_ns:
            raise ValueError("gimbal angle timestamps must be monotonic")
        self._samples.append(angles)

    def clear(self, *, reject_before_first: bool = False) -> None:
        self._samples.clear()
        self._reject_before_first = reject_before_first

    def at(self, timestamp_ns: int) -> GimbalAngles | None:
        if not self._samples:
            return None
        if timestamp_ns < self._samples[0].sampled_at_monotonic_ns:
            return None if self._reject_before_first else self._samples[0]
        if timestamp_ns == self._samples[0].sampled_at_monotonic_ns:
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
        return self._samples[-1]


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

    def handle_feedback_loss(self) -> None:
        """Stop the active correction and discard angles spanning a feedback gap."""
        self.clear_target()
        self.history.clear(reject_before_first=True)

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

        error_x = float(target_point["x"] - laser_point["x"])
        error_y = float(target_point["y"] - laser_point["y"])
        limit = abs(self.config.max_visual_correction_deg)
        correction_x = clamp(
            error_x * self.config.x_angle_gain_deg,
            -limit,
            limit,
        )
        correction_y = clamp(
            error_y * self.config.y_angle_gain_deg,
            -limit,
            limit,
        )
        capture_angles = self.history.at(vision.captured_at_monotonic_ns)
        if capture_angles is None:
            return AimUpdate(
                False,
                False,
                False,
                "angle_feedback_unavailable_at_capture",
                None,
                None,
            )
        self.target = AngularTarget(
            source_frame_id=vision.source_frame_id,
            captured_at_monotonic_ns=vision.captured_at_monotonic_ns,
            visual_error_x=error_x,
            visual_error_y=error_y,
            x_correction_deg=correction_x,
            y_correction_deg=correction_y,
            capture_x_deg=capture_angles.x_deg,
            capture_y_deg=capture_angles.y_deg,
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
                x_cumulative_moved_deg=0.0,
                y_cumulative_moved_deg=0.0,
                x_command_deg=None,
                y_command_deg=None,
                x_delta_deg=0.0,
                y_delta_deg=0.0,
                settled=False,
                reason="no_target",
            )

        target_age_seconds = max(
            now - target.captured_at_monotonic_ns / 1_000_000_000.0,
            0.0,
        )
        if target_age_seconds > self.config.max_vision_age_seconds:
            self.clear_target()
            return MotorControlUpdate(
                target=None,
                actual=actual,
                x_error_deg=0.0,
                y_error_deg=0.0,
                x_cumulative_moved_deg=0.0,
                y_cumulative_moved_deg=0.0,
                x_command_deg=None,
                y_command_deg=None,
                x_delta_deg=0.0,
                y_delta_deg=0.0,
                settled=False,
                reason="target_expired",
            )

        # Each visual result defines a total angular correction at capture time.
        # The 50 Hz motor loop subtracts the encoder-measured cumulative motion
        # since that capture, leaving the angle that this PID cycle still needs
        # to execute. This is algebraically equivalent to desired - actual, but
        # keeps the control contract explicit and never relies on commanded motion.
        x_cumulative_moved = actual.x_deg - target.capture_x_deg
        y_cumulative_moved = actual.y_deg - target.capture_y_deg
        x_error = target.x_correction_deg - x_cumulative_moved
        y_error = target.y_correction_deg - y_cumulative_moved
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
            x_cumulative_moved_deg=x_cumulative_moved,
            y_cumulative_moved_deg=y_cumulative_moved,
            x_command_deg=None if settled else actual.x_deg + x_delta,
            y_command_deg=None if settled else actual.y_deg + y_delta,
            x_delta_deg=x_delta,
            y_delta_deg=y_delta,
            settled=settled,
            x_output=x_output,
            y_output=y_output,
        )
