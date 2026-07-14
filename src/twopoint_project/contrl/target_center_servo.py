from __future__ import annotations

from dataclasses import dataclass
from math import hypot
import time
from typing import Protocol, Sequence, TypedDict


TARGET_CENTER_LABEL = "target_center"
DEFAULT_CONTROL_CONF_THRESHOLD = 0.5


def validate_conf_threshold(conf_threshold: float) -> float:
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf-threshold must be between 0 and 1")
    return conf_threshold


def control_conf_threshold() -> float:
    return DEFAULT_CONTROL_CONF_THRESHOLD


class PointPrediction(TypedDict):
    label: str
    x: float
    y: float
    confidence: float


class GimbalLike(Protocol):
    def move_by(self, x_delta_deg: float, y_delta_deg: float) -> None:
        ...


@dataclass(frozen=True)
class TargetCenterObservation:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class CenteringError:
    x: float
    y: float
    distance: float


@dataclass(frozen=True)
class GimbalStep:
    x_delta_deg: float
    y_delta_deg: float
    error: CenteringError
    settled: bool


@dataclass(frozen=True)
class AimUpdate:
    valid: bool
    moved: bool
    settled: bool
    reason: str | None
    target: TargetCenterObservation | None
    step: GimbalStep | None


def select_target_center(
    points: Sequence[PointPrediction],
    *,
    conf_threshold: float = DEFAULT_CONTROL_CONF_THRESHOLD,
) -> TargetCenterObservation | None:
    conf_threshold = validate_conf_threshold(conf_threshold)
    candidates = [
        point
        for point in points
        if point["label"] == TARGET_CENTER_LABEL and point["confidence"] >= conf_threshold
    ]
    if not candidates:
        return None

    point = max(candidates, key=lambda item: item["confidence"])
    return TargetCenterObservation(
        x=float(point["x"]),
        y=float(point["y"]),
        confidence=float(point["confidence"]),
    )


def clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


class TargetCenterServo:
    def __init__(
        self,
        *,
        center_x: float = 0.5,
        center_y: float = 0.5,
        x_gain_deg: float = 8.0,
        y_gain_deg: float = -8.0,
        max_step_deg: float = 1.0,
        deadband: float = 0.006,
        conf_threshold: float = DEFAULT_CONTROL_CONF_THRESHOLD,
    ) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.x_gain_deg = x_gain_deg
        self.y_gain_deg = y_gain_deg
        self.max_step_deg = abs(max_step_deg)
        self.deadband = abs(deadband)
        self.conf_threshold = validate_conf_threshold(conf_threshold)

    def compute_error(self, target: TargetCenterObservation) -> CenteringError:
        x_error = target.x - self.center_x
        y_error = target.y - self.center_y
        return CenteringError(
            x=x_error,
            y=y_error,
            distance=hypot(x_error, y_error),
        )

    def compute_step(self, target: TargetCenterObservation) -> GimbalStep:
        error = self.compute_error(target)
        x_settled = abs(error.x) <= self.deadband
        y_settled = abs(error.y) <= self.deadband
        settled = x_settled and y_settled

        x_delta = 0.0 if x_settled else self.x_gain_deg * error.x
        y_delta = 0.0 if y_settled else self.y_gain_deg * error.y

        return GimbalStep(
            x_delta_deg=clamp(x_delta, -self.max_step_deg, self.max_step_deg),
            y_delta_deg=clamp(y_delta, -self.max_step_deg, self.max_step_deg),
            error=error,
            settled=settled,
        )

    def update(
        self,
        gimbal: GimbalLike,
        points: Sequence[PointPrediction],
    ) -> AimUpdate:
        target = select_target_center(points, conf_threshold=self.conf_threshold)
        if target is None:
            return AimUpdate(
                valid=False,
                moved=False,
                settled=False,
                reason="target_center_low_confidence",
                target=None,
                step=None,
            )

        step = self.compute_step(target)
        if step.settled:
            return AimUpdate(
                valid=True,
                moved=False,
                settled=True,
                reason=None,
                target=target,
                step=step,
            )

        gimbal.move_by(step.x_delta_deg, step.y_delta_deg)
        return AimUpdate(
            valid=True,
            moved=True,
            settled=False,
            reason=None,
            target=target,
            step=step,
        )


def sleep_for_loop_rate(loop_started_at: float, loop_hz: float) -> None:
    if loop_hz <= 0:
        return
    min_interval = 1.0 / loop_hz
    elapsed = time.monotonic() - loop_started_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
