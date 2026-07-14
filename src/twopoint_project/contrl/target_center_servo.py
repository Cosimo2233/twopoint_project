from __future__ import annotations

from dataclasses import dataclass, replace
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
    x_output: PIDOutput | None = None
    y_output: PIDOutput | None = None


@dataclass(frozen=True)
class PIDAxisGains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.0
    output_limit_deg: float = 1.0


@dataclass(frozen=True)
class PIDOutput:
    p: float
    i: float
    d: float
    raw: float
    clamped: float


@dataclass
class PIDAxisState:
    integral: float = 0.0
    previous_error: float | None = None
    previous_time: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None
        self.previous_time = None

    def update(self, error: float, gains: PIDAxisGains, now: float) -> PIDOutput:
        dt = 0.0 if self.previous_time is None else max(now - self.previous_time, 0.0)
        if gains.integral_limit > 0 and dt > 0:
            self.integral = clamp(
                self.integral + error * dt,
                -abs(gains.integral_limit),
                abs(gains.integral_limit),
            )
        elif gains.integral_limit <= 0:
            self.integral = 0.0

        derivative = 0.0
        if self.previous_error is not None and dt > 0:
            derivative = (error - self.previous_error) / dt

        p = gains.kp * error
        i = gains.ki * self.integral
        d = gains.kd * derivative
        raw = p + i + d
        output_limit = abs(gains.output_limit_deg)
        clamped = clamp(raw, -output_limit, output_limit) if output_limit > 0 else raw

        self.previous_error = error
        self.previous_time = now
        return PIDOutput(p=p, i=i, d=d, raw=raw, clamped=clamped)


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
        x_pid: PIDAxisGains | None = None,
        y_pid: PIDAxisGains | None = None,
    ) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.deadband = abs(deadband)
        self.conf_threshold = validate_conf_threshold(conf_threshold)
        self.x_pid = x_pid or PIDAxisGains(kp=x_gain_deg, output_limit_deg=max_step_deg)
        self.y_pid = y_pid or PIDAxisGains(kp=y_gain_deg, output_limit_deg=max_step_deg)
        self._x_state = PIDAxisState()
        self._y_state = PIDAxisState()

    def compute_error(self, target: TargetCenterObservation) -> CenteringError:
        x_error = target.x - self.center_x
        y_error = target.y - self.center_y
        return CenteringError(
            x=x_error,
            y=y_error,
            distance=hypot(x_error, y_error),
        )

    def compute_step(self, target: TargetCenterObservation, *, now: float | None = None) -> GimbalStep:
        error = self.compute_error(target)
        x_settled = abs(error.x) <= self.deadband
        y_settled = abs(error.y) <= self.deadband
        settled = x_settled and y_settled

        if settled:
            self._x_state.reset()
            self._y_state.reset()
            return GimbalStep(
                x_delta_deg=0.0,
                y_delta_deg=0.0,
                error=error,
                settled=True,
                x_output=None,
                y_output=None,
            )

        now = time.monotonic() if now is None else now
        if x_settled:
            self._x_state.reset()
            x_output = None
            x_delta = 0.0
        else:
            x_output = self._x_state.update(error.x, self.x_pid, now)
            x_delta = x_output.clamped

        if y_settled:
            self._y_state.reset()
            y_output = None
            y_delta = 0.0
        else:
            y_output = self._y_state.update(error.y, self.y_pid, now)
            y_delta = y_output.clamped

        return GimbalStep(
            x_delta_deg=x_delta,
            y_delta_deg=y_delta,
            error=error,
            settled=settled,
            x_output=x_output,
            y_output=y_output,
        )

    def update(
        self,
        gimbal: GimbalLike,
        points: Sequence[PointPrediction],
        *,
        step_scale: float = 1.0,
    ) -> AimUpdate:
        if step_scale < 0:
            raise ValueError("step_scale must be non-negative")
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

        if step_scale != 1.0:
            step = replace(
                step,
                x_delta_deg=step.x_delta_deg * step_scale,
                y_delta_deg=step.y_delta_deg * step_scale,
            )
        moved = step.x_delta_deg != 0.0 or step.y_delta_deg != 0.0
        if moved:
            gimbal.move_by(step.x_delta_deg, step.y_delta_deg)
        return AimUpdate(
            valid=True,
            moved=moved,
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
