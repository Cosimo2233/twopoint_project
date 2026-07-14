from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from twopoint_project.config import (
    BehaviorConfig,
    CameraConfig,
    CenterConfig,
    CenterThenFlashConfig,
    F32CConfig,
    LaserConfig,
    RuntimeConfig,
    VisionConfig,
    WebRtcConfig,
)
from twopoint_project.contrl import center_then_flash


class FakeCapture:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.frame = SimpleNamespace(frame_id=1, frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8))
        self.closed = False

    def __enter__(self) -> FakeCapture:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def read_frame(self) -> object:
        return self.frame


class FakeGimbal:
    def __init__(self) -> None:
        self.initialized = False
        self.disabled = False
        self.moves: list[tuple[float, float]] = []

    def __enter__(self) -> FakeGimbal:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def initialize(self) -> None:
        self.initialized = True

    def move_by(self, x_delta_deg: float, y_delta_deg: float) -> None:
        self.moves.append((x_delta_deg, y_delta_deg))

    def disable(self) -> None:
        self.disabled = True


class FakeLaser:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.events: list[str] = []

    def __enter__(self) -> FakeLaser:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def on(self) -> None:
        self.events.append("on")

    def off(self) -> None:
        self.events.append("off")


class FakeInferencer:
    @property
    def providers(self) -> list[str]:
        return ["fake"]

    def predict(self, frame_bgr: np.ndarray) -> list[dict[str, float | str]]:
        return [{"label": "target_center", "x": 0.5, "y": 0.5, "confidence": 1.0}]


def make_task_config() -> CenterThenFlashConfig:
    return CenterThenFlashConfig(
        mode="center_then_flash",
        camera=CameraConfig(index=0, width=640, height=480, fps=30),
        f32c=F32CConfig(
            port="/dev/null",
            baudrate=115200,
            x_id=1,
            y_id=2,
            speed_rpm=100,
            startup_delay=0,
            command_interval=0,
            enable_settle_delay=0,
            debug_frames=False,
        ),
        center=CenterConfig(
            conf_threshold=0.5,
            target_x=0.5,
            target_y=0.5,
            x_gain_deg=8.0,
            y_gain_deg=-8.0,
            max_step_deg=1.0,
            deadband=0.006,
            loop_hz=0,
            timeout=2.0,
            stable_frames=1,
        ),
        laser=LaserConfig(hold_seconds=0),
        behavior=BehaviorConfig(fire_after_timeout=True, exit_after_fire=True),
    )


def make_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        config_path=Path("unused.json"),
        vision=VisionConfig(
            backend="traditional",
            onnx_path="model-bin/runs/twopoint/best.onnx",
            img_size=640,
        ),
        webrtc=WebRtcConfig(enabled=False, host="0.0.0.0", port=8080),
    )


class CenterThenFlashTest(unittest.TestCase):
    def test_run_centers_then_turns_laser_on_and_disables_gimbal(self) -> None:
        fake_gimbal = FakeGimbal()
        fake_laser = FakeLaser()

        with patch.object(center_then_flash, "open_camera_capture", return_value=FakeCapture()), patch.object(
            center_then_flash,
            "open_serial_gimbal",
            return_value=fake_gimbal,
        ) as open_serial_gimbal, patch.object(
            center_then_flash,
            "open_laser_pointer",
            return_value=fake_laser,
        ), patch.object(
            center_then_flash,
            "build_vision_inferencer",
            return_value=FakeInferencer(),
        ) as build_vision_inferencer:
            result = center_then_flash.run(make_task_config(), make_runtime_config())

        self.assertTrue(result)
        build_vision_inferencer.assert_called_once_with(
            backend="traditional",
            onnx_path="model-bin/runs/twopoint/best.onnx",
            img_size=640,
        )
        open_serial_gimbal.assert_called_once()
        self.assertNotIn("init_zero", open_serial_gimbal.call_args.kwargs)
        self.assertTrue(fake_gimbal.initialized)
        self.assertTrue(fake_gimbal.disabled)
        self.assertEqual(fake_gimbal.moves, [])
        self.assertIn("on", fake_laser.events)
        self.assertEqual(fake_laser.events[-1], "off")


if __name__ == "__main__":
    unittest.main()
