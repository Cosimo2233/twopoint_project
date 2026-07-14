from __future__ import annotations

from argparse import Namespace
import unittest
from unittest.mock import patch

import numpy as np

from twopoint_project.contrl import center_then_flash


class FakeCapture:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.frame = Namespace(frame_id=1, frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8))
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


def make_args() -> Namespace:
    return Namespace(
        camera=0,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        port="/dev/null",
        baudrate=115200,
        x_id=1,
        y_id=2,
        speed_rpm=100,
        init_zero=True,
        startup_delay=0,
        command_interval=0,
        enable_settle_delay=0,
        debug_frames=False,
        conf_threshold=0.5,
        center_x=0.5,
        center_y=0.5,
        x_gain_deg=8.0,
        y_gain_deg=-8.0,
        max_step_deg=1.0,
        deadband=0.006,
        loop_hz=0,
        center_timeout=2.0,
        stable_frames=1,
        laser_hold_seconds=0,
        vision_backend="traditional",
        onnx="model-bin/runs/twopoint/best.onnx",
        img_size=640,
    )


class CenterThenFlashTest(unittest.TestCase):
    def test_run_centers_then_turns_laser_on_and_disables_gimbal(self) -> None:
        fake_gimbal = FakeGimbal()
        fake_laser = FakeLaser()

        with patch.object(center_then_flash, "CameraCapture", FakeCapture), patch.object(
            center_then_flash,
            "open_serial_gimbal",
            return_value=fake_gimbal,
        ), patch.object(
            center_then_flash,
            "open_laser_pointer",
            return_value=fake_laser,
        ), patch.object(
            center_then_flash,
            "build_vision_inferencer",
            return_value=FakeInferencer(),
        ) as build_vision_inferencer:
            result = center_then_flash.run(make_args())

        self.assertTrue(result)
        build_vision_inferencer.assert_called_once_with(
            backend="traditional",
            onnx_path="model-bin/runs/twopoint/best.onnx",
            img_size=640,
        )
        self.assertTrue(fake_gimbal.initialized)
        self.assertTrue(fake_gimbal.disabled)
        self.assertEqual(fake_gimbal.moves, [])
        self.assertIn("on", fake_laser.events)
        self.assertEqual(fake_laser.events[-1], "off")


if __name__ == "__main__":
    unittest.main()
