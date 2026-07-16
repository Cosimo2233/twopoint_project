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
    CenterFlashTrackConfig,
    CenterThenFlashConfig,
    F32CConfig,
    LaserConfig,
    RecordingConfig,
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


class FakeVisionFrames:
    def __init__(self, frames: list[object]) -> None:
        self.frames = list(frames)

    def read_nowait_latest(self) -> object | None:
        if not self.frames:
            return None
        return self.frames.pop(0)


class FakeVideoRecorder:
    instances: list[FakeVideoRecorder] = []

    def __init__(self, output_path: Path, fps: float, label: str) -> None:
        self.output_path = output_path
        self.fps = fps
        self.label = label
        self.frames: list[object] = []
        FakeVideoRecorder.instances.append(self)

    def write(self, frame_bgr: object) -> None:
        self.frames.append(frame_bgr)

    def close(self) -> None:
        pass


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
        recording=RecordingConfig(),
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


def make_track_config() -> CenterFlashTrackConfig:
    return CenterFlashTrackConfig(
        mode="center_flash_track",
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
        behavior=BehaviorConfig(fire_after_timeout=True, exit_after_fire=False),
        recording=RecordingConfig(),
    )


def make_constant_laser_track_config() -> CenterFlashTrackConfig:
    task_config = make_track_config()
    return CenterFlashTrackConfig(
        mode=task_config.mode,
        camera=task_config.camera,
        f32c=task_config.f32c,
        center=task_config.center,
        laser=LaserConfig(hold_seconds=0, on_during_run=True),
        behavior=task_config.behavior,
        recording=task_config.recording,
    )


class StopAfterCalls:
    def __init__(self, calls: int) -> None:
        self.calls = 0
        self.limit = calls

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.limit


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
            npu_model_path="model-bin/pose/best_pcq_a733.nb",
            npu_library_path="build/npu/libyolo11_pose_npu.so",
            img_size=640,
            npu_score_threshold=0.4,
            npu_nms_threshold=0.45,
            npu_target_keypoint_index=0,
        )
        open_serial_gimbal.assert_called_once()
        self.assertNotIn("init_zero", open_serial_gimbal.call_args.kwargs)
        self.assertTrue(fake_gimbal.initialized)
        self.assertTrue(fake_gimbal.disabled)
        self.assertEqual(fake_gimbal.moves, [])
        self.assertIn("on", fake_laser.events)
        self.assertEqual(fake_laser.events[-1], "off")

    def test_track_mode_fires_then_keeps_aiming_until_stop_requested(self) -> None:
        fake_gimbal = FakeGimbal()
        fake_laser = FakeLaser()
        stop_requested = StopAfterCalls(3)

        with patch.object(center_then_flash, "open_camera_capture", return_value=FakeCapture()), patch.object(
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
        ):
            result = center_then_flash.run_track(make_track_config(), make_runtime_config(), stop_requested)

        self.assertTrue(result)
        self.assertGreaterEqual(stop_requested.calls, 3)
        self.assertTrue(fake_gimbal.initialized)
        self.assertTrue(fake_gimbal.disabled)
        self.assertIn("on", fake_laser.events)
        self.assertEqual(fake_laser.events[-1], "off")

    def test_track_mode_can_keep_laser_on_until_stop_requested(self) -> None:
        fake_gimbal = FakeGimbal()
        fake_laser = FakeLaser()
        stop_requested = StopAfterCalls(3)

        with patch.object(center_then_flash, "open_camera_capture", return_value=FakeCapture()), patch.object(
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
        ):
            result = center_then_flash.run_track(
                make_constant_laser_track_config(),
                make_runtime_config(),
                stop_requested,
            )

        self.assertTrue(result)
        self.assertEqual(fake_laser.events.count("on"), 1)
        self.assertEqual(fake_laser.events[-1], "off")
        self.assertTrue(fake_gimbal.disabled)

    def test_monitor_writes_raw_video_when_enabled(self) -> None:
        FakeVideoRecorder.instances = []
        captured = SimpleNamespace(
            frame_id=1,
            timestamp=0.0,
            frame_bgr=np.zeros((8, 12, 3), dtype=np.uint8),
        )
        update = SimpleNamespace(
            valid=True,
            moved=False,
            settled=True,
            reason="settled",
            target=None,
            step=None,
        )

        with patch.object(center_then_flash, "VideoRecorder", FakeVideoRecorder):
            monitor = center_then_flash.CenterRunMonitor(
                enabled=True,
                output_path=Path("outputs/center_flash_track_20260714_120000.mp4"),
                fps=15.0,
                save_raw_video=True,
                webrtc_host="127.0.0.1",
                webrtc_port=8080,
                backend="traditional",
                providers=["fake"],
                conf_threshold=0.5,
            )
            monitor.on_frame(captured, [], update)

        self.assertEqual([recorder.label for recorder in FakeVideoRecorder.instances], ["annotated", "raw"])
        self.assertEqual(FakeVideoRecorder.instances[1].output_path.name, "center_flash_track_20260714_120000_raw.mp4")
        self.assertEqual(len(FakeVideoRecorder.instances[0].frames), 1)
        self.assertEqual(len(FakeVideoRecorder.instances[1].frames), 1)

    def test_track_target_reuses_last_valid_frame_when_no_new_frame_is_available(self) -> None:
        fake_gimbal = FakeGimbal()
        vision = FakeVisionFrames([
            SimpleNamespace(
                captured=SimpleNamespace(frame_id=1),
                points=[{"label": "target_center", "x": 0.6, "y": 0.45, "confidence": 1.0}],
            )
        ])
        servo = center_then_flash.TargetCenterServo(
            x_gain_deg=10.0,
            y_gain_deg=-10.0,
            max_step_deg=2.0,
            deadband=0.01,
            conf_threshold=0.5,
        )
        stop_requested = StopAfterCalls(3)

        center_then_flash.track_target(
            vision=vision,
            gimbal=fake_gimbal,
            servo=servo,
            loop_hz=0,
            stale_target_seconds=1.0,
            stale_target_step_scale=0.5,
            stop_requested=stop_requested,
        )

        self.assertEqual(len(fake_gimbal.moves), 2)
        self.assertAlmostEqual(fake_gimbal.moves[0][0], 1.0)
        self.assertAlmostEqual(fake_gimbal.moves[0][1], 0.5)
        self.assertAlmostEqual(fake_gimbal.moves[1][0], 0.5)
        self.assertAlmostEqual(fake_gimbal.moves[1][1], 0.25)


if __name__ == "__main__":
    unittest.main()
