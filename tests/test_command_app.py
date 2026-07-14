from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from twopoint_project.command.app import app
from twopoint_project.config import load_task_config


class CommandAppTest(unittest.TestCase):
    def test_run_loads_config_and_dispatches_task(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "configs/tasks/center_then_flash.json"
        runner = CliRunner()
        seen: dict[str, object] = {}

        def fake_run_task(task_config: object, runtime_config: object) -> bool:
            seen["task_config"] = task_config
            seen["runtime_config"] = runtime_config
            return True

        env = {
            "TWOPOINT_VISION_BACKEND": "traditional",
            "TWOPOINT_ONNX_PATH": "model-bin/test.onnx",
            "TWOPOINT_IMG_SIZE": "320",
            "TWOPOINT_WEBRTC_ENABLED": "false",
            "TWOPOINT_WEBRTC_HOST": "127.0.0.1",
            "TWOPOINT_WEBRTC_PORT": "18080",
        }
        with patch("twopoint_project.command.app.run_task", side_effect=fake_run_task):
            result = runner.invoke(app, ["run", "--config", str(config_path)], env=env)

        self.assertEqual(result.exit_code, 0, result.output)
        task_config = seen["task_config"]
        runtime_config = seen["runtime_config"]
        self.assertEqual(getattr(task_config, "mode"), "center_then_flash")
        self.assertEqual(runtime_config.vision.backend, "traditional")
        self.assertEqual(runtime_config.vision.onnx_path, "model-bin/test.onnx")
        self.assertEqual(runtime_config.vision.img_size, 320)
        self.assertFalse(runtime_config.webrtc.enabled)
        self.assertEqual(runtime_config.webrtc.host, "127.0.0.1")
        self.assertEqual(runtime_config.webrtc.port, 18080)

    def test_center_flash_track_config_loads_as_implemented_mode(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        task_config = load_task_config(repo_root / "configs/tasks/center_flash_track.json")

        self.assertEqual(task_config.mode, "center_flash_track")
        self.assertFalse(task_config.behavior.exit_after_fire)
        self.assertIsNotNone(task_config.center.pid)
        self.assertGreater(task_config.center.pid.x.output_limit_deg, 0)
        self.assertGreater(task_config.center.pid.y.output_limit_deg, 0)


if __name__ == "__main__":
    unittest.main()
