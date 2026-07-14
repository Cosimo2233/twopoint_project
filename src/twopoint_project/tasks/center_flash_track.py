from __future__ import annotations

from twopoint_project.config import CenterFlashTrackConfig, RuntimeConfig, TaskConfig
from twopoint_project.contrl import center_then_flash


def run(task_config: TaskConfig, runtime_config: RuntimeConfig) -> bool:
    if not isinstance(task_config, CenterFlashTrackConfig):
        raise TypeError("center_flash_track runner requires CenterFlashTrackConfig")
    return center_then_flash.run_track(task_config, runtime_config)
