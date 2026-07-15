from __future__ import annotations

from twopoint_project.config import CenterThenFlashConfig, RuntimeConfig, TaskConfig
from twopoint_project.contrl import center_then_flash


def run(task_config: TaskConfig, runtime_config: RuntimeConfig) -> bool:
    if not isinstance(task_config, CenterThenFlashConfig):
        raise TypeError("center_then_flash runner requires CenterThenFlashConfig")
    return center_then_flash.run(task_config, runtime_config)
