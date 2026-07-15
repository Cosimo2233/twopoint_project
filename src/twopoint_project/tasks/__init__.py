from __future__ import annotations

from typing import Callable

from twopoint_project.config import RuntimeConfig, TaskConfig, UnsupportedTaskConfig


TaskRunner = Callable[[TaskConfig, RuntimeConfig], bool]


def run_task(task_config: TaskConfig, runtime_config: RuntimeConfig) -> bool:
    if task_config.mode == "center_then_flash":
        from twopoint_project.tasks.center_then_flash import run

        return run(task_config, runtime_config)
    if task_config.mode == "center_flash_track":
        from twopoint_project.tasks.center_flash_track import run

        return run(task_config, runtime_config)
    if isinstance(task_config, UnsupportedTaskConfig):
        raise NotImplementedError(f"task mode is registered but not implemented yet: {task_config.mode}")
    raise ValueError(f"unsupported task mode: {task_config.mode}")
