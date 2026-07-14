from __future__ import annotations

from pathlib import Path

import typer

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv() -> bool:
        return False

from twopoint_project.config import load_task_config, runtime_config_from_env
from twopoint_project.tasks import run_task


app = typer.Typer(help="Unified twopoint task runner.")


@app.callback()
def main() -> None:
    """Unified twopoint task runner."""


@app.command()
def run(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Task JSON config path. Defaults to TWOPOINT_CONFIG or configs/tasks/center_then_flash.json.",
    ),
) -> None:
    load_dotenv()
    try:
        runtime_config = runtime_config_from_env(config)
        task_config = load_task_config(runtime_config.config_path)
        success = run_task(task_config, runtime_config)
    except NotImplementedError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    raise typer.Exit(0 if success else 1)


if __name__ == "__main__":
    app()
