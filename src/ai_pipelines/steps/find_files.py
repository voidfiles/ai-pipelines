"""find_files step: glob files in a directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import StepExecutionError
from ai_pipelines.expressions import evaluate
from ai_pipelines.models import FindFilesStep


async def execute_find_files(
    step: FindFilesStep, context: PipelineContext
) -> list[dict[str, str]]:
    """Find files matching a glob pattern in a directory.

    Returns a list of dicts with ``name`` and ``path`` keys,
    sorted by filename for deterministic ordering.
    """
    directory = evaluate(step.arguments, context.get_data())
    base = Path(str(directory))

    if not base.is_dir():
        raise StepExecutionError(step.name, f"Directory not found: {directory}")

    files = sorted(
        ({"name": p.name, "path": str(p)} for p in base.glob(step.pattern) if p.is_file()),
        key=lambda f: f["name"],
    )
    return files
