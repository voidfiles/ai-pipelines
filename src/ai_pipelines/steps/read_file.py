"""read_file step: read file contents as text."""

from __future__ import annotations

from pathlib import Path

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import StepExecutionError
from ai_pipelines.expressions import evaluate
from ai_pipelines.models import ReadFileStep


async def execute_read_file(
    step: ReadFileStep, context: PipelineContext
) -> str:
    """Read a file's contents as UTF-8 text.

    The ``arguments`` expression should resolve to a file path string
    or a dict with a ``path`` key.
    """
    resolved = evaluate(step.arguments, context.get_data())

    # Handle both "item.path" -> str and "item" -> {"path": "..."} patterns
    if isinstance(resolved, dict) and "path" in resolved:
        file_path = resolved["path"]
    else:
        file_path = str(resolved)

    path = Path(file_path)
    if not path.is_file():
        raise StepExecutionError(step.name, f"File not found: {file_path}")

    return path.read_text(encoding="utf-8")
