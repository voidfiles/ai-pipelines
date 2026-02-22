"""transform step: reshape data via JSONata expressions."""

from __future__ import annotations

from typing import Any

from ai_pipelines.context import PipelineContext
from ai_pipelines.expressions import evaluate
from ai_pipelines.models import TransformStep


async def execute_transform(
    step: TransformStep, context: PipelineContext
) -> Any:
    """Evaluate the step's arguments expression and return the result.

    This is the simplest step: just evaluate a JSONata expression.
    Use it to reshape data, filter arrays, construct objects, or
    project fields from arrays.
    """
    return evaluate(step.arguments, context.get_data())
