"""Pipeline executor: the main orchestrator.

Dispatches steps to their executors, accumulates results in context,
and tracks timing and cost.
"""

from __future__ import annotations

import time
from typing import Any

from ai_pipelines import pipeline_logger
from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import PipelineError, StepExecutionError
from ai_pipelines.loader import validate_input
from ai_pipelines.models import (
    ChunkStep,
    EvaluateStep,
    FindFilesStep,
    ForEachStep,
    PipelineDefinition,
    PipelineResult,
    PromptStep,
    ReadFileStep,
    Step,
    StepResult,
    TransformStep,
)
from ai_pipelines.steps.chunk import execute_chunk
from ai_pipelines.steps.evaluate import execute_evaluate
from ai_pipelines.steps.find_files import execute_find_files
from ai_pipelines.steps.for_each import execute_for_each
from ai_pipelines.steps.prompt import execute_prompt
from ai_pipelines.steps.read_file import execute_read_file
from ai_pipelines.steps.transform import execute_transform


async def execute_step(step: Step, context: PipelineContext) -> Any:
    """Dispatch a step to its executor based on kind.

    Uses structural pattern matching on the Pydantic model type.
    """
    match step:
        case FindFilesStep():
            return await execute_find_files(step, context)
        case ReadFileStep():
            return await execute_read_file(step, context)
        case TransformStep():
            return await execute_transform(step, context)
        case ChunkStep():
            return await execute_chunk(step, context)
        case PromptStep():
            return await execute_prompt(step, context)
        case EvaluateStep():
            return await execute_evaluate(step, context)
        case ForEachStep():
            return await execute_for_each(
                step, context, step_executor=execute_step
            )
        case _:
            raise StepExecutionError(
                getattr(step, "name", "unknown"),
                f"Unknown step kind: {getattr(step, 'kind', 'unknown')}",
            )


async def run_pipeline(
    definition: PipelineDefinition,
    input_data: dict[str, Any],
) -> PipelineResult:
    """Execute a pipeline definition with the given input.

    1. Validates input against the pipeline's input JSON schema.
    2. Creates a PipelineContext with the input.
    3. Executes each step in sequence.
    4. Returns the final PipelineResult.

    Args:
        definition: Parsed pipeline definition.
        input_data: Input data matching the pipeline's input schema.

    Returns:
        PipelineResult with output, step results, timing, and cost.

    Raises:
        ValidationError: If input doesn't match the schema.
        StepExecutionError: If any step fails.
    """
    validate_input(definition.input, input_data)

    context = PipelineContext(input_data)
    step_results: list[StepResult] = []
    total_cost = 0.0
    start = time.monotonic()

    last_result: Any = None
    for step in definition.steps:
        step_start = time.monotonic()
        pipeline_logger.log_step_start(step.name, step.kind)

        try:
            last_result = await execute_step(step, context)
        except PipelineError:
            raise
        except Exception as e:
            raise StepExecutionError(step.name, str(e), cause=e) from e

        duration_ms = (time.monotonic() - step_start) * 1000

        context.set_result(step.name, last_result)
        step_results.append(
            StepResult(
                step_name=step.name,
                value=last_result,
                duration_ms=duration_ms,
            )
        )

        pipeline_logger.log_step_complete(step.name, duration_ms)

    total_ms = (time.monotonic() - start) * 1000

    return PipelineResult(
        output=last_result,
        step_results=step_results,
        total_duration_ms=total_ms,
        total_cost_usd=total_cost,
    )
