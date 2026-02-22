"""for_each step: iterate over an array, running nested steps per item.

Handles both ``kind: for_each`` and ``kind: pipeline`` YAML syntax
(they're aliases for the same behavior).
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import StepExecutionError
from ai_pipelines.expressions import evaluate
from ai_pipelines.models import ForEachStep, Step

StepExecutorFn = Callable[[Step, PipelineContext], Coroutine[Any, Any, Any]]


async def execute_for_each(
    step: ForEachStep,
    context: PipelineContext,
    *,
    step_executor: StepExecutorFn,
) -> list[Any]:
    """Iterate over items, executing nested steps for each.

    For each item in the resolved array:
    1. Create a child context with ``item`` bound to the current element.
    2. Execute all nested steps sequentially in that child context.
    3. Collect the last step's result from each iteration.

    Args:
        step: The ForEachStep definition.
        context: Parent pipeline context.
        step_executor: Callable that executes a single Step within a context.
            Injected from the orchestrator to avoid circular imports and
            to make testing straightforward.

    Returns:
        List of results, one per iteration (the last nested step's result).
    """
    items = evaluate(step.arguments, context.get_data())

    if not isinstance(items, list):
        raise StepExecutionError(
            step.name,
            f"arguments must resolve to a list, got {type(items).__name__}",
        )

    results: list[Any] = []

    for i, item in enumerate(items):
        child_ctx = context.child({"item": item, "item_index": i})

        last_result: Any = None
        for nested_step in step.steps:
            last_result = await step_executor(nested_step, child_ctx)
            child_ctx.set_result(nested_step.name, last_result)

        results.append(last_result)

    return results
