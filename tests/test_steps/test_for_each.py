"""Tests for for_each step executor.

The for_each step iterates over an array and runs nested steps per item.
It handles both `kind: for_each` and `kind: pipeline` YAML syntax.
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import StepExecutionError
from ai_pipelines.models import ForEachStep, Step, TransformStep
from ai_pipelines.steps.for_each import execute_for_each


async def simple_step_executor(step: Step, context: PipelineContext) -> Any:
    """Minimal step executor that only handles transform steps."""
    if isinstance(step, TransformStep):
        from ai_pipelines.steps.transform import execute_transform

        return await execute_transform(step, context)
    raise StepExecutionError(step.name, f"Unsupported step kind: {step.kind}")


@pytest.mark.asyncio
async def test_for_each_simple():
    step = ForEachStep(
        kind="for_each",
        name="results",
        arguments="input.items",
        steps=[
            TransformStep(
                kind="transform", name="doubled", arguments="item * 2"
            )
        ],
    )
    ctx = PipelineContext({"items": [1, 2, 3]})
    result = await execute_for_each(step, ctx, step_executor=simple_step_executor)
    assert result == [2, 4, 6]


@pytest.mark.asyncio
async def test_for_each_with_object_items():
    step = ForEachStep(
        kind="for_each",
        name="results",
        arguments="input.files",
        steps=[
            TransformStep(
                kind="transform",
                name="extracted",
                arguments='{"name": item.name}',
            )
        ],
    )
    ctx = PipelineContext(
        {"files": [{"name": "a.txt", "size": 10}, {"name": "b.txt", "size": 20}]}
    )
    result = await execute_for_each(step, ctx, step_executor=simple_step_executor)
    assert result == [{"name": "a.txt"}, {"name": "b.txt"}]


@pytest.mark.asyncio
async def test_for_each_multiple_nested_steps():
    """Last step's result is collected per iteration."""
    step = ForEachStep(
        kind="for_each",
        name="results",
        arguments="input.items",
        steps=[
            TransformStep(
                kind="transform", name="step1", arguments="item * 10"
            ),
            TransformStep(
                kind="transform", name="step2", arguments="step1 + 1"
            ),
        ],
    )
    ctx = PipelineContext({"items": [1, 2, 3]})
    result = await execute_for_each(step, ctx, step_executor=simple_step_executor)
    # Each iteration: item*10 + 1
    assert result == [11, 21, 31]


@pytest.mark.asyncio
async def test_for_each_child_context_isolation():
    """Inner step results don't leak across iterations."""
    step = ForEachStep(
        kind="for_each",
        name="results",
        arguments="input.items",
        steps=[
            TransformStep(
                kind="transform", name="inner", arguments="item"
            ),
        ],
    )
    ctx = PipelineContext({"items": ["a", "b", "c"]})
    result = await execute_for_each(step, ctx, step_executor=simple_step_executor)
    assert result == ["a", "b", "c"]
    # Parent context should not have 'inner' or 'item'
    assert "inner" not in ctx.get_data()
    assert "item" not in ctx.get_data()


@pytest.mark.asyncio
async def test_for_each_empty_array():
    step = ForEachStep(
        kind="for_each",
        name="results",
        arguments="input.items",
        steps=[
            TransformStep(kind="transform", name="x", arguments="item"),
        ],
    )
    ctx = PipelineContext({"items": []})
    result = await execute_for_each(step, ctx, step_executor=simple_step_executor)
    assert result == []


@pytest.mark.asyncio
async def test_pipeline_kind_works_same_as_for_each():
    """kind: pipeline is just an alias for for_each."""
    step = ForEachStep(
        kind="pipeline",
        name="sessions",
        arguments="input.files",
        steps=[
            TransformStep(
                kind="transform", name="mapped", arguments="item.name"
            ),
        ],
    )
    ctx = PipelineContext({"files": [{"name": "a"}, {"name": "b"}]})
    result = await execute_for_each(step, ctx, step_executor=simple_step_executor)
    assert result == ["a", "b"]


@pytest.mark.asyncio
async def test_for_each_non_iterable_raises():
    step = ForEachStep(
        kind="for_each",
        name="results",
        arguments="input.value",
        steps=[
            TransformStep(kind="transform", name="x", arguments="item"),
        ],
    )
    ctx = PipelineContext({"value": 42})
    with pytest.raises(StepExecutionError, match="must resolve to a list"):
        await execute_for_each(step, ctx, step_executor=simple_step_executor)


@pytest.mark.asyncio
async def test_for_each_sees_parent_context():
    """Nested steps can reference data from the parent scope."""
    step = ForEachStep(
        kind="for_each",
        name="results",
        arguments="input.items",
        steps=[
            TransformStep(
                kind="transform",
                name="combined",
                arguments='"prefix: " & parent_data & " / " & item',
            ),
        ],
    )
    ctx = PipelineContext({"items": ["a", "b"]})
    ctx.set_result("parent_data", "hello")
    result = await execute_for_each(step, ctx, step_executor=simple_step_executor)
    assert result == ["prefix: hello / a", "prefix: hello / b"]
