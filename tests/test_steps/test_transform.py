"""Tests for transform step executor."""

from __future__ import annotations

import pytest

from ai_pipelines.context import PipelineContext
from ai_pipelines.models import TransformStep
from ai_pipelines.steps.transform import execute_transform


@pytest.mark.asyncio
async def test_passthrough():
    step = TransformStep(kind="transform", name="out", arguments="input.items")
    ctx = PipelineContext({"items": [1, 2, 3]})
    result = await execute_transform(step, ctx)
    assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_object_construction():
    step = TransformStep(
        kind="transform",
        name="out",
        arguments='{"filename": item.name, "title": item.name}',
    )
    ctx = PipelineContext({})
    ctx = ctx.child({"item": {"name": "test.md"}})
    result = await execute_transform(step, ctx)
    assert result == {"filename": "test.md", "title": "test.md"}


@pytest.mark.asyncio
async def test_filtering():
    step = TransformStep(
        kind="transform",
        name="out",
        arguments="claims[verified = true]",
    )
    ctx = PipelineContext({})
    ctx = ctx.child({
        "claims": [
            {"text": "a", "verified": True},
            {"text": "b", "verified": False},
            {"text": "c", "verified": True},
        ]
    })
    result = await execute_transform(step, ctx)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_nested_path():
    step = TransformStep(
        kind="transform", name="out", arguments="step1.data.value"
    )
    ctx = PipelineContext({})
    ctx.set_result("step1", {"data": {"value": 42}})
    result = await execute_transform(step, ctx)
    assert result == 42


@pytest.mark.asyncio
async def test_array_projection():
    step = TransformStep(
        kind="transform", name="out", arguments="items.name"
    )
    ctx = PipelineContext({})
    ctx = ctx.child({"items": [{"name": "a"}, {"name": "b"}]})
    result = await execute_transform(step, ctx)
    assert result == ["a", "b"]
