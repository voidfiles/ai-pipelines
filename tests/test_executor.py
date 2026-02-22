"""Tests for the pipeline executor.

Integration tests that run full multi-step pipelines with real step
executors and a fake LLM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

from ai_pipelines.errors import StepExecutionError, ValidationError
from ai_pipelines.executor import execute_step, run_pipeline
from ai_pipelines.models import (
    ChunkStep,
    FindFilesStep,
    ForEachStep,
    PipelineDefinition,
    TransformStep,
)


# ── Helpers ───────────────────────────────────────────────────────


async def fake_llm_fn(
    *, prompt: str, options: ClaudeAgentOptions | None = None, **kwargs: Any
) -> AsyncIterator[ResultMessage]:
    """Fake LLM that echoes a structured response."""
    yield ResultMessage(
        subtype="success",
        duration_ms=50,
        duration_api_ms=40,
        is_error=False,
        num_turns=1,
        session_id="test",
        total_cost_usd=0.001,
        result="echo response",
        structured_output={"echo": prompt[:50]},
    )


# ── Step dispatch ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_step_dispatches_transform():
    from ai_pipelines.context import PipelineContext

    step = TransformStep(kind="transform", name="x", arguments="input.val")
    ctx = PipelineContext({"val": 42})
    result = await execute_step(step, ctx)
    assert result == 42


@pytest.mark.asyncio
async def test_execute_step_dispatches_find_files(tmp_path):
    from ai_pipelines.context import PipelineContext

    (tmp_path / "a.txt").write_text("hi")
    step = FindFilesStep(
        kind="find_files", name="f", arguments="input.dir", pattern="*.txt"
    )
    ctx = PipelineContext({"dir": str(tmp_path)})
    result = await execute_step(step, ctx)
    assert len(result) == 1


# ── Full pipeline runs ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_find_and_transform(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")

    defn = PipelineDefinition(
        input={
            "type": "object",
            "properties": {"dir": {"type": "string"}},
            "required": ["dir"],
        },
        steps=[
            FindFilesStep(
                kind="find_files",
                name="files",
                arguments="input.dir",
                pattern="*.txt",
            ),
            TransformStep(
                kind="transform",
                name="count",
                arguments="$count(files)",
            ),
        ],
    )

    result = await run_pipeline(defn, {"dir": str(tmp_path)})
    assert result.output == 2
    assert len(result.step_results) == 2
    assert result.total_duration_ms > 0


@pytest.mark.asyncio
async def test_pipeline_with_for_each(tmp_path):
    defn = PipelineDefinition(
        input={
            "type": "object",
            "properties": {"items": {"type": "array"}},
        },
        steps=[
            ForEachStep(
                kind="for_each",
                name="doubled",
                arguments="input.items",
                steps=[
                    TransformStep(
                        kind="transform", name="d", arguments="item * 2"
                    ),
                ],
            ),
        ],
    )

    result = await run_pipeline(defn, {"items": [1, 2, 3]})
    assert result.output == [2, 4, 6]


@pytest.mark.asyncio
async def test_pipeline_chunk_and_for_each():
    defn = PipelineDefinition(
        input={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
        steps=[
            ChunkStep(
                kind="chunk",
                name="chunked",
                arguments="input.text",
                chunk_size=10,
                overlap=0,
            ),
            ForEachStep(
                kind="for_each",
                name="processed",
                arguments="chunked.chunks",
                steps=[
                    TransformStep(
                        kind="transform",
                        name="info",
                        arguments='{"index": item.index, "length": $length(item.text)}',
                    ),
                ],
            ),
        ],
    )

    result = await run_pipeline(defn, {"text": "a" * 25})
    assert len(result.output) == 3
    assert result.output[0]["index"] == 0
    assert result.output[0]["length"] == 10


@pytest.mark.asyncio
async def test_pipeline_invalid_input():
    defn = PipelineDefinition(
        input={
            "type": "object",
            "properties": {"dir": {"type": "string"}},
            "required": ["dir"],
        },
        steps=[],
    )

    with pytest.raises(ValidationError):
        await run_pipeline(defn, {})


@pytest.mark.asyncio
async def test_pipeline_step_failure_stops_execution(tmp_path):
    defn = PipelineDefinition(
        input={"type": "object", "properties": {}},
        steps=[
            FindFilesStep(
                kind="find_files",
                name="files",
                arguments='"nonexistent_dir"',
                pattern="*.txt",
            ),
            TransformStep(
                kind="transform",
                name="should_not_reach",
                arguments="files",
            ),
        ],
    )

    with pytest.raises(StepExecutionError, match="files"):
        await run_pipeline(defn, {})


@pytest.mark.asyncio
async def test_pipeline_tracks_timing():
    defn = PipelineDefinition(
        input={"type": "object", "properties": {}},
        steps=[
            TransformStep(
                kind="transform", name="a", arguments="42"
            ),
            TransformStep(
                kind="transform", name="b", arguments="a + 1"
            ),
        ],
    )

    result = await run_pipeline(defn, {})
    assert result.output == 43
    assert result.total_duration_ms > 0
    for sr in result.step_results:
        assert sr.duration_ms >= 0
