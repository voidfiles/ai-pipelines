"""Tests for read_file step executor."""

from __future__ import annotations

import pytest

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import StepExecutionError
from ai_pipelines.models import ReadFileStep
from ai_pipelines.steps.read_file import execute_read_file


@pytest.mark.asyncio
async def test_read_file_content(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello, world!")
    step = ReadFileStep(kind="read_file", name="content", arguments="item.path")
    ctx = PipelineContext({})
    ctx_data = ctx.child({"item": {"path": str(f)}})
    result = await execute_read_file(step, ctx_data)
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_read_file_multiline(tmp_path):
    f = tmp_path / "multi.txt"
    f.write_text("line 1\nline 2\nline 3")
    step = ReadFileStep(kind="read_file", name="content", arguments="item")
    ctx = PipelineContext({})
    ctx_data = ctx.child({"item": str(f)})
    result = await execute_read_file(step, ctx_data)
    assert result == "line 1\nline 2\nline 3"


@pytest.mark.asyncio
async def test_read_file_not_found():
    step = ReadFileStep(kind="read_file", name="content", arguments="input.path")
    ctx = PipelineContext({"path": "/nonexistent/file.txt"})
    with pytest.raises(StepExecutionError, match="File not found"):
        await execute_read_file(step, ctx)


@pytest.mark.asyncio
async def test_read_file_utf8(tmp_path):
    f = tmp_path / "unicode.txt"
    f.write_text("Hello 🌍 world", encoding="utf-8")
    step = ReadFileStep(kind="read_file", name="content", arguments="input.path")
    ctx = PipelineContext({"path": str(f)})
    result = await execute_read_file(step, ctx)
    assert "🌍" in result
