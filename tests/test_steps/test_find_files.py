"""Tests for find_files step executor."""

from __future__ import annotations

import pytest

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import StepExecutionError
from ai_pipelines.models import FindFilesStep
from ai_pipelines.steps.find_files import execute_find_files


@pytest.fixture
def populated_dir(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    (tmp_path / "c.md").write_text("markdown")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "d.txt").write_text("nested")
    return tmp_path


@pytest.mark.asyncio
async def test_find_txt_files(populated_dir):
    step = FindFilesStep(
        kind="find_files",
        name="files",
        arguments="input.dir",
        pattern="*.txt",
    )
    ctx = PipelineContext({"dir": str(populated_dir)})
    result = await execute_find_files(step, ctx)
    assert len(result) == 2
    filenames = [r["name"] for r in result]
    assert "a.txt" in filenames
    assert "b.txt" in filenames


@pytest.mark.asyncio
async def test_find_md_files(populated_dir):
    step = FindFilesStep(
        kind="find_files",
        name="files",
        arguments="input.dir",
        pattern="*.md",
    )
    ctx = PipelineContext({"dir": str(populated_dir)})
    result = await execute_find_files(step, ctx)
    assert len(result) == 1
    assert result[0]["name"] == "c.md"


@pytest.mark.asyncio
async def test_find_no_matches(populated_dir):
    step = FindFilesStep(
        kind="find_files",
        name="files",
        arguments="input.dir",
        pattern="*.csv",
    )
    ctx = PipelineContext({"dir": str(populated_dir)})
    result = await execute_find_files(step, ctx)
    assert result == []


@pytest.mark.asyncio
async def test_find_files_nonexistent_dir():
    step = FindFilesStep(
        kind="find_files",
        name="files",
        arguments="input.dir",
        pattern="*.txt",
    )
    ctx = PipelineContext({"dir": "/nonexistent/path"})
    with pytest.raises(StepExecutionError, match="Directory not found"):
        await execute_find_files(step, ctx)


@pytest.mark.asyncio
async def test_result_has_name_and_path(populated_dir):
    step = FindFilesStep(
        kind="find_files",
        name="files",
        arguments="input.dir",
        pattern="*.txt",
    )
    ctx = PipelineContext({"dir": str(populated_dir)})
    result = await execute_find_files(step, ctx)
    for item in result:
        assert "name" in item
        assert "path" in item
        assert item["path"].endswith(item["name"])
