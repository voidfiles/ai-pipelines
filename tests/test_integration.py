"""End-to-end integration tests loading real YAML pipelines.

These tests exercise the full flow: YAML load -> validate -> execute,
using real filesystem operations and no LLM calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_pipelines import load_pipeline, run_pipeline

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.asyncio
async def test_simple_pipeline_from_yaml(tmp_path):
    """Load simple_pipeline.yaml, create files, run it end-to-end."""
    (tmp_path / "hello.txt").write_text("Hello, world!")
    (tmp_path / "goodbye.txt").write_text("Goodbye, world!")

    defn = load_pipeline(FIXTURES / "simple_pipeline.yaml")
    result = await run_pipeline(defn, {"dir": str(tmp_path)})

    assert result.output["file_count"] == 2
    assert len(result.output["contents"]) == 2
    assert "Hello, world!" in result.output["contents"]
    assert "Goodbye, world!" in result.output["contents"]


@pytest.mark.asyncio
async def test_simple_pipeline_no_files(tmp_path):
    """Pipeline with no matching files should produce empty results."""
    defn = load_pipeline(FIXTURES / "simple_pipeline.yaml")
    result = await run_pipeline(defn, {"dir": str(tmp_path)})

    assert result.output["file_count"] == 0
    assert result.output["contents"] == []


@pytest.mark.asyncio
async def test_nested_pipeline_from_yaml():
    """Load nested_pipeline.yaml with inline documents and nested for_each."""
    defn = load_pipeline(FIXTURES / "nested_pipeline.yaml")

    input_data = {
        "documents": [
            {
                "title": "Doc A",
                "content": "a" * 120,  # Will create multiple chunks at chunk_size=50
            },
            {
                "title": "Doc B",
                "content": "b" * 60,
            },
        ]
    }

    result = await run_pipeline(defn, input_data)

    # Result should be a flat list of chunk info dicts from all documents
    all_chunks = result.output
    assert isinstance(all_chunks, list)
    assert len(all_chunks) > 0

    # Each chunk should have title from parent scope
    for chunk in all_chunks:
        assert "title" in chunk
        assert "index" in chunk
        assert "length" in chunk

    # Check we have chunks from both documents
    titles = {c["title"] for c in all_chunks}
    assert titles == {"Doc A", "Doc B"}


@pytest.mark.asyncio
async def test_pipeline_result_has_step_results():
    """Step results should be populated with timing info."""
    defn = load_pipeline(FIXTURES / "simple_pipeline.yaml")
    tmp = Path("/tmp/ai_pipelines_test_step_results")
    tmp.mkdir(exist_ok=True)
    (tmp / "test.txt").write_text("content")

    try:
        result = await run_pipeline(defn, {"dir": str(tmp)})
        assert len(result.step_results) == 3  # find_files, for_each, transform
        for sr in result.step_results:
            assert sr.step_name
            assert sr.duration_ms >= 0
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.asyncio
async def test_logging_writes_to_disk(tmp_path):
    """When logging is configured, pipeline.log should contain events."""
    from ai_pipelines import configure_logging

    log_dir = tmp_path / "logs"
    configure_logging(log_dir)

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("hi")

    defn = load_pipeline(FIXTURES / "simple_pipeline.yaml")
    await run_pipeline(defn, {"dir": str(tmp_path / "data")})

    log_file = log_dir / "pipeline.log"
    assert log_file.exists()

    import json

    lines = log_file.read_text().strip().split("\n")
    events = [json.loads(line) for line in lines]
    event_types = [e["event"] for e in events]
    assert "step_start" in event_types
    assert "step_complete" in event_types
