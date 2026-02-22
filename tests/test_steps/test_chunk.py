"""Tests for chunk step executor."""

from __future__ import annotations

import pytest

from ai_pipelines.context import PipelineContext
from ai_pipelines.models import ChunkStep
from ai_pipelines.steps.chunk import execute_chunk, split_text_chunks


# ── Unit tests for split_text_chunks ──────────────────────────────


def test_split_short_text_single_chunk():
    chunks = split_text_chunks("Hello world", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello world"
    assert chunks[0]["index"] == 0


def test_split_exact_boundary():
    text = "a" * 100
    chunks = split_text_chunks(text, chunk_size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_split_with_overlap():
    text = "a" * 200
    chunks = split_text_chunks(text, chunk_size=100, overlap=20)
    assert len(chunks) == 3
    # First chunk: 0-100
    assert len(chunks[0]["text"]) == 100
    # Second chunk starts at 80 (100 - 20 overlap)
    assert len(chunks[1]["text"]) == 100
    # Third chunk starts at 160 (80 + 100 - 20)
    assert len(chunks[2]["text"]) == 40


def test_split_preserves_indices():
    text = "a" * 300
    chunks = split_text_chunks(text, chunk_size=100, overlap=0)
    for i, chunk in enumerate(chunks):
        assert chunk["index"] == i


def test_split_no_overlap():
    text = "a" * 250
    chunks = split_text_chunks(text, chunk_size=100, overlap=0)
    assert len(chunks) == 3
    assert len(chunks[0]["text"]) == 100
    assert len(chunks[1]["text"]) == 100
    assert len(chunks[2]["text"]) == 50


def test_split_empty_text():
    chunks = split_text_chunks("", chunk_size=100, overlap=10)
    assert chunks == []


def test_overlap_larger_than_chunk_clamped():
    """Overlap >= chunk_size should not cause infinite loops."""
    text = "a" * 200
    chunks = split_text_chunks(text, chunk_size=100, overlap=100)
    # Should still make progress (overlap clamped to chunk_size - 1)
    assert len(chunks) > 1
    total_unique = sum(1 for _ in chunks)
    assert total_unique > 0


# ── Integration with step executor ────────────────────────────────


@pytest.mark.asyncio
async def test_chunk_step():
    step = ChunkStep(
        kind="chunk",
        name="chunks",
        arguments="input.content",
        chunk_size=50,
        overlap=10,
    )
    text = "word " * 40  # 200 chars
    ctx = PipelineContext({"content": text})
    result = await execute_chunk(step, ctx)
    assert "chunks" in result
    assert len(result["chunks"]) > 1
    for chunk in result["chunks"]:
        assert "text" in chunk
        assert "index" in chunk
        assert isinstance(chunk["index"], int)


@pytest.mark.asyncio
async def test_chunk_step_default_size():
    step = ChunkStep(
        kind="chunk",
        name="chunks",
        arguments="input.content",
    )
    text = "x" * 100
    ctx = PipelineContext({"content": text})
    result = await execute_chunk(step, ctx)
    # 100 chars < 4000 default chunk_size, so single chunk
    assert len(result["chunks"]) == 1
