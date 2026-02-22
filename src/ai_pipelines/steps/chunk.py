"""chunk step: split text into overlapping chunks."""

from __future__ import annotations

from typing import Any

from ai_pipelines.context import PipelineContext
from ai_pipelines.expressions import evaluate
from ai_pipelines.models import ChunkStep


def split_text_chunks(
    text: str, chunk_size: int, overlap: int
) -> list[dict[str, Any]]:
    """Split text into overlapping chunks by character count.

    Args:
        text: Source text to split.
        chunk_size: Max characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of dicts with ``text`` (chunk content) and ``index`` (position).
    """
    if not text:
        return []

    # Clamp overlap so we always make forward progress
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    stride = chunk_size - overlap
    chunks: list[dict[str, Any]] = []
    pos = 0
    index = 0

    while pos < len(text):
        chunk_text = text[pos : pos + chunk_size]
        chunks.append({"text": chunk_text, "index": index})
        pos += stride
        index += 1

    return chunks


async def execute_chunk(
    step: ChunkStep, context: PipelineContext
) -> dict[str, Any]:
    """Split content into chunks.

    Returns ``{"chunks": [{"text": str, "index": int}, ...]}``
    so downstream steps can reference ``chunks.chunks`` and iterate
    with ``item.text``, ``item.index``.
    """
    text = evaluate(step.arguments, context.get_data())
    chunks = split_text_chunks(str(text), step.chunk_size, step.overlap)
    return {"chunks": chunks}
