"""Tests for Pydantic pipeline models.

Validates that YAML-like dicts parse into correct step types via
the discriminated union on the ``kind`` field, and that invalid
data is properly rejected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_pipelines.models import (
    ChunkStep,
    FindFilesStep,
    ForEachStep,
    PipelineDefinition,
    PipelineResult,
    PromptStep,
    ReadFileStep,
    StepResult,
    TransformStep,
)


# ── Basic step parsing ───────────────────────────────────────────


def test_find_files_step():
    raw = {
        "kind": "find_files",
        "name": "transcripts",
        "arguments": "input.transcript_dir",
        "pattern": "*.md",
    }
    step = FindFilesStep.model_validate(raw)
    assert step.kind == "find_files"
    assert step.pattern == "*.md"
    assert step.arguments == "input.transcript_dir"


def test_read_file_step():
    raw = {
        "kind": "read_file",
        "name": "content",
        "arguments": "item.path",
    }
    step = ReadFileStep.model_validate(raw)
    assert step.kind == "read_file"


def test_transform_step():
    raw = {
        "kind": "transform",
        "name": "transcript",
        "arguments": '{"filename": item.name, "title": item.name}',
    }
    step = TransformStep.model_validate(raw)
    assert step.kind == "transform"


def test_chunk_step_defaults():
    raw = {
        "kind": "chunk",
        "name": "chunks",
        "arguments": "item.content",
    }
    step = ChunkStep.model_validate(raw)
    assert step.chunk_size == 4000
    assert step.overlap == 200


def test_chunk_step_custom():
    raw = {
        "kind": "chunk",
        "name": "chunks",
        "arguments": "item.content",
        "chunk_size": 2000,
        "overlap": 300,
    }
    step = ChunkStep.model_validate(raw)
    assert step.chunk_size == 2000
    assert step.overlap == 300


def test_prompt_step():
    raw = {
        "kind": "prompt",
        "name": "extraction",
        "arguments": '{"chunk_text": item.text}',
        "model": "haiku",
        "system_prompt": "You are an expert extractor.",
        "template": "Extract from: {{ args.chunk_text }}",
        "output": {
            "type": "object",
            "properties": {"claims": {"type": "array"}},
            "required": ["claims"],
        },
    }
    step = PromptStep.model_validate(raw)
    assert step.model == "haiku"
    assert step.output is not None


def test_prompt_step_defaults():
    raw = {
        "kind": "prompt",
        "name": "simple",
        "template": "Hello",
    }
    step = PromptStep.model_validate(raw)
    assert step.model == "sonnet"
    assert step.output is None
    assert step.system_prompt is None
    assert step.arguments is None


def test_for_each_step():
    raw = {
        "kind": "for_each",
        "name": "chunk_results",
        "arguments": "chunks.chunks",
        "steps": [
            {
                "kind": "transform",
                "name": "inner",
                "arguments": "item.text",
            }
        ],
    }
    step = ForEachStep.model_validate(raw)
    assert step.kind == "for_each"
    assert len(step.steps) == 1
    assert isinstance(step.steps[0], TransformStep)


def test_pipeline_kind_parses_as_for_each():
    """YAML `kind: pipeline` is an alias for `for_each`."""
    raw = {
        "kind": "pipeline",
        "name": "sessions",
        "arguments": "transcripts.files",
        "steps": [
            {
                "kind": "transform",
                "name": "inner",
                "arguments": "item",
            }
        ],
    }
    step = ForEachStep.model_validate(raw)
    assert step.kind == "pipeline"
    assert len(step.steps) == 1


# ── Pipeline definition ──────────────────────────────────────────


def test_pipeline_definition():
    raw = {
        "input": {
            "type": "object",
            "properties": {"dir": {"type": "string"}},
            "required": ["dir"],
        },
        "steps": [
            {
                "kind": "find_files",
                "name": "files",
                "arguments": "input.dir",
                "pattern": "*.txt",
            },
            {
                "kind": "transform",
                "name": "count",
                "arguments": "$count(files)",
            },
        ],
    }
    defn = PipelineDefinition.model_validate(raw)
    assert len(defn.steps) == 2
    assert isinstance(defn.steps[0], FindFilesStep)
    assert isinstance(defn.steps[1], TransformStep)


def test_pipeline_definition_with_nested_for_each():
    raw = {
        "input": {"type": "object", "properties": {}},
        "steps": [
            {
                "kind": "for_each",
                "name": "loop",
                "arguments": "input.items",
                "steps": [
                    {
                        "kind": "prompt",
                        "name": "inner_prompt",
                        "template": "Process: {{ args.text }}",
                        "arguments": '{"text": item}',
                    }
                ],
            }
        ],
    }
    defn = PipelineDefinition.model_validate(raw)
    loop = defn.steps[0]
    assert isinstance(loop, ForEachStep)
    assert isinstance(loop.steps[0], PromptStep)


# ── Deeply nested (for_each inside for_each) ─────────────────────


def test_deeply_nested_for_each():
    raw = {
        "input": {"type": "object", "properties": {}},
        "steps": [
            {
                "kind": "for_each",
                "name": "outer",
                "arguments": "input.items",
                "steps": [
                    {
                        "kind": "for_each",
                        "name": "inner",
                        "arguments": "item.sub_items",
                        "steps": [
                            {
                                "kind": "transform",
                                "name": "leaf",
                                "arguments": "item",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    defn = PipelineDefinition.model_validate(raw)
    outer = defn.steps[0]
    assert isinstance(outer, ForEachStep)
    inner = outer.steps[0]
    assert isinstance(inner, ForEachStep)
    assert isinstance(inner.steps[0], TransformStep)


# ── Validation errors ─────────────────────────────────────────────


def test_unknown_kind_raises():
    raw = {
        "input": {"type": "object"},
        "steps": [
            {"kind": "nonexistent", "name": "bad", "arguments": "x"}
        ],
    }
    with pytest.raises(ValidationError):
        PipelineDefinition.model_validate(raw)


def test_missing_name_raises():
    raw = {
        "input": {"type": "object"},
        "steps": [
            {"kind": "transform", "arguments": "x"}
        ],
    }
    with pytest.raises(ValidationError):
        PipelineDefinition.model_validate(raw)


def test_prompt_step_requires_template():
    with pytest.raises(ValidationError):
        PromptStep.model_validate({
            "kind": "prompt",
            "name": "bad",
            "model": "haiku",
        })


def test_find_files_requires_pattern():
    with pytest.raises(ValidationError):
        FindFilesStep.model_validate({
            "kind": "find_files",
            "name": "bad",
            "arguments": "input.dir",
        })


# ── Result models ────────────────────────────────────────────────


def test_step_result():
    sr = StepResult(step_name="s1", value=[1, 2, 3], duration_ms=100.5)
    assert sr.step_name == "s1"
    assert sr.cost_usd is None


def test_pipeline_result():
    pr = PipelineResult(
        output={"key": "value"},
        step_results=[],
        total_duration_ms=500.0,
        total_cost_usd=0.01,
    )
    assert pr.total_cost_usd == 0.01
