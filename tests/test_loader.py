"""Tests for YAML pipeline loading and JSON Schema validation."""

from __future__ import annotations

import pytest

from ai_pipelines.errors import PipelineLoadError, ValidationError
from ai_pipelines.loader import load_pipeline, validate_input, validate_output
from ai_pipelines.models import FindFilesStep, ForEachStep, TransformStep


def test_load_simple_pipeline(tmp_path):
    yaml_content = """\
input:
  type: object
  properties:
    dir:
      type: string
  required:
    - dir
steps:
  - kind: find_files
    name: files
    arguments: input.dir
    pattern: "*.txt"
  - kind: transform
    name: count
    arguments: $count(files)
"""
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml_content)
    defn = load_pipeline(path)
    assert len(defn.steps) == 2
    assert isinstance(defn.steps[0], FindFilesStep)
    assert isinstance(defn.steps[1], TransformStep)


def test_load_pipeline_with_for_each(tmp_path):
    yaml_content = """\
input:
  type: object
  properties:
    items:
      type: array
steps:
  - kind: for_each
    name: results
    arguments: input.items
    steps:
      - kind: transform
        name: inner
        arguments: item
"""
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml_content)
    defn = load_pipeline(path)
    assert isinstance(defn.steps[0], ForEachStep)


def test_load_pipeline_kind_pipeline(tmp_path):
    """kind: pipeline should load as ForEachStep."""
    yaml_content = """\
input:
  type: object
  properties: {}
steps:
  - kind: pipeline
    name: sessions
    arguments: input.files
    steps:
      - kind: transform
        name: mapped
        arguments: item
"""
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml_content)
    defn = load_pipeline(path)
    assert isinstance(defn.steps[0], ForEachStep)


def test_load_nonexistent_file():
    with pytest.raises(PipelineLoadError, match="not found"):
        load_pipeline("/nonexistent/pipeline.yaml")


def test_load_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("{{invalid yaml}}")
    with pytest.raises(PipelineLoadError):
        load_pipeline(path)


def test_load_invalid_structure(tmp_path):
    path = tmp_path / "bad_structure.yaml"
    path.write_text("steps:\n  - kind: nonexistent\n    name: bad\n    arguments: x\n")
    with pytest.raises(PipelineLoadError):
        load_pipeline(path)


# ── Input validation ──────────────────────────────────────────────


def test_validate_input_passes():
    schema = {
        "type": "object",
        "properties": {"dir": {"type": "string"}},
        "required": ["dir"],
    }
    validate_input(schema, {"dir": "/data"})


def test_validate_input_missing_required():
    schema = {
        "type": "object",
        "properties": {"dir": {"type": "string"}},
        "required": ["dir"],
    }
    with pytest.raises(ValidationError, match="Input validation failed"):
        validate_input(schema, {})


def test_validate_input_wrong_type():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
    }
    with pytest.raises(ValidationError):
        validate_input(schema, {"count": "not an int"})


# ── Output validation ────────────────────────────────────────────


def test_validate_output_passes():
    schema = {
        "type": "object",
        "properties": {"claims": {"type": "array"}},
        "required": ["claims"],
    }
    validate_output(schema, {"claims": ["a", "b"]})


def test_validate_output_fails():
    schema = {
        "type": "object",
        "properties": {"claims": {"type": "array"}},
        "required": ["claims"],
    }
    with pytest.raises(ValidationError, match="Output validation failed"):
        validate_output(schema, {"wrong": "data"})
