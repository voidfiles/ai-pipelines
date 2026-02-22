"""YAML pipeline loading and JSON Schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import yaml
from pydantic import ValidationError as PydanticValidationError

from ai_pipelines.errors import PipelineLoadError, ValidationError
from ai_pipelines.models import PipelineDefinition


def load_pipeline(path: str | Path) -> PipelineDefinition:
    """Load a pipeline definition from a YAML file.

    Parses YAML, then validates the structure via Pydantic.

    Args:
        path: Path to the YAML pipeline file.

    Returns:
        Validated PipelineDefinition.

    Raises:
        PipelineLoadError: If the file doesn't exist, YAML is invalid,
            or the structure doesn't match the expected schema.
    """
    path = Path(path)
    if not path.is_file():
        raise PipelineLoadError(f"Pipeline file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise PipelineLoadError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise PipelineLoadError(
            f"Pipeline YAML must be a mapping, got {type(raw).__name__}"
        )

    try:
        return PipelineDefinition.model_validate(raw)
    except PydanticValidationError as e:
        raise PipelineLoadError(
            f"Pipeline structure invalid: {e}"
        ) from e


def validate_input(schema: dict[str, Any], data: dict[str, Any]) -> None:
    """Validate pipeline input data against the pipeline's input JSON Schema.

    Raises:
        ValidationError: If data doesn't match schema.
    """
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValidationError(f"Input validation failed: {e.message}") from e


def validate_output(schema: dict[str, Any], data: Any) -> None:
    """Validate step output against a JSON Schema.

    Raises:
        ValidationError: If data doesn't match schema.
    """
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValidationError(f"Output validation failed: {e.message}") from e
