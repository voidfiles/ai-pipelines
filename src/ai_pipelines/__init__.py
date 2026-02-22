"""ai-pipelines: YAML-driven AI/LLM pipeline executor."""

# Patch the SDK before anything imports query(). See sdk_patch.py for why.
from ai_pipelines.sdk_patch import apply as _apply_sdk_patch

_apply_sdk_patch()

from ai_pipelines.errors import (
    ExpressionError,
    LLMError,
    PipelineError,
    PipelineLoadError,
    StepExecutionError,
    ValidationError,
)
from ai_pipelines.executor import run_pipeline
from ai_pipelines.loader import load_pipeline, validate_input
from ai_pipelines.models import EvaluateStep, PipelineDefinition, PipelineResult, StepResult
from ai_pipelines.pipeline_logger import configure_logging
from ai_pipelines.validator import (
    Diagnostic,
    Severity,
    ValidationResult,
    load_and_validate_pipeline,
    validate_pipeline,
)

__all__ = [
    "configure_logging",
    "Diagnostic",
    "load_and_validate_pipeline",
    "load_pipeline",
    "run_pipeline",
    "Severity",
    "validate_input",
    "validate_pipeline",
    "ValidationResult",
    "EvaluateStep",
    "ExpressionError",
    "LLMError",
    "PipelineDefinition",
    "PipelineError",
    "PipelineLoadError",
    "PipelineResult",
    "StepExecutionError",
    "StepResult",
    "ValidationError",
]
