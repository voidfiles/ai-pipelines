"""Custom exception hierarchy for ai-pipelines.

All exceptions inherit from PipelineError so callers can catch broadly
or narrowly as needed.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base for all ai-pipelines errors."""


class PipelineLoadError(PipelineError):
    """YAML parsing or pipeline structure validation failed."""


class ExpressionError(PipelineError):
    """JSONata expression evaluation failed."""


class StepExecutionError(PipelineError):
    """A step failed during execution."""

    def __init__(
        self,
        step_name: str,
        message: str,
        cause: Exception | None = None,
    ) -> None:
        self.step_name = step_name
        self.cause = cause
        super().__init__(f"Step '{step_name}' failed: {message}")


class ValidationError(PipelineError):
    """Input or output validation against JSON Schema failed."""


class LLMError(PipelineError):
    """LLM call failed (network, structured output parse, etc.)."""
