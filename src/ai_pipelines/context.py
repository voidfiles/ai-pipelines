"""Pipeline execution context.

Accumulates named step results during pipeline execution. Supports
child contexts for nested loops (for_each) where inner step results
don't leak back into the parent scope.
"""

from __future__ import annotations

from typing import Any

from ai_pipelines.errors import ExpressionError


class PipelineContext:
    """Scoped dict that accumulates step results.

    This is the one class in the system. It's a data container with
    child-scoping for loop isolation, not a logic holder.
    """

    def __init__(self, pipeline_input: dict[str, Any]) -> None:
        self._data: dict[str, Any] = {"input": pipeline_input}

    def set_result(self, step_name: str, value: Any) -> None:
        """Store a step's result. Raises on duplicate names."""
        if step_name in self._data:
            raise ExpressionError(f"Duplicate step name: '{step_name}'")
        self._data[step_name] = value

    def get_data(self) -> dict[str, Any]:
        """Return the full context dict for expression evaluation."""
        return self._data

    def child(self, extra: dict[str, Any] | None = None) -> PipelineContext:
        """Create a child context for nested execution.

        The child sees all parent data plus any extra bindings
        (like the current loop item). Results set on the child
        do NOT leak back into the parent.
        """
        child_data = {**self._data}
        if extra:
            child_data.update(extra)
        ctx = PipelineContext.__new__(PipelineContext)
        ctx._data = child_data
        return ctx
