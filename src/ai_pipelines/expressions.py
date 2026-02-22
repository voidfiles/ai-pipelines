"""JSONata expression evaluator for pipeline data access.

Thin wrapper over jsonata-python that provides the expression language
used throughout pipeline YAML files for referencing step results,
filtering arrays, constructing objects, and projecting data.
"""

from __future__ import annotations

from typing import Any

import jsonata

from ai_pipelines.errors import ExpressionError


def evaluate(expression: str, context: dict[str, Any]) -> Any:
    """Evaluate a JSONata expression against a pipeline context dict.

    Supports all JSONata syntax: dot access, array indexing, filtering,
    object construction, projection, built-in functions, and more.

    Args:
        expression: JSONata expression string (e.g., "input.transcript_dir",
            "claims[verified != false]", '{"key": item.name}').
        context: Dict mapping step names and "input" to their result values.

    Returns:
        The resolved value. Returns None for paths that don't exist
        (JSONata's undefined behavior).

    Raises:
        ExpressionError: If the expression syntax is invalid.
    """
    try:
        expr = jsonata.Jsonata(expression)
        return expr.evaluate(context)
    except Exception as e:
        raise ExpressionError(
            f"Expression '{expression}' failed: {e}"
        ) from e
