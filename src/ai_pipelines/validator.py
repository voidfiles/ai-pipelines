"""Pre-flight pipeline validator.

Statically validates a pipeline definition without executing it.
Catches broken references, bad expressions, template issues, and
structural problems before any LLM tokens are burned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import jinja2
from jinja2 import nodes as jinja_nodes
from jsonata import Jsonata
from jsonata.jexception import JException

from ai_pipelines.models import (
    EvaluateStep,
    ForEachStep,
    PipelineDefinition,
    PromptStep,
    Step,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Diagnostic:
    """A single validation finding."""

    severity: Severity
    step_name: str
    message: str
    field: str  # "name", "arguments", "template"


@dataclass(frozen=True)
class ValidationResult:
    """Aggregate result of pipeline validation."""

    diagnostics: list[Diagnostic]

    @property
    def ok(self) -> bool:
        """True when there are no error-severity diagnostics."""
        return not any(d.severity == Severity.ERROR for d in self.diagnostics)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.WARNING]


# ---------------------------------------------------------------------------
# Reserved names that steps must not shadow
# ---------------------------------------------------------------------------

RESERVED_NAMES: frozenset[str] = frozenset({"input", "item", "item_index"})

# ---------------------------------------------------------------------------
# Jinja2 environment for template parsing (not rendering)
# ---------------------------------------------------------------------------

_JINJA_ENV = jinja2.Environment()

# ---------------------------------------------------------------------------
# 1. Name uniqueness
# ---------------------------------------------------------------------------


def _check_name_uniqueness(
    steps: list[Step],
    scope_path: str = "",
) -> list[Diagnostic]:
    """Detect duplicate step names and reserved name usage at each scope."""
    diagnostics: list[Diagnostic] = []
    seen: dict[str, int] = {}

    for i, step in enumerate(steps):
        if step.name in RESERVED_NAMES:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    step_name=step.name,
                    message=f"Step name '{step.name}' is reserved",
                    field="name",
                )
            )

        if step.name in seen:
            scope_label = scope_path or "top level"
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    step_name=step.name,
                    message=(
                        f"Duplicate step name '{step.name}' at {scope_label} "
                        f"(first at index {seen[step.name]})"
                    ),
                    field="name",
                )
            )
        else:
            seen[step.name] = i

        if isinstance(step, ForEachStep):
            nested_path = f"{scope_path}/{step.name}" if scope_path else step.name
            diagnostics.extend(_check_name_uniqueness(step.steps, nested_path))

    return diagnostics


# ---------------------------------------------------------------------------
# 2. JSONata expression parsing
# ---------------------------------------------------------------------------


def _parse_jsonata(
    expression: str,
    step_name: str,
    field: str,
) -> tuple[Any | None, list[Diagnostic]]:
    """Try to parse a JSONata expression. Return (ast, diagnostics)."""
    try:
        parsed = Jsonata(expression)
        return parsed.ast, []
    except (JException, Exception) as e:
        return None, [
            Diagnostic(
                severity=Severity.ERROR,
                step_name=step_name,
                message=f"Invalid JSONata expression: {e}",
                field=field,
            )
        ]


# ---------------------------------------------------------------------------
# 3. Root reference extraction from JSONata AST
# ---------------------------------------------------------------------------


def _extract_root_references(node: Any | None) -> set[str]:
    """Walk a JSONata AST and collect root-level context names referenced.

    For path expressions like ``input.name`` this extracts ``{"input"}``.
    For function calls like ``$count(files)`` it recurses into arguments.
    JSONata variables (``$append``, ``$count``) are builtins, not context refs.

    References inside array filter expressions (``items[verified!=false]``)
    are relative to the array element, not the root context, so we skip
    the ``stages`` children on path steps.
    """
    if node is None:
        return set()

    refs: set[str] = set()
    node_type = getattr(node, "type", None)

    if node_type == "path":
        steps = getattr(node, "steps", None) or []
        if steps:
            first = steps[0]
            first_type = getattr(first, "type", None)
            if first_type == "name":
                refs.add(str(first.value))
            else:
                refs.update(_extract_root_references(first))
        # Do NOT recurse into stages (filter predicates reference
        # array element fields, not root context names)

    elif node_type == "binary":
        refs.update(_extract_root_references(getattr(node, "lhs", None)))
        refs.update(_extract_root_references(getattr(node, "rhs", None)))

    elif node_type == "unary":
        lhs_object = getattr(node, "lhs_object", None)
        if lhs_object:
            for pair in lhs_object:
                refs.update(_extract_root_references(pair[0]))
                refs.update(_extract_root_references(pair[1]))
        expressions = getattr(node, "expressions", None)
        if expressions:
            for expr in expressions:
                refs.update(_extract_root_references(expr))

    elif node_type == "function":
        for arg in getattr(node, "arguments", None) or []:
            refs.update(_extract_root_references(arg))

    elif node_type == "condition":
        refs.update(_extract_root_references(getattr(node, "condition", None)))
        refs.update(_extract_root_references(getattr(node, "then", None)))
        refs.update(_extract_root_references(getattr(node, "_else", None)))

    elif node_type == "block":
        for expr in getattr(node, "expressions", None) or []:
            refs.update(_extract_root_references(expr))

    elif node_type == "bind":
        refs.update(_extract_root_references(getattr(node, "rhs", None)))

    elif node_type == "apply":
        refs.update(_extract_root_references(getattr(node, "lhs", None)))
        refs.update(_extract_root_references(getattr(node, "rhs", None)))

    # Leaves (name, variable, string, number, value, regex, wildcard,
    # descendant) don't produce root references on their own.

    return refs


# ---------------------------------------------------------------------------
# 4. Reference resolution
# ---------------------------------------------------------------------------


def _check_references(
    steps: list[Step],
    parent_available: set[str],
    scope_path: str = "",
) -> list[Diagnostic]:
    """Verify that every JSONata expression only references available names."""
    diagnostics: list[Diagnostic] = []
    available = set(parent_available)

    for step in steps:
        arguments_expr = getattr(step, "arguments", None)
        if arguments_expr is not None:
            ast, parse_diags = _parse_jsonata(arguments_expr, step.name, "arguments")
            diagnostics.extend(parse_diags)

            if ast is not None:
                refs = _extract_root_references(ast)
                for ref in sorted(refs):
                    if ref not in available:
                        diagnostics.append(
                            Diagnostic(
                                severity=Severity.ERROR,
                                step_name=step.name,
                                message=(
                                    f"Reference '{ref}' is not available. "
                                    f"Available names: {sorted(available)}"
                                ),
                                field="arguments",
                            )
                        )

        # Prompt-specific checks
        if isinstance(step, PromptStep):
            diagnostics.extend(_check_template(step))
            diagnostics.extend(_cross_check_template_args(step))

        # Evaluate-specific checks
        if isinstance(step, EvaluateStep):
            diagnostics.extend(_check_evaluate_arguments(step))

        # for_each type hint
        if isinstance(step, ForEachStep):
            diagnostics.extend(
                _check_for_each_target_type(step, available, steps)
            )
            inner_available = available | {"item", "item_index"}
            nested_path = f"{scope_path}/{step.name}" if scope_path else step.name
            diagnostics.extend(
                _check_references(step.steps, inner_available, nested_path)
            )

        available.add(step.name)

    return diagnostics


# ---------------------------------------------------------------------------
# 5. Jinja2 template syntax
# ---------------------------------------------------------------------------


def _check_template(step: PromptStep) -> list[Diagnostic]:
    """Parse a prompt step's Jinja2 template for syntax errors."""
    try:
        _JINJA_ENV.parse(step.template)
        return []
    except jinja2.TemplateSyntaxError as e:
        return [
            Diagnostic(
                severity=Severity.ERROR,
                step_name=step.name,
                message=f"Invalid Jinja2 template: {e}",
                field="template",
            )
        ]


# ---------------------------------------------------------------------------
# 6. Template / arguments cross-check
# ---------------------------------------------------------------------------


def _extract_jsonata_object_keys(expression: str) -> set[str] | None:
    """If the expression is a JSONata object literal, return its keys.

    Returns None if parsing fails or the expression isn't an object literal.
    """
    try:
        parsed = Jsonata(expression)
        ast = parsed.ast
    except (JException, Exception):
        return None

    if (
        getattr(ast, "type", None) == "unary"
        and str(getattr(ast, "value", "")) == "{"
        and getattr(ast, "lhs_object", None)
    ):
        return {str(pair[0].value) for pair in ast.lhs_object}
    return None


def _extract_template_arg_keys(template: str) -> set[str]:
    """Extract all ``args.KEY`` references from a Jinja2 template.

    Walks the Jinja2 AST for ``Getattr`` nodes whose target is the
    ``args`` name.
    """
    try:
        ast = _JINJA_ENV.parse(template)
    except jinja2.TemplateSyntaxError:
        return set()

    keys: set[str] = set()
    for node in ast.find_all(jinja_nodes.Getattr):
        if isinstance(node.node, jinja_nodes.Name) and node.node.name == "args":
            keys.add(node.attr)
    return keys


def _cross_check_template_args(step: PromptStep) -> list[Diagnostic]:
    """Compare a prompt's arguments keys with its template's args.* refs."""
    if not step.arguments:
        return []

    obj_keys = _extract_jsonata_object_keys(step.arguments)
    if obj_keys is None:
        return []

    template_keys = _extract_template_arg_keys(step.template)
    diagnostics: list[Diagnostic] = []

    missing = template_keys - obj_keys
    for key in sorted(missing):
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                step_name=step.name,
                message=(
                    f"Template references 'args.{key}' but arguments "
                    f"does not produce key '{key}'. "
                    f"Arguments keys: {sorted(obj_keys)}"
                ),
                field="template",
            )
        )

    unused = obj_keys - template_keys
    for key in sorted(unused):
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                step_name=step.name,
                message=(
                    f"Arguments produces key '{key}' but template "
                    f"never references 'args.{key}'"
                ),
                field="arguments",
            )
        )

    return diagnostics


# ---------------------------------------------------------------------------
# 7. for_each target type hint
# ---------------------------------------------------------------------------


def _check_for_each_target_type(
    step: ForEachStep,
    available: set[str],
    sibling_steps: list[Step],
) -> list[Diagnostic]:
    """Warn if for_each iterates over a step known to produce a non-list."""
    try:
        parsed = Jsonata(step.arguments)
        ast = parsed.ast
    except (JException, Exception):
        return []

    if getattr(ast, "type", None) != "path":
        return []

    steps = getattr(ast, "steps", None) or []
    if not steps:
        return []

    first = steps[0]
    if getattr(first, "type", None) != "name":
        return []

    ref_name = str(first.value)

    # Only flag bare references (e.g., "my_file") not sub-paths
    # like "my_file.lines" which might be fine
    if len(steps) > 1:
        return []

    step_kinds: dict[str, str] = {}
    for s in sibling_steps:
        step_kinds[s.name] = s.kind
        if s.name == step.name:
            break

    ref_kind = step_kinds.get(ref_name)
    if ref_kind == "read_file":
        return [
            Diagnostic(
                severity=Severity.WARNING,
                step_name=step.name,
                message=(
                    f"for_each iterates over '{ref_name}' which is a "
                    f"read_file step (produces a string). Iterating a "
                    f"string yields individual characters."
                ),
                field="arguments",
            )
        ]

    return []


# ---------------------------------------------------------------------------
# 8. Evaluate step arguments check
# ---------------------------------------------------------------------------

_EVALUATE_REQUIRED_KEYS: dict[str, set[str]] = {
    "summarization": {"source", "summary"},
    "faithfulness": {"source", "response"},
    "hallucination": {"context", "response"},
    "context_relevance": {"question", "context"},
    "context_utilization": {"question", "context", "response"},
    "factual_accuracy": {"question", "context", "response"},
    "context_conciseness": {"question", "context", "concise_context"},
}


def _check_evaluate_arguments(step: EvaluateStep) -> list[Diagnostic]:
    """Verify that evaluate step arguments produce the required keys."""
    obj_keys = _extract_jsonata_object_keys(step.arguments)
    if obj_keys is None:
        return []

    required = _EVALUATE_REQUIRED_KEYS.get(step.strategy, set())
    diagnostics: list[Diagnostic] = []

    missing = required - obj_keys
    for key in sorted(missing):
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                step_name=step.name,
                message=(
                    f"Strategy '{step.strategy}' requires key '{key}' "
                    f"in arguments. Got keys: {sorted(obj_keys)}"
                ),
                field="arguments",
            )
        )

    expected_all = {
        "source", "summary", "response", "context",
        "question", "concise_context",
    }
    extra = obj_keys - required
    unexpected = extra & expected_all  # only warn on keys meant for wrong strategy
    for key in sorted(extra - unexpected):
        pass  # genuinely unknown keys are fine, user might have reasons
    for key in sorted(unexpected):
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                step_name=step.name,
                message=(
                    f"Strategy '{step.strategy}' does not use key '{key}'"
                ),
                field="arguments",
            )
        )

    return diagnostics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_pipeline(definition: PipelineDefinition) -> ValidationResult:
    """Statically validate a pipeline definition without executing it.

    Checks:
    - Step name uniqueness and reserved name avoidance
    - JSONata expression syntax
    - Jinja2 template syntax
    - Reference resolution (all referenced names must be available)
    - Template/arguments key cross-check
    - for_each target type hints

    Returns a ``ValidationResult``. The pipeline is considered valid
    when ``result.ok`` is True (no error-severity diagnostics).
    """
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_check_name_uniqueness(definition.steps))
    diagnostics.extend(_check_references(definition.steps, {"input"}))
    return ValidationResult(diagnostics=diagnostics)


def load_and_validate_pipeline(
    path: str | Path,
) -> tuple[PipelineDefinition, ValidationResult]:
    """Load a pipeline from YAML and validate it.

    Convenience wrapper: calls ``load_pipeline`` then ``validate_pipeline``.
    Raises ``PipelineLoadError`` if YAML/Pydantic parsing fails.
    """
    from ai_pipelines.loader import load_pipeline

    definition = load_pipeline(path)
    result = validate_pipeline(definition)
    return definition, result
