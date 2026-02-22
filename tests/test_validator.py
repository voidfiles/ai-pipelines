"""Tests for the pre-flight pipeline validator.

Every test constructs real Pydantic models or loads real YAML fixtures.
No mocks, no fakes, just real parsing and real validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_pipelines.loader import load_pipeline
from ai_pipelines.models import (
    ChunkStep,
    EvaluateStep,
    FindFilesStep,
    ForEachStep,
    PipelineDefinition,
    PromptStep,
    ReadFileStep,
    TransformStep,
)
from ai_pipelines.validator import (
    Diagnostic,
    Severity,
    ValidationResult,
    _check_evaluate_arguments,
    _check_for_each_target_type,
    _check_name_uniqueness,
    _check_references,
    _check_template,
    _cross_check_template_args,
    _extract_jsonata_object_keys,
    _extract_root_references,
    _extract_template_arg_keys,
    _parse_jsonata,
    load_and_validate_pipeline,
    validate_pipeline,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
E2E_DIR = Path(__file__).parent.parent / "e2e"


# ─────────────────────────────────────────────────────────────────────────
# ValidationResult
# ─────────────────────────────────────────────────────────────────────────


def test_validation_result_ok_with_no_diagnostics():
    r = ValidationResult(diagnostics=[])
    assert r.ok is True
    assert r.errors == []
    assert r.warnings == []


def test_validation_result_ok_with_only_warnings():
    r = ValidationResult(
        diagnostics=[
            Diagnostic(Severity.WARNING, "s", "unused key", "arguments")
        ]
    )
    assert r.ok is True
    assert len(r.warnings) == 1
    assert len(r.errors) == 0


def test_validation_result_not_ok_with_errors():
    r = ValidationResult(
        diagnostics=[
            Diagnostic(Severity.ERROR, "s", "bad ref", "arguments"),
            Diagnostic(Severity.WARNING, "s", "unused key", "arguments"),
        ]
    )
    assert r.ok is False
    assert len(r.errors) == 1
    assert len(r.warnings) == 1


# ─────────────────────────────────────────────────────────────────────────
# _check_name_uniqueness
# ─────────────────────────────────────────────────────────────────────────


def test_unique_names_no_errors():
    steps = [
        TransformStep(kind="transform", name="a", arguments="input"),
        TransformStep(kind="transform", name="b", arguments="a"),
    ]
    assert _check_name_uniqueness(steps) == []


def test_duplicate_names_at_same_level():
    steps = [
        TransformStep(kind="transform", name="data", arguments="input"),
        TransformStep(kind="transform", name="data", arguments="input"),
    ]
    diags = _check_name_uniqueness(steps)
    assert len(diags) == 1
    assert diags[0].severity == Severity.ERROR
    assert "Duplicate" in diags[0].message
    assert diags[0].step_name == "data"


def test_reserved_name_input():
    steps = [TransformStep(kind="transform", name="input", arguments="input")]
    diags = _check_name_uniqueness(steps)
    assert len(diags) == 1
    assert "reserved" in diags[0].message


def test_reserved_name_item():
    steps = [TransformStep(kind="transform", name="item", arguments="input")]
    diags = _check_name_uniqueness(steps)
    assert len(diags) == 1
    assert "reserved" in diags[0].message


def test_reserved_name_item_index():
    steps = [
        TransformStep(kind="transform", name="item_index", arguments="input")
    ]
    diags = _check_name_uniqueness(steps)
    assert len(diags) == 1
    assert "reserved" in diags[0].message


def test_same_name_in_different_scopes_ok():
    """A nested step can reuse a name from the parent scope."""
    steps = [
        ForEachStep(
            kind="for_each",
            name="loop",
            arguments="input",
            steps=[
                TransformStep(kind="transform", name="data", arguments="item"),
            ],
        ),
        TransformStep(kind="transform", name="data", arguments="loop"),
    ]
    # "data" appears in nested scope and top level, different scopes = OK
    # Only the top-level "data" and the nested "data" are checked independently
    diags = _check_name_uniqueness(steps)
    assert diags == []


def test_duplicate_names_within_nested_scope():
    steps = [
        ForEachStep(
            kind="for_each",
            name="loop",
            arguments="input",
            steps=[
                TransformStep(kind="transform", name="x", arguments="item"),
                TransformStep(kind="transform", name="x", arguments="item"),
            ],
        ),
    ]
    diags = _check_name_uniqueness(steps)
    assert len(diags) == 1
    assert diags[0].step_name == "x"
    assert "loop" in diags[0].message  # scope path mentioned


# ─────────────────────────────────────────────────────────────────────────
# _parse_jsonata
# ─────────────────────────────────────────────────────────────────────────


def test_parse_valid_expression():
    ast, diags = _parse_jsonata("input.name", "step1", "arguments")
    assert ast is not None
    assert diags == []


def test_parse_invalid_expression():
    ast, diags = _parse_jsonata("input..bad..syntax(((", "step1", "arguments")
    assert ast is None
    assert len(diags) == 1
    assert diags[0].severity == Severity.ERROR
    assert diags[0].step_name == "step1"
    assert diags[0].field == "arguments"


# ─────────────────────────────────────────────────────────────────────────
# _extract_root_references
# ─────────────────────────────────────────────────────────────────────────


def _refs(expression: str) -> set[str]:
    """Helper: parse expression and extract root references."""
    from jsonata import Jsonata

    return _extract_root_references(Jsonata(expression).ast)


def test_refs_simple_path():
    assert _refs("input.name") == {"input"}


def test_refs_multi_step_path():
    assert _refs("step1.data.value") == {"step1"}


def test_refs_bare_name():
    assert _refs("files") == {"files"}


def test_refs_function_call():
    assert _refs("$count(files)") == {"files"}


def test_refs_object_construction():
    assert _refs('{"text": item.text, "count": $count(items)}') == {
        "item",
        "items",
    }


def test_refs_binary_expression():
    # Not a typical pipeline expression, but tests coverage
    assert _refs("a + b") == {"a", "b"}


def test_refs_string_concat():
    assert _refs('"prefix" & item.name') == {"item"}


def test_refs_variable_not_context_ref():
    """JSONata $variables are builtins, not context references."""
    assert _refs("$reduce(all_points, $append)") == {"all_points"}


def test_refs_number_literal():
    assert _refs("42") == set()


def test_refs_string_literal():
    assert _refs('"hello"') == set()


def test_refs_mixed():
    assert _refs(
        '{"points": $reduce(all_points, $append), "topics": summaries.topic}'
    ) == {"all_points", "summaries"}


def test_refs_filter_does_not_leak():
    """Filter predicates reference array element fields, not root context."""
    refs = _refs("items[verified != false]")
    assert refs == {"items"}
    # "verified" should NOT appear as a root reference


# ─────────────────────────────────────────────────────────────────────────
# _check_references
# ─────────────────────────────────────────────────────────────────────────


def test_valid_references_no_errors():
    steps = [
        FindFilesStep(
            kind="find_files", name="files", arguments="input.dir", pattern="*.txt"
        ),
        ForEachStep(
            kind="for_each",
            name="contents",
            arguments="files",
            steps=[
                ReadFileStep(kind="read_file", name="content", arguments="item.path"),
            ],
        ),
        TransformStep(
            kind="transform",
            name="summary",
            arguments='{"count": $count(files), "data": contents}',
        ),
    ]
    diags = _check_references(steps, {"input"})
    assert diags == []


def test_forward_reference_error():
    steps = [
        TransformStep(kind="transform", name="first", arguments="second"),
        TransformStep(kind="transform", name="second", arguments="input"),
    ]
    diags = _check_references(steps, {"input"})
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "second" in errors[0].message
    assert errors[0].step_name == "first"


def test_input_always_available():
    steps = [
        TransformStep(kind="transform", name="x", arguments="input.data"),
    ]
    diags = _check_references(steps, {"input"})
    assert diags == []


def test_for_each_scope_has_item():
    steps = [
        TransformStep(kind="transform", name="data", arguments="input"),
        ForEachStep(
            kind="for_each",
            name="loop",
            arguments="data",
            steps=[
                TransformStep(
                    kind="transform", name="inner", arguments="item.text"
                ),
            ],
        ),
    ]
    diags = _check_references(steps, {"input"})
    assert diags == []


def test_for_each_scope_has_item_index():
    steps = [
        ForEachStep(
            kind="for_each",
            name="loop",
            arguments="input",
            steps=[
                TransformStep(
                    kind="transform",
                    name="inner",
                    arguments='{"i": item_index, "val": item}',
                ),
            ],
        ),
    ]
    diags = _check_references(steps, {"input"})
    assert diags == []


def test_nested_for_each_sees_parent_scope():
    steps = [
        TransformStep(kind="transform", name="parent_data", arguments="input"),
        ForEachStep(
            kind="for_each",
            name="outer",
            arguments="input",
            steps=[
                TransformStep(kind="transform", name="mid", arguments="item"),
                ForEachStep(
                    kind="for_each",
                    name="inner",
                    arguments="mid",
                    steps=[
                        TransformStep(
                            kind="transform",
                            name="deep",
                            arguments="parent_data",
                        ),
                    ],
                ),
            ],
        ),
    ]
    diags = _check_references(steps, {"input"})
    assert diags == []


def test_reference_nonexistent_step():
    steps = [
        TransformStep(kind="transform", name="x", arguments="typo_step"),
    ]
    diags = _check_references(steps, {"input"})
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "typo_step" in errors[0].message


def test_nested_step_results_visible_in_order():
    """Steps within a for_each can reference earlier siblings."""
    steps = [
        ForEachStep(
            kind="for_each",
            name="loop",
            arguments="input",
            steps=[
                TransformStep(kind="transform", name="a", arguments="item"),
                TransformStep(kind="transform", name="b", arguments="a"),
            ],
        ),
    ]
    diags = _check_references(steps, {"input"})
    assert diags == []


# ─────────────────────────────────────────────────────────────────────────
# _check_template
# ─────────────────────────────────────────────────────────────────────────


def test_valid_template_no_errors():
    step = PromptStep(
        kind="prompt",
        name="p",
        template="Hello {{ args.name }}, you have {{ args.count }} items.",
    )
    assert _check_template(step) == []


def test_invalid_template_syntax_error():
    step = PromptStep(
        kind="prompt",
        name="p",
        template="Hello {{ args.name }",  # unclosed
    )
    diags = _check_template(step)
    assert len(diags) == 1
    assert diags[0].severity == Severity.ERROR
    assert diags[0].field == "template"


# ─────────────────────────────────────────────────────────────────────────
# _extract_jsonata_object_keys
# ─────────────────────────────────────────────────────────────────────────


def test_extract_object_keys():
    keys = _extract_jsonata_object_keys('{"text": item.text, "count": $count(items)}')
    assert keys == {"text", "count"}


def test_extract_non_object_returns_none():
    assert _extract_jsonata_object_keys("input.data") is None


def test_extract_bare_name_returns_none():
    assert _extract_jsonata_object_keys("files") is None


def test_extract_invalid_returns_none():
    assert _extract_jsonata_object_keys("invalid((((") is None


# ─────────────────────────────────────────────────────────────────────────
# _extract_template_arg_keys
# ─────────────────────────────────────────────────────────────────────────


def test_extract_template_keys():
    keys = _extract_template_arg_keys(
        "{{ args.name }} has {{ args.count }} items"
    )
    assert keys == {"name", "count"}


def test_extract_template_keys_in_loop():
    keys = _extract_template_arg_keys(
        "{% for x in args.items %}{{ x }}{% endfor %}"
    )
    assert keys == {"items"}


def test_extract_template_keys_no_args():
    keys = _extract_template_arg_keys("Hello world, no args here.")
    assert keys == set()


def test_extract_template_keys_bad_syntax():
    keys = _extract_template_arg_keys("{{ broken }")
    assert keys == set()


# ─────────────────────────────────────────────────────────────────────────
# _cross_check_template_args
# ─────────────────────────────────────────────────────────────────────────


def test_cross_check_matching_keys():
    step = PromptStep(
        kind="prompt",
        name="p",
        arguments='{"text": item.text, "count": $count(items)}',
        template="Process {{ args.text }} ({{ args.count }} total)",
    )
    diags = _cross_check_template_args(step)
    assert diags == []


def test_cross_check_missing_key_error():
    step = PromptStep(
        kind="prompt",
        name="p",
        arguments='{"text": item.text}',
        template="Process {{ args.text }} {{ args.missing_key }}",
    )
    diags = _cross_check_template_args(step)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "missing_key" in errors[0].message


def test_cross_check_unused_key_warning():
    step = PromptStep(
        kind="prompt",
        name="p",
        arguments='{"text": item.text, "unused": item.other}',
        template="{{ args.text }}",
    )
    diags = _cross_check_template_args(step)
    warnings = [d for d in diags if d.severity == Severity.WARNING]
    assert len(warnings) == 1
    assert "unused" in warnings[0].message


def test_cross_check_non_object_args_skipped():
    step = PromptStep(
        kind="prompt",
        name="p",
        arguments="input.data",
        template="{{ args.whatever }}",
    )
    diags = _cross_check_template_args(step)
    assert diags == []


def test_cross_check_no_arguments_skipped():
    step = PromptStep(
        kind="prompt",
        name="p",
        template="{{ args.x }}",
    )
    diags = _cross_check_template_args(step)
    assert diags == []


# ─────────────────────────────────────────────────────────────────────────
# _check_for_each_target_type
# ─────────────────────────────────────────────────────────────────────────


def test_for_each_over_find_files_no_warning():
    sibling_steps: list[Step] = [
        FindFilesStep(
            kind="find_files", name="files", arguments="input.dir", pattern="*.txt"
        ),
        ForEachStep(
            kind="for_each",
            name="loop",
            arguments="files",
            steps=[
                TransformStep(kind="transform", name="x", arguments="item"),
            ],
        ),
    ]
    fe = sibling_steps[1]
    assert isinstance(fe, ForEachStep)
    diags = _check_for_each_target_type(fe, {"input", "files"}, sibling_steps)
    assert diags == []


def test_for_each_over_read_file_warns():
    sibling_steps: list[Step] = [
        ReadFileStep(kind="read_file", name="paper", arguments="input.path"),
        ForEachStep(
            kind="for_each",
            name="loop",
            arguments="paper",
            steps=[
                TransformStep(kind="transform", name="x", arguments="item"),
            ],
        ),
    ]
    fe = sibling_steps[1]
    assert isinstance(fe, ForEachStep)
    diags = _check_for_each_target_type(fe, {"input", "paper"}, sibling_steps)
    assert len(diags) == 1
    assert diags[0].severity == Severity.WARNING
    assert "string" in diags[0].message


def test_for_each_over_read_file_with_subpath_no_warning():
    """If the expression has a sub-path like paper.lines, don't warn."""
    sibling_steps: list[Step] = [
        ReadFileStep(kind="read_file", name="paper", arguments="input.path"),
        ForEachStep(
            kind="for_each",
            name="loop",
            arguments="paper.lines",
            steps=[
                TransformStep(kind="transform", name="x", arguments="item"),
            ],
        ),
    ]
    fe = sibling_steps[1]
    assert isinstance(fe, ForEachStep)
    diags = _check_for_each_target_type(fe, {"input", "paper"}, sibling_steps)
    assert diags == []


# ─────────────────────────────────────────────────────────────────────────
# validate_pipeline (integration)
# ─────────────────────────────────────────────────────────────────────────


def _make_pipeline(steps: list[Step]) -> PipelineDefinition:
    return PipelineDefinition(
        input={
            "type": "object",
            "properties": {"dir": {"type": "string"}},
        },
        steps=steps,
    )


def test_validate_empty_pipeline():
    defn = _make_pipeline([])
    result = validate_pipeline(defn)
    assert result.ok is True


def test_validate_simple_pipeline():
    defn = _make_pipeline(
        [
            FindFilesStep(
                kind="find_files",
                name="files",
                arguments="input.dir",
                pattern="*.txt",
            ),
            TransformStep(
                kind="transform",
                name="count",
                arguments="$count(files)",
            ),
        ]
    )
    result = validate_pipeline(defn)
    assert result.ok is True


def test_validate_catches_multiple_errors():
    defn = _make_pipeline(
        [
            TransformStep(
                kind="transform",
                name="input",  # reserved
                arguments="nonexistent",  # bad ref
            ),
            TransformStep(
                kind="transform",
                name="input",  # duplicate AND reserved
                arguments="also_bad",
            ),
        ]
    )
    result = validate_pipeline(defn)
    assert result.ok is False
    assert len(result.errors) >= 3  # reserved, bad ref, duplicate+reserved+bad ref


def test_validate_simple_pipeline_yaml(tmp_path: Path):
    """Load the real simple_pipeline.yaml fixture and validate it."""
    fixture = FIXTURES_DIR / "simple_pipeline.yaml"
    if fixture.exists():
        defn = load_pipeline(fixture)
        result = validate_pipeline(defn)
        assert result.ok is True, [
            f"[{d.severity}] {d.step_name}.{d.field}: {d.message}"
            for d in result.diagnostics
        ]


def test_validate_nested_pipeline_yaml():
    fixture = FIXTURES_DIR / "nested_pipeline.yaml"
    if fixture.exists():
        defn = load_pipeline(fixture)
        result = validate_pipeline(defn)
        assert result.ok is True, [
            f"[{d.severity}] {d.step_name}.{d.field}: {d.message}"
            for d in result.diagnostics
        ]


def test_validate_summarize_paper_yaml():
    fixture = E2E_DIR / "summarize_paper.yaml"
    if fixture.exists():
        defn = load_pipeline(fixture)
        result = validate_pipeline(defn)
        assert result.ok is True, [
            f"[{d.severity}] {d.step_name}.{d.field}: {d.message}"
            for d in result.diagnostics
        ]


# ─────────────────────────────────────────────────────────────────────────
# load_and_validate_pipeline
# ─────────────────────────────────────────────────────────────────────────


def test_load_and_validate_pipeline():
    fixture = FIXTURES_DIR / "simple_pipeline.yaml"
    if fixture.exists():
        defn, result = load_and_validate_pipeline(fixture)
        assert defn is not None
        assert result.ok is True


def test_load_and_validate_bad_yaml(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: valid: yaml: {{{}}")
    with pytest.raises(Exception):
        load_and_validate_pipeline(bad)


# ─────────────────────────────────────────────────────────────────────────
# Evaluate step validation
# ─────────────────────────────────────────────────────────────────────────


def test_evaluate_summarization_valid_args():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"source": input.text, "summary": input.summary}',
        strategy="summarization",
    )
    diags = _check_evaluate_arguments(step)
    assert len(diags) == 0


def test_evaluate_faithfulness_valid_args():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"source": input.text, "response": input.resp}',
        strategy="faithfulness",
    )
    diags = _check_evaluate_arguments(step)
    assert len(diags) == 0


def test_evaluate_summarization_missing_source():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"summary": input.summary}',
        strategy="summarization",
    )
    diags = _check_evaluate_arguments(step)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "source" in errors[0].message


def test_evaluate_summarization_missing_summary():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"source": input.text}',
        strategy="summarization",
    )
    diags = _check_evaluate_arguments(step)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "summary" in errors[0].message


def test_evaluate_faithfulness_missing_response():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"source": input.text}',
        strategy="faithfulness",
    )
    diags = _check_evaluate_arguments(step)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "response" in errors[0].message


def test_evaluate_summarization_with_response_key_warns():
    """Using 'response' key with summarization strategy should warn."""
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"source": input.text, "summary": input.s, "response": input.r}',
        strategy="summarization",
    )
    diags = _check_evaluate_arguments(step)
    warnings = [d for d in diags if d.severity == Severity.WARNING]
    assert len(warnings) == 1
    assert "response" in warnings[0].message


def test_evaluate_faithfulness_with_summary_key_warns():
    """Using 'summary' key with faithfulness strategy should warn."""
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"source": input.text, "response": input.r, "summary": input.s}',
        strategy="faithfulness",
    )
    diags = _check_evaluate_arguments(step)
    warnings = [d for d in diags if d.severity == Severity.WARNING]
    assert len(warnings) == 1
    assert "summary" in warnings[0].message


def test_evaluate_hallucination_valid_args():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"context": input.ctx, "response": input.resp}',
        strategy="hallucination",
    )
    diags = _check_evaluate_arguments(step)
    assert len(diags) == 0


def test_evaluate_hallucination_missing_context():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"response": input.resp}',
        strategy="hallucination",
    )
    diags = _check_evaluate_arguments(step)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "context" in errors[0].message


def test_evaluate_context_relevance_valid_args():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"question": input.q, "context": input.ctx}',
        strategy="context_relevance",
    )
    diags = _check_evaluate_arguments(step)
    assert len(diags) == 0


def test_evaluate_context_relevance_missing_question():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"context": input.ctx}',
        strategy="context_relevance",
    )
    diags = _check_evaluate_arguments(step)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "question" in errors[0].message


def test_evaluate_context_utilization_valid_args():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"question": input.q, "context": input.ctx, "response": input.resp}',
        strategy="context_utilization",
    )
    diags = _check_evaluate_arguments(step)
    assert len(diags) == 0


def test_evaluate_context_utilization_missing_response():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"question": input.q, "context": input.ctx}',
        strategy="context_utilization",
    )
    diags = _check_evaluate_arguments(step)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "response" in errors[0].message


def test_evaluate_factual_accuracy_valid_args():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"question": input.q, "context": input.ctx, "response": input.resp}',
        strategy="factual_accuracy",
    )
    diags = _check_evaluate_arguments(step)
    assert len(diags) == 0


def test_evaluate_factual_accuracy_missing_context():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"question": input.q, "response": input.resp}',
        strategy="factual_accuracy",
    )
    diags = _check_evaluate_arguments(step)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "context" in errors[0].message


def test_evaluate_context_conciseness_valid_args():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"question": input.q, "context": input.ctx, "concise_context": input.cc}',
        strategy="context_conciseness",
    )
    diags = _check_evaluate_arguments(step)
    assert len(diags) == 0


def test_evaluate_context_conciseness_missing_concise_context():
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"question": input.q, "context": input.ctx}',
        strategy="context_conciseness",
    )
    diags = _check_evaluate_arguments(step)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    assert len(errors) == 1
    assert "concise_context" in errors[0].message


def test_evaluate_hallucination_with_source_key_warns():
    """Using 'source' key with hallucination strategy should warn (wrong strategy)."""
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments='{"context": input.ctx, "response": input.resp, "source": input.src}',
        strategy="hallucination",
    )
    diags = _check_evaluate_arguments(step)
    warnings = [d for d in diags if d.severity == Severity.WARNING]
    assert len(warnings) == 1
    assert "source" in warnings[0].message


def test_evaluate_non_object_args_skipped():
    """Non-object-literal arguments are skipped (can't statically check)."""
    step = EvaluateStep(
        kind="evaluate",
        name="eval",
        arguments="some_step",
        strategy="summarization",
    )
    diags = _check_evaluate_arguments(step)
    assert len(diags) == 0


def test_evaluate_step_in_full_pipeline():
    """EvaluateStep integrates with the full validate_pipeline flow."""
    defn = PipelineDefinition(
        input={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
        steps=[
            TransformStep(
                kind="transform",
                name="summary",
                arguments="input.text",
            ),
            EvaluateStep(
                kind="evaluate",
                name="quality",
                arguments='{"source": input.text, "summary": summary}',
                strategy="summarization",
            ),
        ],
    )
    result = validate_pipeline(defn)
    assert result.ok is True


def test_validate_evaluate_pipeline_yaml():
    """Load the evaluate pipeline fixture and validate it."""
    fixture = FIXTURES_DIR / "evaluate_pipeline.yaml"
    if fixture.exists():
        defn = load_pipeline(fixture)
        result = validate_pipeline(defn)
        assert result.ok is True, [
            f"[{d.severity}] {d.step_name}.{d.field}: {d.message}"
            for d in result.diagnostics
        ]


def test_evaluate_step_bad_reference_caught():
    """Referencing a non-existent step in evaluate arguments is caught."""
    defn = PipelineDefinition(
        input={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
        steps=[
            EvaluateStep(
                kind="evaluate",
                name="quality",
                arguments='{"source": input.text, "summary": nonexistent}',
                strategy="summarization",
            ),
        ],
    )
    result = validate_pipeline(defn)
    assert result.ok is False
    assert any("nonexistent" in d.message for d in result.errors)
