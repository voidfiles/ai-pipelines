"""Tests for the JSONata expression evaluator.

These cover every expression pattern used in pipeline YAML files:
dot access, array indexing, filtering, object construction, projection, and aggregation.
"""

from __future__ import annotations

import pytest

from ai_pipelines.expressions import evaluate
from ai_pipelines.errors import ExpressionError


# ── Simple path access ────────────────────────────────────────────


def test_dot_access_top_level():
    ctx = {"input": {"name": "test"}}
    assert evaluate("input.name", ctx) == "test"


def test_dot_access_nested():
    ctx = {"step1": {"data": {"value": 42}}}
    assert evaluate("step1.data.value", ctx) == 42


def test_dot_access_returns_dict():
    ctx = {"input": {"nested": {"a": 1, "b": 2}}}
    result = evaluate("input.nested", ctx)
    assert result == {"a": 1, "b": 2}


def test_dot_access_returns_list():
    ctx = {"files": ["a.txt", "b.txt", "c.txt"]}
    assert evaluate("files", ctx) == ["a.txt", "b.txt", "c.txt"]


# ── Array indexing ────────────────────────────────────────────────


def test_array_index_first():
    ctx = {"files": ["a.txt", "b.txt", "c.txt"]}
    assert evaluate("files[0]", ctx) == "a.txt"


def test_array_index_second():
    ctx = {"files": ["a.txt", "b.txt", "c.txt"]}
    assert evaluate("files[1]", ctx) == "b.txt"


def test_array_index_negative():
    ctx = {"files": ["a.txt", "b.txt", "c.txt"]}
    assert evaluate("files[-1]", ctx) == "c.txt"


def test_dot_access_into_array_element():
    ctx = {"extraction": {"claims": [{"text": "claim1"}, {"text": "claim2"}]}}
    assert evaluate("extraction.claims[0].text", ctx) == "claim1"


# ── Array filtering ──────────────────────────────────────────────


def test_filter_boolean_true():
    ctx = {
        "claims": [
            {"text": "a", "verified": True},
            {"text": "b", "verified": False},
            {"text": "c", "verified": True},
        ]
    }
    result = evaluate("claims[verified = true]", ctx)
    assert len(result) == 2
    assert all(c["verified"] for c in result)


def test_filter_not_equal():
    ctx = {
        "claims": [
            {"text": "a", "verified": True},
            {"text": "b", "verified": False},
            {"text": "c", "verified": True},
        ]
    }
    result = evaluate("claims[verified != false]", ctx)
    assert len(result) == 2


def test_filter_numeric_comparison():
    ctx = {
        "scores": [
            {"name": "a", "value": 10},
            {"name": "b", "value": 50},
            {"name": "c", "value": 90},
        ]
    }
    result = evaluate("scores[value > 20]", ctx)
    assert len(result) == 2
    assert result[0]["name"] == "b"


# ── Object construction ──────────────────────────────────────────


def test_object_construction():
    ctx = {"item": {"name": "test.md", "size": 42}}
    result = evaluate('{"filename": item.name, "title": item.name}', ctx)
    assert result == {"filename": "test.md", "title": "test.md"}


def test_object_construction_mixed_sources():
    ctx = {
        "transcript": {"title": "Session 1"},
        "summary": {"text": "Good session"},
    }
    result = evaluate(
        '{"title": transcript.title, "summary": summary.text}', ctx
    )
    assert result == {"title": "Session 1", "summary": "Good session"}


# ── Array projection (map) ───────────────────────────────────────


def test_dot_access_maps_over_array():
    """JSONata's dot operator auto-maps when navigating through arrays."""
    ctx = {
        "claims": [
            {"text": "claim1", "score": 0.9},
            {"text": "claim2", "score": 0.5},
        ]
    }
    result = evaluate("claims.text", ctx)
    assert result == ["claim1", "claim2"]


def test_deep_path_flattens_nested_arrays():
    """JSONata flattens when traversing through arrays of arrays."""
    ctx = {
        "chunk_results": [
            {"verification": {"verified_claims": [{"text": "a"}, {"text": "b"}]}},
            {"verification": {"verified_claims": [{"text": "c"}]}},
        ]
    }
    result = evaluate("chunk_results.verification.verified_claims", ctx)
    assert len(result) == 3
    texts = [c["text"] for c in result]
    assert texts == ["a", "b", "c"]


# ── Built-in functions ────────────────────────────────────────────


def test_count_function():
    ctx = {"items": [1, 2, 3, 4, 5]}
    assert evaluate("$count(items)", ctx) == 5


def test_sum_function():
    ctx = {"values": [{"n": 10}, {"n": 20}, {"n": 30}]}
    assert evaluate("$sum(values.n)", ctx) == 60


def test_string_function():
    ctx = {"value": 42}
    assert evaluate("$string(value)", ctx) == "42"


# ── String concatenation ─────────────────────────────────────────


def test_string_concat():
    ctx = {"item": {"name": "test.md"}}
    result = evaluate('"File: " & item.name', ctx)
    assert result == "File: test.md"


# ── Error handling ────────────────────────────────────────────────


def test_missing_path_returns_none():
    """JSONata returns None (undefined) for missing paths, not an error."""
    ctx = {"input": {"name": "test"}}
    result = evaluate("input.nonexistent", ctx)
    assert result is None


def test_invalid_expression_raises():
    ctx = {"input": {}}
    with pytest.raises(ExpressionError):
        evaluate("}{invalid", ctx)


# ── Wildcard / complex patterns from the YAML example ────────────


def test_expression_with_input_prefix():
    """The most common pattern: input.field_name."""
    ctx = {"input": {"transcript_dir": "/data/transcripts"}}
    assert evaluate("input.transcript_dir", ctx) == "/data/transcripts"


def test_step_result_reference():
    """Steps reference prior step results by name."""
    ctx = {
        "input": {"dir": "/data"},
        "transcripts": {"files": ["a.md", "b.md"]},
    }
    assert evaluate("transcripts.files", ctx) == ["a.md", "b.md"]
