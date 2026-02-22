"""Tests for PipelineContext.

Validates result accumulation, child scoping (for for_each loops),
and duplicate name detection.
"""

from __future__ import annotations

import pytest

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import ExpressionError


def test_initial_context_has_input():
    ctx = PipelineContext({"key": "value"})
    data = ctx.get_data()
    assert data["input"] == {"key": "value"}


def test_set_and_get_result():
    ctx = PipelineContext({"k": "v"})
    ctx.set_result("step1", [1, 2, 3])
    data = ctx.get_data()
    assert data["step1"] == [1, 2, 3]
    assert data["input"] == {"k": "v"}


def test_multiple_results():
    ctx = PipelineContext({})
    ctx.set_result("a", "alpha")
    ctx.set_result("b", "beta")
    data = ctx.get_data()
    assert data["a"] == "alpha"
    assert data["b"] == "beta"


def test_duplicate_name_raises():
    ctx = PipelineContext({})
    ctx.set_result("step1", "first")
    with pytest.raises(ExpressionError, match="Duplicate step name"):
        ctx.set_result("step1", "second")


def test_child_sees_parent_data():
    ctx = PipelineContext({"k": "v"})
    ctx.set_result("step1", "parent_data")
    child = ctx.child({"item": "loop_value"})
    child_data = child.get_data()
    assert child_data["step1"] == "parent_data"
    assert child_data["input"] == {"k": "v"}
    assert child_data["item"] == "loop_value"


def test_child_results_isolated_from_parent():
    ctx = PipelineContext({})
    ctx.set_result("step1", "parent_data")
    child = ctx.child({"item": "x"})
    child.set_result("inner_step", "child_data")
    assert "inner_step" not in ctx.get_data()


def test_child_can_shadow_parent_binding():
    """Child's extra bindings override parent values (e.g., 'item' in nested loops)."""
    ctx = PipelineContext({})
    ctx_with_item = ctx.child({"item": "outer"})
    inner = ctx_with_item.child({"item": "inner"})
    assert inner.get_data()["item"] == "inner"


def test_child_without_extra():
    ctx = PipelineContext({"k": "v"})
    child = ctx.child()
    assert child.get_data()["input"] == {"k": "v"}


def test_nested_children():
    ctx = PipelineContext({})
    ctx.set_result("root", "r")
    child1 = ctx.child({"level": 1})
    child1.set_result("c1", "one")
    child2 = child1.child({"level": 2})
    child2.set_result("c2", "two")

    # child2 sees everything from ancestors plus its own
    data = child2.get_data()
    assert data["root"] == "r"
    assert data["c1"] == "one"
    assert data["c2"] == "two"
    assert data["level"] == 2

    # parent doesn't see child results
    assert "c2" not in child1.get_data()
    assert "c1" not in ctx.get_data()
