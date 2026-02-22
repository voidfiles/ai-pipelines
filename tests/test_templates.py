"""Tests for Jinja2 template rendering.

Prompt steps use Jinja2 templates with {{ args.field }} syntax.
StrictUndefined ensures missing variables fail loudly.
"""

from __future__ import annotations

import pytest
from jinja2 import UndefinedError

from ai_pipelines.templates import render_template


def test_simple_variable():
    result = render_template("Hello {{ args.name }}", {"name": "World"})
    assert result == "Hello World"


def test_nested_access():
    result = render_template(
        "{{ args.user.name }}", {"user": {"name": "Alex"}}
    )
    assert result == "Alex"


def test_missing_variable_raises():
    with pytest.raises(UndefinedError):
        render_template("{{ args.missing }}", {})


def test_multiline_template():
    template = """Extract claims from:
{{ args.content }}

Focus on factual statements."""
    result = render_template(template, {"content": "The sky is blue."})
    assert "The sky is blue." in result
    assert "Extract claims from:" in result
    assert "Focus on factual statements." in result


def test_loop_in_template():
    template = """{% for claim in args.claims %}
- {{ claim.text }}
{% endfor %}"""
    result = render_template(
        template,
        {"claims": [{"text": "Claim 1"}, {"text": "Claim 2"}]},
    )
    assert "- Claim 1" in result
    assert "- Claim 2" in result


def test_conditional_in_template():
    template = """{% if args.speaker %}Speaker: {{ args.speaker }}{% endif %}"""
    assert "Speaker: Alice" in render_template(template, {"speaker": "Alice"})
    assert render_template(template, {"speaker": ""}) == ""


def test_loop_index():
    template = """{% for claim in args.claims %}
{{ loop.index }}. {{ claim.text }}
{% endfor %}"""
    result = render_template(
        template,
        {"claims": [{"text": "First"}, {"text": "Second"}]},
    )
    assert "1. First" in result
    assert "2. Second" in result


def test_preserves_trailing_newline():
    result = render_template("line one\n", {"unused": True})
    assert result == "line one\n"


def test_dict_get_in_template():
    """Templates can use .get() for optional fields (common in YAML example)."""
    template = """{% if args.claim.get('speaker') %}{{ args.claim.speaker }}{% endif %}"""
    with_speaker = render_template(template, {"claim": {"speaker": "Bob"}})
    assert with_speaker == "Bob"

    without_speaker = render_template(template, {"claim": {}})
    assert without_speaker == ""


def test_format_filter():
    """The YAML example uses format filter for confidence scores."""
    template = """{{ "%.2f"|format(args.confidence) }}"""
    result = render_template(template, {"confidence": 0.8567})
    assert result == "0.86"


def test_complex_prompt_template():
    """Realistic prompt template from the YAML example."""
    template = """Session: {{ args.session_title }}
Source: {{ args.source_file }}
Chunk: {{ args.chunk_index }}

Transcript segment:
{{ args.chunk_text }}

Extract key claims, quotes, and statements from this transcript segment."""

    result = render_template(
        template,
        {
            "session_title": "Keynote",
            "source_file": "keynote.md",
            "chunk_index": 0,
            "chunk_text": "The CEO said we're pivoting to AI.",
        },
    )
    assert "Session: Keynote" in result
    assert "Source: keynote.md" in result
    assert "Chunk: 0" in result
    assert "The CEO said we're pivoting to AI." in result
