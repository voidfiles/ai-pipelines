"""Tests for prompt step executor.

Uses a deterministic fake llm_fn that yields real ResultMessage objects
instead of calling the actual claude-agent-sdk API.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
from claude_agent_sdk.types import AssistantMessage, TextBlock, ToolUseBlock

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import LLMError
from ai_pipelines.models import PromptStep
from ai_pipelines.steps.prompt import _parse_structured_output, execute_prompt


async def make_fake_llm(
    structured_output: Any = None,
    result_text: str | None = None,
    is_error: bool = False,
    cost: float = 0.001,
) -> Any:
    """Build a fake llm_fn that returns a predetermined response."""

    async def fake_query(
        *, prompt: str, options: ClaudeAgentOptions | None = None, **kwargs: Any
    ) -> AsyncIterator[ResultMessage]:
        yield ResultMessage(
            subtype="success" if not is_error else "error",
            duration_ms=100,
            duration_api_ms=80,
            is_error=is_error,
            num_turns=1,
            session_id="test-session",
            total_cost_usd=cost,
            result=result_text,
            structured_output=structured_output,
        )

    return fake_query


@pytest.mark.asyncio
async def test_prompt_structured_output():
    output = {"claims": [{"text": "test claim", "confidence": 0.9}]}
    fake = await make_fake_llm(structured_output=output)

    step = PromptStep(
        kind="prompt",
        name="extraction",
        arguments='{"chunk_text": input.text}',
        model="haiku",
        system_prompt="You are an expert extractor.",
        template="Extract from: {{ args.chunk_text }}",
        output={
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                    },
                }
            },
            "required": ["claims"],
        },
    )

    ctx = PipelineContext({"text": "The Earth is round."})
    result = await execute_prompt(step, ctx, llm_fn=fake)
    assert "claims" in result
    assert result["claims"][0]["text"] == "test claim"


@pytest.mark.asyncio
async def test_prompt_text_output():
    fake = await make_fake_llm(result_text="Hello from the LLM")

    step = PromptStep(
        kind="prompt",
        name="greeting",
        template="Say hello to {{ args.name }}",
        arguments='{"name": input.user}',
    )

    ctx = PipelineContext({"user": "Alex"})
    result = await execute_prompt(step, ctx, llm_fn=fake)
    assert result == "Hello from the LLM"


@pytest.mark.asyncio
async def test_prompt_no_arguments():
    fake = await make_fake_llm(result_text="response")

    step = PromptStep(
        kind="prompt",
        name="simple",
        template="Just a static prompt",
    )

    ctx = PipelineContext({})
    result = await execute_prompt(step, ctx, llm_fn=fake)
    assert result == "response"


@pytest.mark.asyncio
async def test_prompt_llm_error():
    fake = await make_fake_llm(is_error=True, result_text="Rate limited")

    step = PromptStep(
        kind="prompt",
        name="failing",
        template="This will fail",
    )

    ctx = PipelineContext({})
    with pytest.raises(LLMError, match="LLM returned error"):
        await execute_prompt(step, ctx, llm_fn=fake)


@pytest.mark.asyncio
async def test_prompt_returns_cost():
    fake = await make_fake_llm(
        result_text="result", cost=0.005
    )

    step = PromptStep(
        kind="prompt",
        name="costed",
        template="Count my money",
    )

    ctx = PipelineContext({})
    result, cost = await execute_prompt(
        step, ctx, llm_fn=fake, return_cost=True
    )
    assert cost == 0.005


@pytest.mark.asyncio
async def test_prompt_output_validation_fails():
    """Structured output that doesn't match the schema should raise."""
    bad_output = {"wrong_key": "bad data"}
    fake = await make_fake_llm(structured_output=bad_output)

    step = PromptStep(
        kind="prompt",
        name="validated",
        template="Extract stuff",
        output={
            "type": "object",
            "properties": {"claims": {"type": "array"}},
            "required": ["claims"],
        },
    )

    ctx = PipelineContext({})
    with pytest.raises(Exception):  # ValidationError from jsonschema
        await execute_prompt(step, ctx, llm_fn=fake)


# ---------------------------------------------------------------------------
# _parse_structured_output unit tests
# ---------------------------------------------------------------------------


class TestParseStructuredOutput:
    """Test the output normalization logic that handles SDK quirks."""

    def test_dict_passthrough(self):
        """Already-parsed dict comes through untouched."""
        data = {"key_points": [{"point": "a"}], "section_topic": "Intro"}
        assert _parse_structured_output(data, None) is data

    def test_list_passthrough(self):
        """Already-parsed list comes through untouched."""
        data = [1, 2, 3]
        assert _parse_structured_output(data, None) is data

    def test_json_string_parsed(self):
        """JSON string in structured_output gets parsed to a dict."""
        data = {"key": "value", "nested": [1, 2]}
        result = _parse_structured_output(json.dumps(data), None)
        assert result == data

    def test_none_falls_back_to_result_text(self):
        """When structured_output is None, parse result_text as JSON."""
        data = {"section_topic": "Methods", "key_points": []}
        result = _parse_structured_output(None, json.dumps(data))
        assert result == data

    def test_none_with_non_json_result_text(self):
        """When both are useless, return None."""
        assert _parse_structured_output(None, "just plain text") is None

    def test_none_with_none_result_text(self):
        """Both None returns None."""
        assert _parse_structured_output(None, None) is None

    def test_invalid_json_string_falls_back(self):
        """Bad JSON in structured_output falls back to result_text."""
        good = {"key": "value"}
        result = _parse_structured_output("not valid json {{{", json.dumps(good))
        assert result == good

    def test_invalid_everything_returns_none(self):
        """Bad JSON in both returns None."""
        assert _parse_structured_output("not json", "also not json") is None

    def test_int_passthrough(self):
        """Non-dict, non-string, non-None values pass through."""
        assert _parse_structured_output(42, None) == 42

    def test_bool_passthrough(self):
        assert _parse_structured_output(True, None) is True


# ---------------------------------------------------------------------------
# Integration tests: structured output fallback paths
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["topic", "points"],
}


@pytest.mark.asyncio
async def test_prompt_structured_output_from_json_string():
    """SDK returns structured_output as a JSON string (not a dict)."""
    data = {"topic": "Testing", "points": ["a", "b"]}
    fake = await make_fake_llm(structured_output=json.dumps(data))

    step = PromptStep(
        kind="prompt",
        name="json_string",
        template="extract",
        output=SIMPLE_SCHEMA,
    )

    ctx = PipelineContext({})
    result = await execute_prompt(step, ctx, llm_fn=fake)
    assert result == data


@pytest.mark.asyncio
async def test_prompt_structured_output_none_fallback_to_result_text():
    """SDK returns structured_output=None but result_text has the JSON."""
    data = {"topic": "Fallback", "points": ["x"]}
    fake = await make_fake_llm(
        structured_output=None, result_text=json.dumps(data)
    )

    step = PromptStep(
        kind="prompt",
        name="fallback",
        template="extract",
        output=SIMPLE_SCHEMA,
    )

    ctx = PipelineContext({})
    result = await execute_prompt(step, ctx, llm_fn=fake)
    assert result == data


@pytest.mark.asyncio
async def test_prompt_no_structured_output_no_json_in_text():
    """No structured output and result_text is plain text: raise LLMError."""
    fake = await make_fake_llm(
        structured_output=None, result_text="just words"
    )

    step = PromptStep(
        kind="prompt",
        name="plain",
        template="say something",
        output=SIMPLE_SCHEMA,
    )

    ctx = PipelineContext({})
    with pytest.raises(LLMError, match="expected structured output"):
        await execute_prompt(step, ctx, llm_fn=fake)


# ---------------------------------------------------------------------------
# StructuredOutput tool use block extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_extracts_structured_output_from_tool_use_block():
    """When ResultMessage has no structured_output, extract it from the
    StructuredOutput tool call in AssistantMessage content blocks.

    This is the real-world path: the SDK delivers structured output via
    a StructuredOutput tool call, and with max_turns=1 the ResultMessage
    comes back empty (subtype='error_max_turns').
    """
    data = {"topic": "Tool use extraction", "points": ["works"]}

    async def fake_query(
        *, prompt: str, options: ClaudeAgentOptions | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        # The LLM calls the StructuredOutput tool
        yield AssistantMessage(
            content=[
                ToolUseBlock(
                    id="toolu_123",
                    name="StructuredOutput",
                    input=data,
                ),
            ],
            model="claude-haiku-4-5-20251001",
        )
        # Then the ResultMessage comes back empty (max_turns hit)
        yield ResultMessage(
            subtype="error_max_turns",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=2,
            session_id="test-session",
            total_cost_usd=0.008,
            result=None,
            structured_output=None,
        )

    step = PromptStep(
        kind="prompt",
        name="tool_use_test",
        template="extract stuff",
        output=SIMPLE_SCHEMA,
    )

    ctx = PipelineContext({})
    result = await execute_prompt(step, ctx, llm_fn=fake_query)
    assert result == data


@pytest.mark.asyncio
async def test_prompt_result_message_structured_output_preferred_over_tool_use():
    """If ResultMessage has structured_output, prefer it over the tool use block."""
    tool_data = {"topic": "from tool", "points": ["tool"]}
    result_data = {"topic": "from result", "points": ["result"]}

    async def fake_query(
        *, prompt: str, options: ClaudeAgentOptions | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        yield AssistantMessage(
            content=[
                ToolUseBlock(
                    id="toolu_456",
                    name="StructuredOutput",
                    input=tool_data,
                ),
            ],
            model="claude-haiku-4-5-20251001",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=1,
            session_id="test-session",
            total_cost_usd=0.008,
            result=None,
            structured_output=result_data,
        )

    step = PromptStep(
        kind="prompt",
        name="prefer_result",
        template="extract stuff",
        output=SIMPLE_SCHEMA,
    )

    ctx = PipelineContext({})
    result = await execute_prompt(step, ctx, llm_fn=fake_query)
    assert result == result_data
