"""prompt step: execute LLM calls via claude-agent-sdk."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import jsonschema
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from claude_agent_sdk.types import AssistantMessage, TextBlock, ToolUseBlock

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import LLMError, ValidationError
from ai_pipelines.expressions import evaluate
from ai_pipelines.models import PromptStep
from ai_pipelines.templates import render_template

_log = logging.getLogger("ai_pipelines")


def _parse_structured_output(
    structured_output: Any, result_text: str | None
) -> Any:
    """Normalize the LLM's structured output into a Python object.

    Handles three cases from the SDK:
    1. ``structured_output`` is already a dict/list: return as-is.
    2. ``structured_output`` is a JSON string: parse it.
    3. ``structured_output`` is None but ``result_text`` is valid JSON: parse that.

    Returns None if nothing parseable is found.
    """
    if structured_output is not None:
        if isinstance(structured_output, str):
            try:
                return json.loads(structured_output)
            except (json.JSONDecodeError, TypeError):
                _log.warning(
                    "structured_output was a string but not valid JSON, "
                    "falling back to result_text"
                )
        else:
            return structured_output

    # Fallback: try parsing result_text as JSON
    if result_text is not None:
        try:
            return json.loads(result_text)
        except (json.JSONDecodeError, TypeError):
            pass

    return None


async def execute_prompt(
    step: PromptStep,
    context: PipelineContext,
    *,
    llm_fn: Callable[..., AsyncIterator[Any]] | None = None,
    return_cost: bool = False,
) -> Any:
    """Execute an LLM prompt step.

    1. Evaluate ``arguments`` JSONata expression to get template variables.
    2. Render the Jinja2 template with those variables.
    3. Build ClaudeAgentOptions with model, system_prompt, output_format.
    4. Call the LLM and collect the result.
    5. Validate structured output against the ``output`` JSON schema.

    Args:
        step: The prompt step definition.
        context: Pipeline context with prior step results.
        llm_fn: Optional replacement for ``claude_agent_sdk.query``
            (injected for testing).
        return_cost: If True, return a tuple of (result, cost_usd).

    Returns:
        Structured output dict if ``output`` schema is set,
        otherwise the raw text response. If ``return_cost`` is True,
        returns ``(result, cost_usd)``.
    """
    data = context.get_data()

    # 1. Resolve arguments
    args: dict[str, Any] = {}
    if step.arguments:
        resolved = evaluate(step.arguments, data)
        if isinstance(resolved, dict):
            args = resolved
        else:
            args = {"value": resolved}

    # 2. Render template
    rendered_prompt = render_template(step.template, args)

    # 3. Build options
    options = ClaudeAgentOptions(
        model=step.model,
        system_prompt=step.system_prompt,
        max_turns=1,
    )
    if step.output:
        options.output_format = {
            "type": "json_schema",
            "schema": step.output,
        }

    # 4. Call LLM
    _query = llm_fn or query
    result_text: str | None = None
    structured_output: Any = None
    cost_usd: float | None = None
    assistant_text_parts: list[str] = []

    # The SDK delivers structured output via a StructuredOutput tool call
    # in AssistantMessage content blocks. With max_turns=1 the session
    # ends before the ResultMessage can include it, so we grab it directly
    # from the tool use block.
    tool_use_structured: Any = None

    # Unknown message types (e.g. rate_limit_event) are patched to return
    # None by sdk_patch.apply(), so we just skip them here.
    async for message in _query(prompt=rendered_prompt, options=options):
        if message is None:
            continue

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    assistant_text_parts.append(block.text)
                elif (
                    isinstance(block, ToolUseBlock)
                    and block.name == "StructuredOutput"
                ):
                    tool_use_structured = block.input

        if isinstance(message, ResultMessage):
            result_text = message.result
            structured_output = message.structured_output
            cost_usd = message.total_cost_usd

            _log.debug(
                "Step '%s' LLM result: structured_output type=%s, "
                "result_text length=%s, cost=$%s",
                step.name,
                type(structured_output).__name__,
                len(result_text) if result_text else 0,
                cost_usd,
            )

            if message.is_error:
                raise LLMError(f"LLM returned error: {result_text}")

    # Prefer ResultMessage.structured_output, then the StructuredOutput
    # tool call payload, then AssistantMessage text as last resort.
    if not structured_output and tool_use_structured is not None:
        structured_output = tool_use_structured
    if not result_text and not structured_output and assistant_text_parts:
        result_text = "".join(assistant_text_parts)

    # 5. Normalize and validate structured output
    #
    # The SDK might hand us structured_output in several forms:
    #   - A parsed dict (ideal path)
    #   - A JSON string (needs json.loads)
    #   - None, while result_text contains the JSON (fallback)
    #
    # We normalize all of these into a proper Python dict before validation.
    if step.output:
        parsed = _parse_structured_output(structured_output, result_text)
        if parsed is not None:
            try:
                jsonschema.validate(instance=parsed, schema=step.output)
            except jsonschema.ValidationError as e:
                raise ValidationError(
                    f"LLM output validation failed: {e.message}"
                ) from e
            result = parsed
        else:
            raise LLMError(
                f"Step '{step.name}' expected structured output but got "
                f"nothing parseable. structured_output={structured_output!r}, "
                f"result_text={result_text!r}"
            )
    else:
        result = result_text

    if return_cost:
        return result, cost_usd
    return result
