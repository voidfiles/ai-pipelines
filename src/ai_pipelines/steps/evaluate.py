"""evaluate step: LLM-as-judge scoring for pipeline outputs.

Implements evaluation strategies synthesized from DeepEval and UpTrain
methodologies, built on the existing claude-agent-sdk infrastructure.
Zero new dependencies.

Strategies:
    summarization  (3 LLM calls) - keyphrase QA + conciseness
    faithfulness   (2 LLM calls) - claim decomposition + NLI verification
    hallucination  (2 LLM calls) - claim decomposition + contradiction detection
    context_relevance   (1 LLM call) - is retrieved context relevant to the query?
    context_utilization (1 LLM call) - did the response use all relevant context?
    factual_accuracy    (2 LLM calls) - fact extraction + 3-way verification
    context_conciseness (1 LLM call) - does compressed context retain info?
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Awaitable
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from claude_agent_sdk.types import AssistantMessage, ToolUseBlock

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import LLMError, StepExecutionError
from ai_pipelines.expressions import evaluate as evaluate_expression
from ai_pipelines.models import EvaluateStep
from ai_pipelines.steps.prompt import _parse_structured_output

_log = logging.getLogger("ai_pipelines")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _llm_structured(
    prompt_text: str,
    output_schema: dict[str, Any],
    model: str,
    system_prompt: str,
    *,
    llm_fn: Callable[..., AsyncIterator[Any]] | None = None,
) -> dict[str, Any]:
    """Make a single structured-output LLM call.

    Handles the AssistantMessage/ToolUseBlock/ResultMessage streaming dance.
    Returns the parsed structured output dict.
    Raises LLMError on failure.
    """
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        max_turns=1,
        output_format={"type": "json_schema", "schema": output_schema},
    )

    _query = llm_fn or query
    result_text: str | None = None
    structured_output: Any = None
    tool_use_structured: Any = None

    async for message in _query(prompt=prompt_text, options=options):
        if message is None:
            continue

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if (
                    isinstance(block, ToolUseBlock)
                    and block.name == "StructuredOutput"
                ):
                    tool_use_structured = block.input

        if isinstance(message, ResultMessage):
            result_text = message.result
            structured_output = message.structured_output

            if message.is_error:
                raise LLMError(f"LLM returned error: {result_text}")

    if not structured_output and tool_use_structured is not None:
        structured_output = tool_use_structured

    parsed = _parse_structured_output(structured_output, result_text)
    if parsed is None or not isinstance(parsed, dict):
        raise LLMError(
            f"Expected structured dict but got {type(parsed).__name__}: "
            f"structured_output={structured_output!r}, "
            f"result_text={result_text!r}"
        )
    return parsed


def _normalize_context(context: Any) -> str:
    """Join list context into a single string with document markers.

    Accepts either a string (passthrough) or a list of strings
    (joined with [Document N] headers).
    """
    if isinstance(context, list):
        if not context:
            return ""
        parts = [f"[Document {i + 1}]\n{str(c)}" for i, c in enumerate(context)]
        return "\n\n".join(parts)
    return str(context)


# ---------------------------------------------------------------------------
# Prompt templates: Summarization (3 LLM calls)
# ---------------------------------------------------------------------------

_KEYPHRASE_SYSTEM = (
    "You are a precise information extraction system. "
    "Extract only the most important entities and concepts."
)

_KEYPHRASE_PROMPT = """Extract the most important keyphrases from the following text.
Include: people, organizations, dates, locations, key concepts, findings, and technical terms.
Return 5-20 keyphrases ordered by importance.

Text:
{source}"""

_KEYPHRASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keyphrases": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["keyphrases"],
}

_QUESTION_SYSTEM = (
    "You are a question generation system. Generate closed-ended yes/no "
    "questions that test whether a text covers specific information."
)

_QUESTION_PROMPT = """Given these keyphrases extracted from a source text, generate one closed-ended yes/no question per keyphrase.
Each question should be answerable from the source text with a simple YES or NO.

Keyphrases:
{keyphrases}

Source text:
{source}"""

_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyphrase": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["keyphrase", "question"],
            },
        }
    },
    "required": ["questions"],
}

_ANSWER_SYSTEM = (
    "You are a careful reading comprehension evaluator. "
    "Determine if a summary contains enough information to answer each question."
)

_ANSWER_PROMPT = """For each question below, determine if the following summary contains enough information to answer it.
Answer YES if the summary provides sufficient information, NO if it does not.

Summary:
{summary}

Questions:
{questions}"""

_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string", "enum": ["YES", "NO"]},
                    "reasoning": {"type": "string"},
                },
                "required": ["question", "answer", "reasoning"],
            },
        }
    },
    "required": ["answers"],
}

# ---------------------------------------------------------------------------
# Prompt templates: Claims (shared by faithfulness + hallucination)
# ---------------------------------------------------------------------------

_CLAIMS_SYSTEM = (
    "You are a precise claim decomposition system. Break text into atomic "
    "factual statements. Use explicit nouns instead of pronouns."
)

_CLAIMS_PROMPT = """Break the following text into atomic factual claims.
Each claim should be a single, self-contained factual statement.
Use explicit nouns (no pronouns like "it", "they", "this").
Skip opinions, meta-statements, and subjective assessments.

Text:
{response}"""

_CLAIMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "original_sentence": {"type": "string"},
                },
                "required": ["claim", "original_sentence"],
            },
        }
    },
    "required": ["claims"],
}

# ---------------------------------------------------------------------------
# Prompt templates: Faithfulness NLI (binary)
# ---------------------------------------------------------------------------

_NLI_SYSTEM = (
    "You are a natural language inference system. For each claim, "
    "determine if the source text supports it."
)

_NLI_PROMPT = """For each claim below, determine if the source text supports it.
Score 1 if the source clearly supports the claim, 0 if it does not or contradicts it.

Source text:
{source}

Claims:
{claims}"""

_NLI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "integer", "enum": [0, 1]},
                    "reasoning": {"type": "string"},
                },
                "required": ["claim", "verdict", "reasoning"],
            },
        }
    },
    "required": ["verdicts"],
}

# ---------------------------------------------------------------------------
# Prompt templates: Hallucination NLI (3-way)
# ---------------------------------------------------------------------------

_HALLUCINATION_NLI_SYSTEM = (
    "You are a natural language inference system specialized in detecting "
    "contradictions. For each claim, determine if the context supports it, "
    "is neutral, or contradicts it."
)

_HALLUCINATION_NLI_PROMPT = """For each claim below, determine its relationship to the context.

- "supported": The context clearly supports this claim.
- "neutral": The context neither supports nor contradicts this claim (the claim may be true but isn't addressed in the context).
- "contradicted": The context directly contradicts this claim.

Context:
{context}

Claims:
{claims}"""

_HALLUCINATION_NLI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "neutral", "contradicted"],
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["claim", "verdict", "reasoning"],
            },
        }
    },
    "required": ["verdicts"],
}

# ---------------------------------------------------------------------------
# Prompt templates: Three-way choice (context_relevance, context_utilization,
#                                      context_conciseness)
# ---------------------------------------------------------------------------

_THREE_WAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["full", "partial", "none"],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
}

_THREE_WAY_SCORES: dict[str, float] = {
    "full": 1.0,
    "partial": 0.5,
    "none": 0.0,
}

_CONTEXT_RELEVANCE_SYSTEM = (
    "You are a retrieval quality evaluator. Assess whether retrieved "
    "context contains sufficient information to answer a question."
)

_CONTEXT_RELEVANCE_PROMPT = """Evaluate whether the following context contains sufficient information to answer the question.

Question:
{question}

Context:
{context}

Respond with:
- "full" if the context can completely answer the question.
- "partial" if the context provides some relevant information but not enough for a complete answer.
- "none" if the context does not contain any relevant information for the question."""

_CONTEXT_UTILIZATION_SYSTEM = (
    "You are a response quality evaluator. Assess whether a response "
    "incorporated all relevant information from the provided context."
)

_CONTEXT_UTILIZATION_PROMPT = """Evaluate whether the response incorporated all relevant information from the context to answer the question.
Focus on whether the response makes use of the context, not on stylistic quality.

Question:
{question}

Context:
{context}

Response:
{response}

Respond with:
- "full" if the response incorporates ALL relevant information from the context.
- "partial" if the response uses SOME relevant context information but misses important parts.
- "none" if the response does not incorporate ANY information from the context."""

_CONTEXT_CONCISENESS_SYSTEM = (
    "You are a context compression evaluator. Assess whether a condensed "
    "version of context retains all information relevant to a question."
)

_CONTEXT_CONCISENESS_PROMPT = """Evaluate whether the concise context retains all relevant information from the original context for answering the question.

Question:
{question}

Original context:
{context}

Concise context:
{concise_context}

Respond with:
- "full" if the concise context retains ALL information relevant to answering the question.
- "partial" if the concise context retains SOME but loses important relevant information.
- "none" if the concise context loses most or all relevant information."""

# ---------------------------------------------------------------------------
# Prompt templates: Factual accuracy (2 LLM calls)
# ---------------------------------------------------------------------------

_FACT_EXTRACT_SYSTEM = (
    "You are a precise fact extraction system. Extract independent, "
    "atomic factual statements from text."
)

_FACT_EXTRACT_PROMPT = """Extract the key factual statements from the following response.
Each fact should be independent and self-contained.
Use explicit nouns (no pronouns). Limit to 10 facts maximum.

Response:
{response}"""

_FACT_EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["facts"],
}

_FACT_VERIFY_SYSTEM = (
    "You are a fact verification system. For each fact, determine "
    "whether the context supports it."
)

_FACT_VERIFY_PROMPT = """For each fact below, determine if the context supports it.

- "yes": The context clearly supports this fact.
- "unclear": The context does not clearly confirm or deny this fact.
- "no": The context contradicts this fact or the fact is clearly unsupported.

Question (for reference):
{question}

Context:
{context}

Facts:
{facts}"""

_FACT_VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["yes", "unclear", "no"],
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["fact", "verdict", "reasoning"],
            },
        }
    },
    "required": ["verdicts"],
}

_FACT_VERDICT_SCORES: dict[str, float] = {
    "yes": 1.0,
    "unclear": 0.5,
    "no": 0.0,
}


# ---------------------------------------------------------------------------
# Strategy: Summarization (3 LLM calls)
# ---------------------------------------------------------------------------


async def _evaluate_summarization(
    *,
    source: Any,
    summary: Any,
    model: str,
    llm_fn: Callable[..., AsyncIterator[Any]] | None = None,
) -> dict[str, Any]:
    """Keyphrase extraction → question generation → answer evaluation.

    Score = qa_score * 0.5 + conciseness * 0.5
    """
    source_str = str(source)
    summary_str = str(summary)

    keyphrase_result = await _llm_structured(
        prompt_text=_KEYPHRASE_PROMPT.format(source=source_str),
        output_schema=_KEYPHRASE_SCHEMA,
        model=model,
        system_prompt=_KEYPHRASE_SYSTEM,
        llm_fn=llm_fn,
    )
    keyphrases = keyphrase_result.get("keyphrases", [])

    if not keyphrases:
        return {
            "score": 0.0,
            "qa_score": 0.0,
            "conciseness": _compute_conciseness(source_str, summary_str),
            "total_questions": 0,
            "correct_answers": 0,
            "keyphrases": [],
            "questions": [],
            "answers": [],
        }

    keyphrases_text = "\n".join(f"- {kp}" for kp in keyphrases)
    question_result = await _llm_structured(
        prompt_text=_QUESTION_PROMPT.format(
            keyphrases=keyphrases_text, source=source_str
        ),
        output_schema=_QUESTION_SCHEMA,
        model=model,
        system_prompt=_QUESTION_SYSTEM,
        llm_fn=llm_fn,
    )
    questions = question_result.get("questions", [])

    if not questions:
        return {
            "score": 0.0,
            "qa_score": 0.0,
            "conciseness": _compute_conciseness(source_str, summary_str),
            "total_questions": 0,
            "correct_answers": 0,
            "keyphrases": keyphrases,
            "questions": [],
            "answers": [],
        }

    questions_text = "\n".join(
        f"{i + 1}. {q['question']}" for i, q in enumerate(questions)
    )
    answer_result = await _llm_structured(
        prompt_text=_ANSWER_PROMPT.format(
            summary=summary_str, questions=questions_text
        ),
        output_schema=_ANSWER_SCHEMA,
        model=model,
        system_prompt=_ANSWER_SYSTEM,
        llm_fn=llm_fn,
    )
    answers = answer_result.get("answers", [])

    correct = sum(1 for a in answers if a.get("answer") == "YES")
    total = len(questions)
    qa_score = correct / total if total > 0 else 0.0
    conciseness = _compute_conciseness(source_str, summary_str)
    score = qa_score * 0.5 + conciseness * 0.5

    return {
        "score": round(score, 4),
        "qa_score": round(qa_score, 4),
        "conciseness": round(conciseness, 4),
        "total_questions": total,
        "correct_answers": correct,
        "keyphrases": keyphrases,
        "questions": questions,
        "answers": answers,
    }


def _compute_conciseness(source: str, summary: str) -> float:
    """Conciseness component: shorter summaries score higher."""
    source_len = len(source)
    summary_len = len(summary)
    if source_len == 0:
        return 1.0
    return 1.0 - min(summary_len, source_len) / (source_len + 1e-10)


# ---------------------------------------------------------------------------
# Strategy: Faithfulness (2 LLM calls)
# ---------------------------------------------------------------------------


async def _evaluate_faithfulness(
    *,
    source: Any,
    response: Any,
    model: str,
    llm_fn: Callable[..., AsyncIterator[Any]] | None = None,
) -> dict[str, Any]:
    """Claim decomposition → binary NLI verification.

    Score = supported_claims / total_claims
    """
    response_str = str(response)
    source_str = str(source)

    claims_result = await _llm_structured(
        prompt_text=_CLAIMS_PROMPT.format(response=response_str),
        output_schema=_CLAIMS_SCHEMA,
        model=model,
        system_prompt=_CLAIMS_SYSTEM,
        llm_fn=llm_fn,
    )
    claims = claims_result.get("claims", [])

    if not claims:
        return {
            "score": 1.0,
            "supported_claims": 0,
            "total_claims": 0,
            "claims": [],
            "verdicts": [],
        }

    claims_text = "\n".join(
        f"{i + 1}. {c['claim']}" for i, c in enumerate(claims)
    )
    nli_result = await _llm_structured(
        prompt_text=_NLI_PROMPT.format(source=source_str, claims=claims_text),
        output_schema=_NLI_SCHEMA,
        model=model,
        system_prompt=_NLI_SYSTEM,
        llm_fn=llm_fn,
    )
    verdicts = nli_result.get("verdicts", [])

    supported = sum(1 for v in verdicts if v.get("verdict") == 1)
    total = len(claims)
    score = supported / total if total > 0 else 1.0

    return {
        "score": round(score, 4),
        "supported_claims": supported,
        "total_claims": total,
        "claims": claims,
        "verdicts": verdicts,
    }


# ---------------------------------------------------------------------------
# Strategy: Hallucination (2 LLM calls)
# ---------------------------------------------------------------------------


async def _evaluate_hallucination(
    *,
    context: Any,
    response: Any,
    model: str,
    llm_fn: Callable[..., AsyncIterator[Any]] | None = None,
) -> dict[str, Any]:
    """Claim decomposition → 3-way contradiction detection.

    Score = 1 - contradicted / total (higher is better = less hallucination).
    "Neutral" claims (not in context, not contradicting it) do NOT penalize.
    """
    response_str = str(response)
    context_str = _normalize_context(context)

    claims_result = await _llm_structured(
        prompt_text=_CLAIMS_PROMPT.format(response=response_str),
        output_schema=_CLAIMS_SCHEMA,
        model=model,
        system_prompt=_CLAIMS_SYSTEM,
        llm_fn=llm_fn,
    )
    claims = claims_result.get("claims", [])

    if not claims:
        return {
            "score": 1.0,
            "contradicted_claims": 0,
            "total_claims": 0,
            "claims": [],
            "verdicts": [],
        }

    claims_text = "\n".join(
        f"{i + 1}. {c['claim']}" for i, c in enumerate(claims)
    )
    nli_result = await _llm_structured(
        prompt_text=_HALLUCINATION_NLI_PROMPT.format(
            context=context_str, claims=claims_text
        ),
        output_schema=_HALLUCINATION_NLI_SCHEMA,
        model=model,
        system_prompt=_HALLUCINATION_NLI_SYSTEM,
        llm_fn=llm_fn,
    )
    verdicts = nli_result.get("verdicts", [])

    contradicted = sum(
        1 for v in verdicts if v.get("verdict") == "contradicted"
    )
    total = len(claims)
    score = 1.0 - contradicted / total if total > 0 else 1.0

    return {
        "score": round(score, 4),
        "contradicted_claims": contradicted,
        "total_claims": total,
        "claims": claims,
        "verdicts": verdicts,
    }


# ---------------------------------------------------------------------------
# Strategy: Context Relevance (1 LLM call)
# ---------------------------------------------------------------------------


async def _evaluate_context_relevance(
    *,
    question: Any,
    context: Any,
    model: str,
    llm_fn: Callable[..., AsyncIterator[Any]] | None = None,
) -> dict[str, Any]:
    """Single 3-way judgment: is the context relevant to the question?

    Evaluates the retriever, not the generator.
    """
    result = await _llm_structured(
        prompt_text=_CONTEXT_RELEVANCE_PROMPT.format(
            question=str(question),
            context=_normalize_context(context),
        ),
        output_schema=_THREE_WAY_SCHEMA,
        model=model,
        system_prompt=_CONTEXT_RELEVANCE_SYSTEM,
        llm_fn=llm_fn,
    )

    verdict = result.get("verdict", "none")
    return {
        "score": _THREE_WAY_SCORES.get(verdict, 0.0),
        "verdict": verdict,
        "reasoning": result.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Strategy: Context Utilization (1 LLM call)
# ---------------------------------------------------------------------------


async def _evaluate_context_utilization(
    *,
    question: Any,
    context: Any,
    response: Any,
    model: str,
    llm_fn: Callable[..., AsyncIterator[Any]] | None = None,
) -> dict[str, Any]:
    """Single 3-way judgment: did the response use all relevant context?

    Evaluates the generator's use of retrieved context.
    """
    result = await _llm_structured(
        prompt_text=_CONTEXT_UTILIZATION_PROMPT.format(
            question=str(question),
            context=_normalize_context(context),
            response=str(response),
        ),
        output_schema=_THREE_WAY_SCHEMA,
        model=model,
        system_prompt=_CONTEXT_UTILIZATION_SYSTEM,
        llm_fn=llm_fn,
    )

    verdict = result.get("verdict", "none")
    return {
        "score": _THREE_WAY_SCORES.get(verdict, 0.0),
        "verdict": verdict,
        "reasoning": result.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Strategy: Context Conciseness (1 LLM call)
# ---------------------------------------------------------------------------


async def _evaluate_context_conciseness(
    *,
    question: Any,
    context: Any,
    concise_context: Any,
    model: str,
    llm_fn: Callable[..., AsyncIterator[Any]] | None = None,
) -> dict[str, Any]:
    """Single 3-way judgment: does the condensed context retain all info?

    Evaluates context compression/summarization quality.
    """
    result = await _llm_structured(
        prompt_text=_CONTEXT_CONCISENESS_PROMPT.format(
            question=str(question),
            context=_normalize_context(context),
            concise_context=_normalize_context(concise_context),
        ),
        output_schema=_THREE_WAY_SCHEMA,
        model=model,
        system_prompt=_CONTEXT_CONCISENESS_SYSTEM,
        llm_fn=llm_fn,
    )

    verdict = result.get("verdict", "none")
    return {
        "score": _THREE_WAY_SCORES.get(verdict, 0.0),
        "verdict": verdict,
        "reasoning": result.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Strategy: Factual Accuracy (2 LLM calls)
# ---------------------------------------------------------------------------


async def _evaluate_factual_accuracy(
    *,
    question: Any,
    context: Any,
    response: Any,
    model: str,
    llm_fn: Callable[..., AsyncIterator[Any]] | None = None,
) -> dict[str, Any]:
    """Fact extraction → 3-way verification (yes/unclear/no).

    Score = mean of per-fact scores (yes=1.0, unclear=0.5, no=0.0).
    Based on FActScore methodology.
    """
    response_str = str(response)
    context_str = _normalize_context(context)

    fact_result = await _llm_structured(
        prompt_text=_FACT_EXTRACT_PROMPT.format(response=response_str),
        output_schema=_FACT_EXTRACT_SCHEMA,
        model=model,
        system_prompt=_FACT_EXTRACT_SYSTEM,
        llm_fn=llm_fn,
    )
    facts = fact_result.get("facts", [])

    if not facts:
        return {
            "score": 1.0,
            "facts": [],
            "verdicts": [],
        }

    facts_text = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    verify_result = await _llm_structured(
        prompt_text=_FACT_VERIFY_PROMPT.format(
            question=str(question),
            context=context_str,
            facts=facts_text,
        ),
        output_schema=_FACT_VERIFY_SCHEMA,
        model=model,
        system_prompt=_FACT_VERIFY_SYSTEM,
        llm_fn=llm_fn,
    )
    verdicts = verify_result.get("verdicts", [])

    per_fact_scores = [
        _FACT_VERDICT_SCORES.get(v.get("verdict", "no"), 0.0)
        for v in verdicts
    ]
    score = sum(per_fact_scores) / len(per_fact_scores) if per_fact_scores else 1.0

    return {
        "score": round(score, 4),
        "facts": facts,
        "verdicts": verdicts,
    }


# ---------------------------------------------------------------------------
# Strategy dispatch table
# ---------------------------------------------------------------------------

_STRATEGIES: dict[
    str, tuple[set[str], Callable[..., Awaitable[dict[str, Any]]]]
] = {
    "summarization": ({"source", "summary"}, _evaluate_summarization),
    "faithfulness": ({"source", "response"}, _evaluate_faithfulness),
    "hallucination": ({"context", "response"}, _evaluate_hallucination),
    "context_relevance": ({"question", "context"}, _evaluate_context_relevance),
    "context_utilization": (
        {"question", "context", "response"},
        _evaluate_context_utilization,
    ),
    "factual_accuracy": (
        {"question", "context", "response"},
        _evaluate_factual_accuracy,
    ),
    "context_conciseness": (
        {"question", "context", "concise_context"},
        _evaluate_context_conciseness,
    ),
}


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------


async def execute_evaluate(
    step: EvaluateStep,
    context: PipelineContext,
    *,
    llm_fn: Callable[..., AsyncIterator[Any]] | None = None,
) -> dict[str, Any]:
    """Execute an evaluate step.

    1. Evaluate JSONata arguments against pipeline context.
    2. Validate required keys for the strategy.
    3. Dispatch to the strategy function.
    """
    data = context.get_data()
    resolved = evaluate_expression(step.arguments, data)

    if not isinstance(resolved, dict):
        raise StepExecutionError(
            step.name,
            f"evaluate arguments must produce a dict, got {type(resolved).__name__}",
        )

    required_keys, strategy_fn = _STRATEGIES[step.strategy]

    for key in sorted(required_keys):
        if resolved.get(key) is None:
            raise StepExecutionError(
                step.name,
                f"Strategy '{step.strategy}' requires key '{key}' in arguments",
            )

    kwargs = {k: resolved[k] for k in required_keys}
    return await strategy_fn(**kwargs, model=step.model, llm_fn=llm_fn)
