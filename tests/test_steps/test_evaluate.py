"""Tests for evaluate step executor.

Uses a sequential fake LLM that returns predetermined responses in order,
since evaluation strategies chain 2-3 internal LLM calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

from ai_pipelines.context import PipelineContext
from ai_pipelines.errors import LLMError, StepExecutionError
from ai_pipelines.models import EvaluateStep
from ai_pipelines.steps.evaluate import (
    _compute_conciseness,
    _evaluate_context_conciseness,
    _evaluate_context_relevance,
    _evaluate_context_utilization,
    _evaluate_factual_accuracy,
    _evaluate_faithfulness,
    _evaluate_hallucination,
    _evaluate_summarization,
    _llm_structured,
    _normalize_context,
    execute_evaluate,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_sequential_fake(*responses: dict[str, Any]) -> Any:
    """Build a fake llm_fn that yields responses in order across calls.

    Each call to the returned function consumes the next response from
    the list and yields it as a ResultMessage with structured_output.
    """
    call_index = {"i": 0}

    async def fake_query(
        *, prompt: str, options: ClaudeAgentOptions | None = None, **kwargs: Any
    ) -> AsyncIterator[ResultMessage]:
        idx = call_index["i"]
        call_index["i"] += 1

        if idx >= len(responses):
            raise LLMError(
                f"Sequential fake exhausted: expected {len(responses)} calls, "
                f"got call #{idx + 1}"
            )

        yield ResultMessage(
            subtype="success",
            duration_ms=50,
            duration_api_ms=40,
            is_error=False,
            num_turns=1,
            session_id="test-eval",
            total_cost_usd=0.001,
            result=None,
            structured_output=responses[idx],
        )

    return fake_query


def make_error_fake(error_text: str = "Rate limited") -> Any:
    """Build a fake llm_fn that returns an error ResultMessage."""

    async def fake_query(
        *, prompt: str, options: ClaudeAgentOptions | None = None, **kwargs: Any
    ) -> AsyncIterator[ResultMessage]:
        yield ResultMessage(
            subtype="error",
            duration_ms=50,
            duration_api_ms=40,
            is_error=True,
            num_turns=1,
            session_id="test-eval",
            total_cost_usd=0.0,
            result=error_text,
            structured_output=None,
        )

    return fake_query


# ---------------------------------------------------------------------------
# _llm_structured tests
# ---------------------------------------------------------------------------


class TestLlmStructured:
    """Test the internal structured LLM helper."""

    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        data = {"keyphrases": ["AI", "machine learning"]}
        fake = make_sequential_fake(data)

        result = await _llm_structured(
            prompt_text="Extract keyphrases",
            output_schema={"type": "object", "properties": {}},
            model="haiku",
            system_prompt="You extract stuff.",
            llm_fn=fake,
        )
        assert result == data

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self):
        fake = make_error_fake("Service unavailable")

        with pytest.raises(LLMError, match="LLM returned error"):
            await _llm_structured(
                prompt_text="Extract",
                output_schema={"type": "object"},
                model="haiku",
                system_prompt="sys",
                llm_fn=fake,
            )

    @pytest.mark.asyncio
    async def test_no_structured_output_raises(self):
        """When LLM returns nothing parseable, raise LLMError."""

        async def empty_fake(
            *, prompt: str, options: ClaudeAgentOptions | None = None, **kwargs: Any
        ) -> AsyncIterator[ResultMessage]:
            yield ResultMessage(
                subtype="success",
                duration_ms=50,
                duration_api_ms=40,
                is_error=False,
                num_turns=1,
                session_id="test",
                total_cost_usd=0.0,
                result="just text, not json",
                structured_output=None,
            )

        with pytest.raises(LLMError, match="Expected structured dict"):
            await _llm_structured(
                prompt_text="Extract",
                output_schema={"type": "object"},
                model="haiku",
                system_prompt="sys",
                llm_fn=empty_fake,
            )


# ---------------------------------------------------------------------------
# _compute_conciseness tests
# ---------------------------------------------------------------------------


class TestComputeConciseness:
    def test_empty_source(self):
        assert _compute_conciseness("", "anything") == 1.0

    def test_identical_lengths(self):
        # summary == source length: conciseness ~ 0.0
        result = _compute_conciseness("abcdef", "ghijkl")
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_much_shorter_summary(self):
        source = "a" * 1000
        summary = "b" * 100
        result = _compute_conciseness(source, summary)
        assert result == pytest.approx(0.9, abs=0.01)

    def test_summary_longer_than_source(self):
        # min(summary_len, source_len) = source_len, so conciseness ~ 0.0
        result = _compute_conciseness("short", "much longer summary text here")
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_empty_summary(self):
        result = _compute_conciseness("some source text", "")
        assert result == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Summarization strategy tests
# ---------------------------------------------------------------------------


class TestEvaluateSummarization:
    @pytest.mark.asyncio
    async def test_perfect_score(self):
        """All questions answered YES."""
        fake = make_sequential_fake(
            # Step 1: keyphrases
            {"keyphrases": ["AI", "neural networks"]},
            # Step 2: questions
            {
                "questions": [
                    {"keyphrase": "AI", "question": "Does it mention AI?"},
                    {
                        "keyphrase": "neural networks",
                        "question": "Does it mention neural networks?",
                    },
                ]
            },
            # Step 3: answers
            {
                "answers": [
                    {
                        "question": "Does it mention AI?",
                        "answer": "YES",
                        "reasoning": "The summary covers AI.",
                    },
                    {
                        "question": "Does it mention neural networks?",
                        "answer": "YES",
                        "reasoning": "Neural networks are discussed.",
                    },
                ]
            },
        )

        # Use a short summary for higher conciseness
        source = "A" * 1000
        summary = "B" * 100

        result = await _evaluate_summarization(
            source=source, summary=summary, model="haiku", llm_fn=fake
        )

        assert result["qa_score"] == 1.0
        assert result["conciseness"] == pytest.approx(0.9, abs=0.01)
        assert result["score"] == pytest.approx(0.95, abs=0.01)
        assert result["total_questions"] == 2
        assert result["correct_answers"] == 2
        assert len(result["keyphrases"]) == 2
        assert len(result["questions"]) == 2
        assert len(result["answers"]) == 2

    @pytest.mark.asyncio
    async def test_partial_score(self):
        """Some questions answered NO."""
        fake = make_sequential_fake(
            {"keyphrases": ["climate", "temperature", "CO2", "ice caps"]},
            {
                "questions": [
                    {"keyphrase": "climate", "question": "Mentions climate?"},
                    {"keyphrase": "temperature", "question": "Mentions temp?"},
                    {"keyphrase": "CO2", "question": "Mentions CO2?"},
                    {"keyphrase": "ice caps", "question": "Mentions ice?"},
                ]
            },
            {
                "answers": [
                    {"question": "Mentions climate?", "answer": "YES", "reasoning": "Yes"},
                    {"question": "Mentions temp?", "answer": "NO", "reasoning": "No"},
                    {"question": "Mentions CO2?", "answer": "YES", "reasoning": "Yes"},
                    {"question": "Mentions ice?", "answer": "NO", "reasoning": "No"},
                ]
            },
        )

        result = await _evaluate_summarization(
            source="x" * 200, summary="y" * 200, model="haiku", llm_fn=fake
        )

        assert result["qa_score"] == 0.5
        assert result["correct_answers"] == 2
        assert result["total_questions"] == 4

    @pytest.mark.asyncio
    async def test_empty_keyphrases(self):
        """No keyphrases extracted: short-circuits with score 0."""
        fake = make_sequential_fake({"keyphrases": []})

        result = await _evaluate_summarization(
            source="source", summary="s", model="haiku", llm_fn=fake
        )

        assert result["score"] == 0.0
        assert result["qa_score"] == 0.0
        assert result["total_questions"] == 0
        assert result["keyphrases"] == []
        assert result["questions"] == []
        assert result["answers"] == []

    @pytest.mark.asyncio
    async def test_empty_questions(self):
        """Keyphrases found but no questions generated."""
        fake = make_sequential_fake(
            {"keyphrases": ["AI"]},
            {"questions": []},
        )

        result = await _evaluate_summarization(
            source="source", summary="s", model="haiku", llm_fn=fake
        )

        assert result["score"] == 0.0
        assert result["total_questions"] == 0
        assert result["keyphrases"] == ["AI"]

    @pytest.mark.asyncio
    async def test_llm_error_on_first_call(self):
        """LLM error on keyphrase extraction propagates."""
        fake = make_error_fake("overloaded")

        with pytest.raises(LLMError):
            await _evaluate_summarization(
                source="src", summary="sum", model="haiku", llm_fn=fake
            )


# ---------------------------------------------------------------------------
# Faithfulness strategy tests
# ---------------------------------------------------------------------------


class TestEvaluateFaithfulness:
    @pytest.mark.asyncio
    async def test_all_supported(self):
        """All claims supported by source."""
        fake = make_sequential_fake(
            # Step 1: claims
            {
                "claims": [
                    {"claim": "The Earth orbits the Sun.", "original_sentence": "..."},
                    {"claim": "Water boils at 100C.", "original_sentence": "..."},
                ]
            },
            # Step 2: NLI
            {
                "verdicts": [
                    {"claim": "The Earth orbits the Sun.", "verdict": 1, "reasoning": "Supported."},
                    {"claim": "Water boils at 100C.", "verdict": 1, "reasoning": "Supported."},
                ]
            },
        )

        result = await _evaluate_faithfulness(
            source="The Earth orbits the Sun. Water boils at 100C.",
            response="The Earth orbits the Sun and water boils at 100C.",
            model="haiku",
            llm_fn=fake,
        )

        assert result["score"] == 1.0
        assert result["supported_claims"] == 2
        assert result["total_claims"] == 2

    @pytest.mark.asyncio
    async def test_partial_support(self):
        """Some claims unsupported."""
        fake = make_sequential_fake(
            {
                "claims": [
                    {"claim": "Claim A.", "original_sentence": "..."},
                    {"claim": "Claim B.", "original_sentence": "..."},
                    {"claim": "Claim C.", "original_sentence": "..."},
                ]
            },
            {
                "verdicts": [
                    {"claim": "Claim A.", "verdict": 1, "reasoning": "Yes"},
                    {"claim": "Claim B.", "verdict": 0, "reasoning": "Not found"},
                    {"claim": "Claim C.", "verdict": 1, "reasoning": "Yes"},
                ]
            },
        )

        result = await _evaluate_faithfulness(
            source="source", response="response", model="haiku", llm_fn=fake
        )

        assert result["score"] == pytest.approx(0.6667, abs=0.01)
        assert result["supported_claims"] == 2
        assert result["total_claims"] == 3

    @pytest.mark.asyncio
    async def test_no_claims(self):
        """No claims extracted: fully faithful by default."""
        fake = make_sequential_fake({"claims": []})

        result = await _evaluate_faithfulness(
            source="source", response="ok", model="haiku", llm_fn=fake
        )

        assert result["score"] == 1.0
        assert result["total_claims"] == 0
        assert result["claims"] == []
        assert result["verdicts"] == []

    @pytest.mark.asyncio
    async def test_all_unsupported(self):
        """Every claim contradicted."""
        fake = make_sequential_fake(
            {
                "claims": [
                    {"claim": "The moon is made of cheese.", "original_sentence": "..."},
                ]
            },
            {
                "verdicts": [
                    {"claim": "The moon is made of cheese.", "verdict": 0, "reasoning": "Nope"},
                ]
            },
        )

        result = await _evaluate_faithfulness(
            source="source", response="response", model="haiku", llm_fn=fake
        )

        assert result["score"] == 0.0
        assert result["supported_claims"] == 0
        assert result["total_claims"] == 1

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self):
        fake = make_error_fake()

        with pytest.raises(LLMError):
            await _evaluate_faithfulness(
                source="src", response="resp", model="haiku", llm_fn=fake
            )


# ---------------------------------------------------------------------------
# _normalize_context tests
# ---------------------------------------------------------------------------


class TestNormalizeContext:
    def test_string_passthrough(self):
        assert _normalize_context("plain text") == "plain text"

    def test_list_joined_with_markers(self):
        result = _normalize_context(["doc one", "doc two"])
        assert "[Document 1]\ndoc one" in result
        assert "[Document 2]\ndoc two" in result
        assert result == "[Document 1]\ndoc one\n\n[Document 2]\ndoc two"

    def test_empty_list(self):
        assert _normalize_context([]) == ""

    def test_single_item_list(self):
        result = _normalize_context(["only one"])
        assert result == "[Document 1]\nonly one"

    def test_non_string_items_coerced(self):
        """List items that aren't strings get str() applied."""
        result = _normalize_context([42, {"key": "val"}])
        assert "[Document 1]\n42" in result
        assert "[Document 2]" in result

    def test_non_string_non_list_coerced(self):
        assert _normalize_context(123) == "123"


# ---------------------------------------------------------------------------
# Hallucination strategy tests
# ---------------------------------------------------------------------------


class TestEvaluateHallucination:
    @pytest.mark.asyncio
    async def test_no_contradictions(self):
        """All claims supported or neutral: score 1.0."""
        fake = make_sequential_fake(
            {
                "claims": [
                    {"claim": "The sky is blue.", "original_sentence": "..."},
                    {"claim": "Grass is green.", "original_sentence": "..."},
                ]
            },
            {
                "verdicts": [
                    {"claim": "The sky is blue.", "verdict": "supported", "reasoning": "Yes"},
                    {"claim": "Grass is green.", "verdict": "neutral", "reasoning": "Not mentioned"},
                ]
            },
        )

        result = await _evaluate_hallucination(
            context="The sky is blue on clear days.",
            response="The sky is blue. Grass is green.",
            model="haiku",
            llm_fn=fake,
        )

        assert result["score"] == 1.0
        assert result["contradicted_claims"] == 0
        assert result["total_claims"] == 2

    @pytest.mark.asyncio
    async def test_partial_contradictions(self):
        """Some claims contradicted: score < 1.0."""
        fake = make_sequential_fake(
            {
                "claims": [
                    {"claim": "A is true.", "original_sentence": "..."},
                    {"claim": "B is false.", "original_sentence": "..."},
                    {"claim": "C is maybe.", "original_sentence": "..."},
                ]
            },
            {
                "verdicts": [
                    {"claim": "A is true.", "verdict": "supported", "reasoning": "Yes"},
                    {"claim": "B is false.", "verdict": "contradicted", "reasoning": "Nope"},
                    {"claim": "C is maybe.", "verdict": "neutral", "reasoning": "Unclear"},
                ]
            },
        )

        result = await _evaluate_hallucination(
            context="context", response="response", model="haiku", llm_fn=fake
        )

        assert result["score"] == pytest.approx(0.6667, abs=0.01)
        assert result["contradicted_claims"] == 1
        assert result["total_claims"] == 3

    @pytest.mark.asyncio
    async def test_all_contradicted(self):
        """Everything contradicted: score 0.0."""
        fake = make_sequential_fake(
            {
                "claims": [
                    {"claim": "X.", "original_sentence": "..."},
                    {"claim": "Y.", "original_sentence": "..."},
                ]
            },
            {
                "verdicts": [
                    {"claim": "X.", "verdict": "contradicted", "reasoning": "Wrong"},
                    {"claim": "Y.", "verdict": "contradicted", "reasoning": "Wrong"},
                ]
            },
        )

        result = await _evaluate_hallucination(
            context="ctx", response="resp", model="haiku", llm_fn=fake
        )

        assert result["score"] == 0.0
        assert result["contradicted_claims"] == 2

    @pytest.mark.asyncio
    async def test_neutral_not_penalized(self):
        """Claims marked 'neutral' do not reduce the score."""
        fake = make_sequential_fake(
            {
                "claims": [
                    {"claim": "Claim A.", "original_sentence": "..."},
                    {"claim": "Claim B.", "original_sentence": "..."},
                ]
            },
            {
                "verdicts": [
                    {"claim": "Claim A.", "verdict": "neutral", "reasoning": "Not in context"},
                    {"claim": "Claim B.", "verdict": "neutral", "reasoning": "Not in context"},
                ]
            },
        )

        result = await _evaluate_hallucination(
            context="unrelated context", response="resp", model="haiku", llm_fn=fake
        )

        assert result["score"] == 1.0
        assert result["contradicted_claims"] == 0

    @pytest.mark.asyncio
    async def test_no_claims(self):
        """No claims extracted: nothing to contradict, score 1.0."""
        fake = make_sequential_fake({"claims": []})

        result = await _evaluate_hallucination(
            context="ctx", response="resp", model="haiku", llm_fn=fake
        )

        assert result["score"] == 1.0
        assert result["total_claims"] == 0
        assert result["claims"] == []
        assert result["verdicts"] == []

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self):
        fake = make_error_fake()

        with pytest.raises(LLMError):
            await _evaluate_hallucination(
                context="ctx", response="resp", model="haiku", llm_fn=fake
            )

    @pytest.mark.asyncio
    async def test_list_context_normalized(self):
        """Context as a list of strings is joined with document markers."""
        fake = make_sequential_fake(
            {"claims": [{"claim": "Fact.", "original_sentence": "..."}]},
            {
                "verdicts": [
                    {"claim": "Fact.", "verdict": "supported", "reasoning": "Ok"}
                ]
            },
        )

        result = await _evaluate_hallucination(
            context=["doc one", "doc two"],
            response="Fact.",
            model="haiku",
            llm_fn=fake,
        )

        assert result["score"] == 1.0


# ---------------------------------------------------------------------------
# Context Relevance strategy tests
# ---------------------------------------------------------------------------


class TestEvaluateContextRelevance:
    @pytest.mark.asyncio
    async def test_fully_relevant(self):
        fake = make_sequential_fake(
            {"verdict": "full", "reasoning": "The context fully answers the question."}
        )

        result = await _evaluate_context_relevance(
            question="What is X?", context="X is defined as...", model="haiku", llm_fn=fake
        )

        assert result["score"] == 1.0
        assert result["verdict"] == "full"
        assert result["reasoning"] != ""

    @pytest.mark.asyncio
    async def test_partially_relevant(self):
        fake = make_sequential_fake(
            {"verdict": "partial", "reasoning": "Some info available."}
        )

        result = await _evaluate_context_relevance(
            question="What is X?", context="Tangential info.", model="haiku", llm_fn=fake
        )

        assert result["score"] == 0.5
        assert result["verdict"] == "partial"

    @pytest.mark.asyncio
    async def test_not_relevant(self):
        fake = make_sequential_fake(
            {"verdict": "none", "reasoning": "Completely unrelated."}
        )

        result = await _evaluate_context_relevance(
            question="What is X?", context="About cooking.", model="haiku", llm_fn=fake
        )

        assert result["score"] == 0.0
        assert result["verdict"] == "none"

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self):
        fake = make_error_fake()

        with pytest.raises(LLMError):
            await _evaluate_context_relevance(
                question="Q?", context="C", model="haiku", llm_fn=fake
            )


# ---------------------------------------------------------------------------
# Context Utilization strategy tests
# ---------------------------------------------------------------------------


class TestEvaluateContextUtilization:
    @pytest.mark.asyncio
    async def test_full_utilization(self):
        fake = make_sequential_fake(
            {"verdict": "full", "reasoning": "All context info used."}
        )

        result = await _evaluate_context_utilization(
            question="Q?", context="Ctx", response="Resp", model="haiku", llm_fn=fake
        )

        assert result["score"] == 1.0
        assert result["verdict"] == "full"

    @pytest.mark.asyncio
    async def test_partial_utilization(self):
        fake = make_sequential_fake(
            {"verdict": "partial", "reasoning": "Some context missed."}
        )

        result = await _evaluate_context_utilization(
            question="Q?", context="Ctx", response="Resp", model="haiku", llm_fn=fake
        )

        assert result["score"] == 0.5

    @pytest.mark.asyncio
    async def test_no_utilization(self):
        fake = make_sequential_fake(
            {"verdict": "none", "reasoning": "Context completely ignored."}
        )

        result = await _evaluate_context_utilization(
            question="Q?", context="Ctx", response="Resp", model="haiku", llm_fn=fake
        )

        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self):
        fake = make_error_fake()

        with pytest.raises(LLMError):
            await _evaluate_context_utilization(
                question="Q?", context="C", response="R", model="haiku", llm_fn=fake
            )


# ---------------------------------------------------------------------------
# Context Conciseness strategy tests
# ---------------------------------------------------------------------------


class TestEvaluateContextConciseness:
    @pytest.mark.asyncio
    async def test_retains_all(self):
        fake = make_sequential_fake(
            {"verdict": "full", "reasoning": "All info retained."}
        )

        result = await _evaluate_context_conciseness(
            question="Q?",
            context="Long context with details.",
            concise_context="Short but complete.",
            model="haiku",
            llm_fn=fake,
        )

        assert result["score"] == 1.0
        assert result["verdict"] == "full"

    @pytest.mark.asyncio
    async def test_retains_some(self):
        fake = make_sequential_fake(
            {"verdict": "partial", "reasoning": "Lost some details."}
        )

        result = await _evaluate_context_conciseness(
            question="Q?",
            context="Full context.",
            concise_context="Partial.",
            model="haiku",
            llm_fn=fake,
        )

        assert result["score"] == 0.5

    @pytest.mark.asyncio
    async def test_loses_info(self):
        fake = make_sequential_fake(
            {"verdict": "none", "reasoning": "Most info lost."}
        )

        result = await _evaluate_context_conciseness(
            question="Q?",
            context="Full context.",
            concise_context="Empty.",
            model="haiku",
            llm_fn=fake,
        )

        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self):
        fake = make_error_fake()

        with pytest.raises(LLMError):
            await _evaluate_context_conciseness(
                question="Q?", context="C", concise_context="CC",
                model="haiku", llm_fn=fake,
            )


# ---------------------------------------------------------------------------
# Factual Accuracy strategy tests
# ---------------------------------------------------------------------------


class TestEvaluateFactualAccuracy:
    @pytest.mark.asyncio
    async def test_all_accurate(self):
        """All facts verified as yes: score 1.0."""
        fake = make_sequential_fake(
            {"facts": ["Earth orbits the Sun.", "Water is H2O."]},
            {
                "verdicts": [
                    {"fact": "Earth orbits the Sun.", "verdict": "yes", "reasoning": "Confirmed."},
                    {"fact": "Water is H2O.", "verdict": "yes", "reasoning": "Confirmed."},
                ]
            },
        )

        result = await _evaluate_factual_accuracy(
            question="Science facts?",
            context="Earth orbits the Sun. Water is H2O.",
            response="Earth orbits the Sun. Water is H2O.",
            model="haiku",
            llm_fn=fake,
        )

        assert result["score"] == 1.0
        assert len(result["facts"]) == 2
        assert len(result["verdicts"]) == 2

    @pytest.mark.asyncio
    async def test_mixed_accuracy(self):
        """Mix of yes/unclear/no: score is mean of per-fact scores."""
        fake = make_sequential_fake(
            {"facts": ["Fact A.", "Fact B.", "Fact C."]},
            {
                "verdicts": [
                    {"fact": "Fact A.", "verdict": "yes", "reasoning": "Ok"},
                    {"fact": "Fact B.", "verdict": "unclear", "reasoning": "Maybe"},
                    {"fact": "Fact C.", "verdict": "no", "reasoning": "Wrong"},
                ]
            },
        )

        result = await _evaluate_factual_accuracy(
            question="Q?", context="C", response="R", model="haiku", llm_fn=fake
        )

        # (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert result["score"] == 0.5

    @pytest.mark.asyncio
    async def test_no_facts(self):
        """No facts extracted: nothing to verify, score 1.0."""
        fake = make_sequential_fake({"facts": []})

        result = await _evaluate_factual_accuracy(
            question="Q?", context="C", response="R", model="haiku", llm_fn=fake
        )

        assert result["score"] == 1.0
        assert result["facts"] == []
        assert result["verdicts"] == []

    @pytest.mark.asyncio
    async def test_all_no_verdicts(self):
        """All facts unsupported: score 0.0."""
        fake = make_sequential_fake(
            {"facts": ["Bad fact."]},
            {
                "verdicts": [
                    {"fact": "Bad fact.", "verdict": "no", "reasoning": "Nope"},
                ]
            },
        )

        result = await _evaluate_factual_accuracy(
            question="Q?", context="C", response="R", model="haiku", llm_fn=fake
        )

        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self):
        fake = make_error_fake()

        with pytest.raises(LLMError):
            await _evaluate_factual_accuracy(
                question="Q?", context="C", response="R", model="haiku", llm_fn=fake
            )

    @pytest.mark.asyncio
    async def test_list_context_normalized(self):
        """Context as list works end to end."""
        fake = make_sequential_fake(
            {"facts": ["Fact from doc."]},
            {
                "verdicts": [
                    {"fact": "Fact from doc.", "verdict": "yes", "reasoning": "Found in doc 1"},
                ]
            },
        )

        result = await _evaluate_factual_accuracy(
            question="Q?",
            context=["document one text", "document two text"],
            response="Fact from doc.",
            model="haiku",
            llm_fn=fake,
        )

        assert result["score"] == 1.0


# ---------------------------------------------------------------------------
# execute_evaluate integration tests
# ---------------------------------------------------------------------------


class TestExecuteEvaluate:
    @pytest.mark.asyncio
    async def test_summarization_full_flow(self):
        """Full flow through execute_evaluate with summarization strategy."""
        fake = make_sequential_fake(
            {"keyphrases": ["topic A"]},
            {"questions": [{"keyphrase": "topic A", "question": "Covers A?"}]},
            {
                "answers": [
                    {"question": "Covers A?", "answer": "YES", "reasoning": "Yes"}
                ]
            },
        )

        step = EvaluateStep(
            kind="evaluate",
            name="eval_summary",
            arguments='{"source": input.paper, "summary": input.summary}',
            strategy="summarization",
            model="haiku",
        )

        ctx = PipelineContext({"paper": "A" * 500, "summary": "B" * 50})
        result = await execute_evaluate(step, ctx, llm_fn=fake)

        assert "score" in result
        assert "qa_score" in result
        assert "conciseness" in result
        assert result["qa_score"] == 1.0

    @pytest.mark.asyncio
    async def test_faithfulness_full_flow(self):
        """Full flow through execute_evaluate with faithfulness strategy."""
        fake = make_sequential_fake(
            {
                "claims": [
                    {"claim": "X is true.", "original_sentence": "X is true."}
                ]
            },
            {
                "verdicts": [
                    {"claim": "X is true.", "verdict": 1, "reasoning": "Supported"}
                ]
            },
        )

        step = EvaluateStep(
            kind="evaluate",
            name="eval_faith",
            arguments='{"source": input.paper, "response": input.output}',
            strategy="faithfulness",
            model="haiku",
        )

        ctx = PipelineContext({"paper": "X is true.", "output": "X is true."})
        result = await execute_evaluate(step, ctx, llm_fn=fake)

        assert result["score"] == 1.0
        assert result["total_claims"] == 1

    @pytest.mark.asyncio
    async def test_missing_source_key(self):
        """Arguments missing 'source' raises StepExecutionError."""
        step = EvaluateStep(
            kind="evaluate",
            name="bad",
            arguments='{"summary": input.text}',
            strategy="summarization",
            model="haiku",
        )

        ctx = PipelineContext({"text": "hello"})
        with pytest.raises(StepExecutionError, match="requires key 'source'"):
            await execute_evaluate(step, ctx)

    @pytest.mark.asyncio
    async def test_missing_summary_key(self):
        """Summarization strategy missing 'summary' raises StepExecutionError."""
        step = EvaluateStep(
            kind="evaluate",
            name="bad",
            arguments='{"source": input.text}',
            strategy="summarization",
            model="haiku",
        )

        ctx = PipelineContext({"text": "hello"})
        with pytest.raises(StepExecutionError, match="requires key 'summary'"):
            await execute_evaluate(step, ctx)

    @pytest.mark.asyncio
    async def test_missing_response_key(self):
        """Faithfulness strategy missing 'response' raises StepExecutionError."""
        step = EvaluateStep(
            kind="evaluate",
            name="bad",
            arguments='{"source": input.text}',
            strategy="faithfulness",
            model="haiku",
        )

        ctx = PipelineContext({"text": "hello"})
        with pytest.raises(StepExecutionError, match="requires key 'response'"):
            await execute_evaluate(step, ctx)

    @pytest.mark.asyncio
    async def test_non_dict_arguments_raises(self):
        """Arguments that resolve to non-dict raise StepExecutionError."""
        step = EvaluateStep(
            kind="evaluate",
            name="bad",
            arguments="input.text",
            strategy="summarization",
            model="haiku",
        )

        ctx = PipelineContext({"text": "just a string"})
        with pytest.raises(StepExecutionError, match="must produce a dict"):
            await execute_evaluate(step, ctx)

    @pytest.mark.asyncio
    async def test_references_prior_step_results(self):
        """Arguments can reference prior step results via JSONata."""
        fake = make_sequential_fake(
            {"keyphrases": ["test"]},
            {"questions": [{"keyphrase": "test", "question": "Q?"}]},
            {"answers": [{"question": "Q?", "answer": "YES", "reasoning": "ok"}]},
        )

        step = EvaluateStep(
            kind="evaluate",
            name="eval",
            arguments='{"source": paper, "summary": summary.executive_summary}',
            strategy="summarization",
            model="haiku",
        )

        ctx = PipelineContext({})
        ctx.set_result("paper", "A" * 200)
        ctx.set_result("summary", {"executive_summary": "B" * 20})
        result = await execute_evaluate(step, ctx, llm_fn=fake)

        assert "score" in result

    @pytest.mark.asyncio
    async def test_hallucination_full_flow(self):
        """Full flow through execute_evaluate with hallucination strategy."""
        fake = make_sequential_fake(
            {"claims": [{"claim": "X.", "original_sentence": "X."}]},
            {"verdicts": [{"claim": "X.", "verdict": "supported", "reasoning": "Ok"}]},
        )

        step = EvaluateStep(
            kind="evaluate",
            name="eval_halluc",
            arguments='{"context": input.ctx, "response": input.resp}',
            strategy="hallucination",
            model="haiku",
        )

        ctx = PipelineContext({"ctx": "X is true.", "resp": "X."})
        result = await execute_evaluate(step, ctx, llm_fn=fake)

        assert result["score"] == 1.0
        assert result["contradicted_claims"] == 0

    @pytest.mark.asyncio
    async def test_context_relevance_full_flow(self):
        """Full flow through execute_evaluate with context_relevance strategy."""
        fake = make_sequential_fake(
            {"verdict": "full", "reasoning": "Relevant."}
        )

        step = EvaluateStep(
            kind="evaluate",
            name="eval_rel",
            arguments='{"question": input.q, "context": input.ctx}',
            strategy="context_relevance",
            model="haiku",
        )

        ctx = PipelineContext({"q": "What is X?", "ctx": "X is defined..."})
        result = await execute_evaluate(step, ctx, llm_fn=fake)

        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_context_utilization_full_flow(self):
        fake = make_sequential_fake(
            {"verdict": "partial", "reasoning": "Some used."}
        )

        step = EvaluateStep(
            kind="evaluate",
            name="eval_util",
            arguments='{"question": input.q, "context": input.ctx, "response": input.resp}',
            strategy="context_utilization",
            model="haiku",
        )

        ctx = PipelineContext({"q": "Q?", "ctx": "Context.", "resp": "Response."})
        result = await execute_evaluate(step, ctx, llm_fn=fake)

        assert result["score"] == 0.5

    @pytest.mark.asyncio
    async def test_factual_accuracy_full_flow(self):
        fake = make_sequential_fake(
            {"facts": ["Fact one."]},
            {"verdicts": [{"fact": "Fact one.", "verdict": "yes", "reasoning": "Ok"}]},
        )

        step = EvaluateStep(
            kind="evaluate",
            name="eval_fact",
            arguments='{"question": input.q, "context": input.ctx, "response": input.resp}',
            strategy="factual_accuracy",
            model="haiku",
        )

        ctx = PipelineContext({"q": "Q?", "ctx": "Fact one.", "resp": "Fact one."})
        result = await execute_evaluate(step, ctx, llm_fn=fake)

        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_context_conciseness_full_flow(self):
        fake = make_sequential_fake(
            {"verdict": "full", "reasoning": "All retained."}
        )

        step = EvaluateStep(
            kind="evaluate",
            name="eval_conc",
            arguments='{"question": input.q, "context": input.ctx, "concise_context": input.cc}',
            strategy="context_conciseness",
            model="haiku",
        )

        ctx = PipelineContext({"q": "Q?", "ctx": "Full context.", "cc": "Short."})
        result = await execute_evaluate(step, ctx, llm_fn=fake)

        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_missing_context_key_hallucination(self):
        step = EvaluateStep(
            kind="evaluate",
            name="bad",
            arguments='{"response": input.text}',
            strategy="hallucination",
            model="haiku",
        )

        ctx = PipelineContext({"text": "hello"})
        with pytest.raises(StepExecutionError, match="requires key 'context'"):
            await execute_evaluate(step, ctx)

    @pytest.mark.asyncio
    async def test_missing_question_key_context_relevance(self):
        step = EvaluateStep(
            kind="evaluate",
            name="bad",
            arguments='{"context": input.text}',
            strategy="context_relevance",
            model="haiku",
        )

        ctx = PipelineContext({"text": "hello"})
        with pytest.raises(StepExecutionError, match="requires key 'question'"):
            await execute_evaluate(step, ctx)

    @pytest.mark.asyncio
    async def test_missing_concise_context_key(self):
        step = EvaluateStep(
            kind="evaluate",
            name="bad",
            arguments='{"question": input.q, "context": input.ctx}',
            strategy="context_conciseness",
            model="haiku",
        )

        ctx = PipelineContext({"q": "Q?", "ctx": "C"})
        with pytest.raises(StepExecutionError, match="requires key 'concise_context'"):
            await execute_evaluate(step, ctx)

    @pytest.mark.asyncio
    async def test_context_as_list_works(self):
        """Context passed as a list of strings works end-to-end."""
        fake = make_sequential_fake(
            {"verdict": "full", "reasoning": "All docs relevant."}
        )

        step = EvaluateStep(
            kind="evaluate",
            name="eval_list_ctx",
            arguments='{"question": input.q, "context": input.docs}',
            strategy="context_relevance",
            model="haiku",
        )

        ctx = PipelineContext({"q": "Q?", "docs": ["doc A", "doc B"]})
        result = await execute_evaluate(step, ctx, llm_fn=fake)

        assert result["score"] == 1.0


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestEvaluateStepModel:
    def test_valid_summarization(self):
        step = EvaluateStep(
            kind="evaluate",
            name="eval",
            arguments='{"source": input.text, "summary": input.summary}',
            strategy="summarization",
        )
        assert step.model == "haiku"  # default

    def test_valid_faithfulness(self):
        step = EvaluateStep(
            kind="evaluate",
            name="eval",
            arguments='{"source": input.text, "response": input.response}',
            strategy="faithfulness",
            model="sonnet",
        )
        assert step.model == "sonnet"

    def test_valid_hallucination(self):
        step = EvaluateStep(
            kind="evaluate",
            name="eval",
            arguments='{"context": input.ctx, "response": input.resp}',
            strategy="hallucination",
        )
        assert step.strategy == "hallucination"

    def test_valid_context_relevance(self):
        step = EvaluateStep(
            kind="evaluate",
            name="eval",
            arguments='{"question": input.q, "context": input.ctx}',
            strategy="context_relevance",
        )
        assert step.strategy == "context_relevance"

    def test_valid_context_utilization(self):
        step = EvaluateStep(
            kind="evaluate",
            name="eval",
            arguments='{"question": input.q, "context": input.ctx, "response": input.r}',
            strategy="context_utilization",
        )
        assert step.strategy == "context_utilization"

    def test_valid_factual_accuracy(self):
        step = EvaluateStep(
            kind="evaluate",
            name="eval",
            arguments='{"question": input.q, "context": input.ctx, "response": input.r}',
            strategy="factual_accuracy",
        )
        assert step.strategy == "factual_accuracy"

    def test_valid_context_conciseness(self):
        step = EvaluateStep(
            kind="evaluate",
            name="eval",
            arguments='{"question": input.q, "context": input.ctx, "concise_context": input.cc}',
            strategy="context_conciseness",
        )
        assert step.strategy == "context_conciseness"

    def test_invalid_strategy_rejected(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            EvaluateStep(
                kind="evaluate",
                name="eval",
                arguments='{"source": input.text}',
                strategy="bogus",  # type: ignore[arg-type]
            )

    def test_arguments_required(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            EvaluateStep(
                kind="evaluate",
                name="eval",
                strategy="summarization",
            )  # type: ignore[call-arg]
