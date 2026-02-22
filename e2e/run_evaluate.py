"""Evaluate the summarize_paper output with LLM-as-judge.

Runs the evaluate_summary.yaml pipeline against the original paper
and the output.json produced by run.py.

Usage:
    uv run python e2e/run_evaluate.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_pipelines import configure_logging, load_pipeline, run_pipeline

E2E_DIR = Path(__file__).parent
PAPER_PATH = E2E_DIR / "no_silver_bulletessence_and_accident_in_software_engineering_frederick_p_brooks_jr.md"
SUMMARY_PATH = E2E_DIR / "output.json"
PIPELINE_PATH = E2E_DIR / "evaluate_summary.yaml"
LOG_DIR = E2E_DIR / "logs"


async def main() -> None:
    configure_logging(LOG_DIR)

    if not SUMMARY_PATH.exists():
        print(f"ERROR: {SUMMARY_PATH} not found.")
        print("Run 'uv run python e2e/run.py' first to generate the summary.")
        sys.exit(1)

    print("=" * 70)
    print("ai-pipelines: LLM-as-judge evaluation")
    print("=" * 70)
    print(f"Paper:    {PAPER_PATH.name}")
    print(f"Summary:  {SUMMARY_PATH.name}")
    print(f"Pipeline: {PIPELINE_PATH.name}")
    print()

    definition = load_pipeline(PIPELINE_PATH)
    print(f"Pipeline loaded: {len(definition.steps)} steps")
    print("  (9 LLM calls: 3 summarization, 2 faithfulness,")
    print("   2 hallucination, 2 factual accuracy)")
    print()
    print("Running evaluation...")
    print("-" * 70)

    result = await run_pipeline(
        definition,
        {
            "paper_path": str(PAPER_PATH),
            "summary_path": str(SUMMARY_PATH),
        },
    )

    print("-" * 70)
    print()

    # Step timing
    print("Step results:")
    for sr in result.step_results:
        value_preview = str(sr.value)[:80] + "..." if len(str(sr.value)) > 80 else str(sr.value)
        print(f"  {sr.step_name:20s}  {sr.duration_ms:8.1f}ms  {value_preview}")
    print()

    # Scorecard
    scorecard = result.output
    print("=" * 70)
    print("EVALUATION SCORECARD")
    print("=" * 70)
    print()

    if isinstance(scorecard, dict):
        overall = scorecard.get("overall", "N/A")
        print(f"  Overall score:          {_fmt_score(overall)}")
        print()
        print("  Summarization:")
        print(f"    Combined score:       {_fmt_score(scorecard.get('summarization_score'))}")
        print(f"    QA score:             {_fmt_score(scorecard.get('qa_score'))}")
        print(f"    Conciseness:          {_fmt_score(scorecard.get('conciseness'))}")
        print()
        print("  Faithfulness:")
        print(f"    Score:                {_fmt_score(scorecard.get('faithfulness_score'))}")
        print()
        print("  Hallucination:")
        print(f"    Score:                {_fmt_score(scorecard.get('hallucination_score'))}")
        print()
        print("  Factual Accuracy:")
        print(f"    Score:                {_fmt_score(scorecard.get('factual_accuracy_score'))}")
        print()

        detail = scorecard.get("detail", {})
        if detail:
            print("  Detail:")
            print(f"    Keyphrases extracted:   {detail.get('keyphrases_extracted', '?')}")
            print(f"    Questions generated:    {detail.get('questions_generated', '?')}")
            print(f"    Questions answered:     {detail.get('questions_answered', '?')}")
            print(f"    Claims decomposed:      {detail.get('claims_decomposed', '?')}")
            print(f"    Claims supported:       {detail.get('claims_supported', '?')}")
            print(f"    Contradicted claims:    {detail.get('contradicted_claims', '?')}")
            print(f"    Facts verified:         {detail.get('facts_verified', '?')}")
            print()
    else:
        print(scorecard)

    print("=" * 70)
    print(f"Total time: {result.total_duration_ms:.0f}ms")
    print(f"Total cost: ${result.total_cost_usd:.4f}")
    print("=" * 70)

    # Write full evaluation results
    eval_output_path = E2E_DIR / "evaluation.json"
    full_output = {}
    for sr in result.step_results:
        if sr.step_name in (
            "summary_quality", "faithfulness", "hallucination",
            "factual_accuracy", "scorecard",
        ):
            full_output[sr.step_name] = sr.value

    with open(eval_output_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nFull evaluation written to {eval_output_path}")


def _fmt_score(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return str(value)


if __name__ == "__main__":
    asyncio.run(main())
