"""Smoke test: summarize an academic paper with a real LLM pipeline.

Usage:
    uv run python e2e/run.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for local dev
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_pipelines import configure_logging, load_pipeline, run_pipeline

E2E_DIR = Path(__file__).parent
PAPER_PATH = E2E_DIR / "no_silver_bulletessence_and_accident_in_software_engineering_frederick_p_brooks_jr.md"
PIPELINE_PATH = E2E_DIR / "summarize_paper.yaml"
LOG_DIR = E2E_DIR / "logs"


async def main() -> None:
    configure_logging(LOG_DIR)

    print("=" * 70)
    print("ai-pipelines smoke test: summarize academic paper")
    print("=" * 70)
    print(f"Paper:    {PAPER_PATH.name}")
    print(f"Pipeline: {PIPELINE_PATH.name}")
    print(f"Logs:     {LOG_DIR}")
    print()

    definition = load_pipeline(PIPELINE_PATH)
    print(f"Pipeline loaded: {len(definition.steps)} top-level steps")
    print()

    print("Running pipeline...")
    print("-" * 70)

    result = await run_pipeline(definition, {"paper_path": str(PAPER_PATH)})

    print("-" * 70)
    print()

    # Print step results
    print("Step results:")
    for sr in result.step_results:
        value_preview = str(sr.value)[:80] + "..." if len(str(sr.value)) > 80 else str(sr.value)
        print(f"  {sr.step_name:20s}  {sr.duration_ms:8.1f}ms  {value_preview}")
    print()

    # Print final output
    output = result.output
    print("=" * 70)
    print("PAPER SUMMARY")
    print("=" * 70)
    print()

    if isinstance(output, dict):
        if "executive_summary" in output:
            print("EXECUTIVE SUMMARY")
            print(output["executive_summary"])
            print()

        if "core_thesis" in output:
            print("CORE THESIS")
            print(output["core_thesis"])
            print()

        if "main_arguments" in output:
            print("MAIN ARGUMENTS")
            for i, arg in enumerate(output["main_arguments"], 1):
                print(f"  {i}. {arg}")
            print()

        if "key_evidence" in output:
            print("KEY EVIDENCE")
            for ev in output["key_evidence"]:
                print(f"  - {ev}")
            print()

        if "conclusions" in output:
            print("CONCLUSIONS")
            for c in output["conclusions"]:
                print(f"  - {c}")
            print()
    else:
        print(output)

    print("=" * 70)
    print(f"Total time: {result.total_duration_ms:.0f}ms")
    print(f"Total cost: ${result.total_cost_usd:.4f}")
    print("=" * 70)

    # Write output to file for inspection
    output_path = E2E_DIR / "output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull output written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
