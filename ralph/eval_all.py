"""Run summarize_paper.yaml + evaluate_summary.yaml on all papers in ralph/papers/.

Usage:
    env -u CLAUDECODE uv run python ralph/eval_all.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_pipelines import load_pipeline, run_pipeline

RALPH_DIR = Path(__file__).parent
PAPERS_DIR = RALPH_DIR / "papers"
PIPELINE_PATH = RALPH_DIR / "summarize_paper.yaml"
EVAL_PIPELINE_PATH = RALPH_DIR / "evaluate_summary.yaml"
RESULTS_PATH = RALPH_DIR / "results_all.json"


async def main() -> None:
    papers = sorted(PAPERS_DIR.glob("*.md"))
    if not papers:
        print("No papers found in ralph/papers/")
        return

    pipeline = load_pipeline(PIPELINE_PATH)
    eval_pipeline = load_pipeline(EVAL_PIPELINE_PATH)

    scores = []
    for paper in papers:
        out_path = RALPH_DIR / f"output_{paper.stem}.json"
        print(f"\nProcessing: {paper.name}", flush=True)

        result = await run_pipeline(pipeline, {"paper_path": str(paper)})
        with open(out_path, "w") as f:
            json.dump(result.output, f, indent=2)

        eval_result = await run_pipeline(eval_pipeline, {
            "paper_path": str(paper),
            "summary_path": str(out_path),
        })
        sc = eval_result.output
        scores.append({"paper": paper.name, "scorecard": sc})

        overall = sc.get("overall", 0)
        qa = sc.get("qa_score", 0)
        conciseness = sc.get("conciseness", 0)
        faithfulness = sc.get("faithfulness_score", 0)
        print(
            f"  overall={overall:.4f}  qa={qa:.4f}  "
            f"conciseness={conciseness:.4f}  faithfulness={faithfulness:.4f}",
            flush=True,
        )

    avg = sum(s["scorecard"].get("overall", 0) for s in scores) / len(scores)
    print(f"\nAVERAGE: {avg:.4f}", flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"scores": scores, "average": avg}, f, indent=2)
    print(f"Results written to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
