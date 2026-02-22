Your task is to iteratively improve the summarization pipeline at `ralph/summarize_paper.yaml` so it achieves an average score across **all papers in `ralph/papers/*.md`** that is at least **50% better than the baseline average**.

ONLY modify `ralph/summarize_paper.yaml`. Do not touch the evaluation pipeline.

## Task Tracking: Beads (REQUIRED)

Track **all work** — research tasks, coding experiments, analysis — via beads. Do not use TodoWrite or markdown files for task tracking.

```bash
bd create --title="..." --type=task|feature|bug --priority=2
bd update <id> --status=in_progress   # before starting
bd close <id>                          # after finishing
bd ready                               # find next available work
```

Create a beads issue for each planned experiment **before** writing any code. At session end, run `bd sync` and push.

## Directories

Ensure these exist before starting:
```bash
mkdir -p ralph/unsuccessful ralph/research
```

## Context

The pipeline summarizes academic papers. It is evaluated on:
1. **QA score** (weight 0.25 of overall): Keyphrases are extracted from the source paper, turned into yes/no questions, then checked against the summary. score = correct / total.
2. **Conciseness** (weight 0.25 of overall): How much shorter the summary is vs the source.
3. **Faithfulness** (weight 0.50 of overall): Summary is decomposed into atomic claims, each checked via NLI against the source. score = supported / total.

```
overall = (qa_score * 0.5 + conciseness * 0.5 + faithfulness) / 2
```

## How to Run (Multi-Paper)

Prefix all `uv run` commands with `env -u CLAUDECODE` to avoid nested session errors.

The pipeline must be evaluated against **every paper in `ralph/papers/*.md`**. Use this pattern:

```bash
env -u CLAUDECODE uv run python -c "
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, 'src')
from ai_pipelines import configure_logging, load_pipeline, run_pipeline

papers = sorted(Path('ralph/papers').glob('*.md'))
pipeline = load_pipeline(Path('ralph/summarize_paper.yaml'))
eval_pipeline = load_pipeline(Path('ralph/evaluate_summary.yaml'))

async def run_all():
    scores = []
    for paper in papers:
        out_path = Path('ralph') / f'output_{paper.stem}.json'
        result = await run_pipeline(pipeline, {'paper_path': str(paper)})
        with open(out_path, 'w') as f:
            json.dump(result.output, f, indent=2)
        eval_result = await run_pipeline(eval_pipeline, {
            'paper_path': str(paper),
            'summary_path': str(out_path),
        })
        sc = eval_result.output
        scores.append({'paper': paper.name, 'scorecard': sc})
        print(f'{paper.name}: overall={sc.get(\"overall\", 0):.4f}')
    avg = sum(s['scorecard'].get('overall', 0) for s in scores) / len(scores)
    print(f'AVERAGE: {avg:.4f}')
    with open('ralph/results_all.json', 'w') as f:
        json.dump({'scores': scores, 'average': avg}, f, indent=2)
    return avg

asyncio.run(run_all())
"
```

## Phase 0: Establish Baseline (FIRST — DO NOT SKIP)

Before modifying anything:

1. Create a beads issue: `bd create --title="Establish baseline for all papers" --type=task --priority=0`
2. Run the pipeline on all papers with the **current unmodified** `ralph/summarize_paper.yaml`
3. Record results in `ralph/research/baseline.md`:
   - Per-paper scores (qa, conciseness, faithfulness, overall)
   - Average overall score
   - Notable weaknesses per paper
4. Set **target = baseline_average × 1.5** (50% improvement goal)
5. Close the beads issue and commit:
   ```bash
   git add ralph/research/baseline.md ralph/results_all.json
   git commit -m "chore: record baseline scores for all papers"
   ```

## Phase 1: Research

Create a beads issue for each research topic **before** investigating. Store all findings in `ralph/research/`. Commit research files after completing each topic:

```bash
git add ralph/research/
git commit -m "research: <topic>"
```

Research topics (create one beads issue per topic):
- Chain of Density summarization prompting (Adams et al. 2023)
- Faithful summarization prompts that minimize hallucination
- Extract-then-abstract vs map-reduce summarization tradeoffs
- Prompt techniques for maximizing keyphrase coverage while staying concise
- Generalizing summarization across diverse paper domains (CS, social science, literature)

For each topic, write `ralph/research/<topic-slug>.md` with:
- Key findings
- Specific prompt techniques to try
- Expected impact on which metric

After all topics, synthesize into `ralph/research/experiment_plan.md` listing experiments ordered by expected impact.

## Phase 2: Experiment Loop

### Before writing code
1. Create a beads issue: `bd create --title="Experiment: <description>" --type=feature --priority=2`
2. Claim it: `bd update <id> --status=in_progress`
3. Record hypothesis: `bd update <id> --description="Hypothesis: <what you expect and why>"`

### Run the experiment
1. Modify `ralph/summarize_paper.yaml`
2. Run on all papers, record average score in `ralph/results_all.json`
3. Compare to baseline and previous best

### If experiment SUCCEEDS (average improves)
```bash
bd close <id> --reason="improved average from X to Y"
git add ralph/summarize_paper.yaml ralph/results_all.json
git commit -m "experiment: <description> (avg: X -> Y)"
```
Update `ralph/research/experiment_log.md` with the result.

### If experiment FAILS (no improvement or regression)
1. Copy with failure notes:
   ```bash
   cp ralph/summarize_paper.yaml ralph/unsuccessful/<description-slug>.yaml
   ```
   Add a comment block at the **very top** of the copied file:
   ```yaml
   # EXPERIMENT: <description>
   # DATE: <date>
   # HYPOTHESIS: <what was expected>
   # RESULT: average was X (baseline Y, no improvement / regression)
   # FAILURE REASON: <analysis of why it failed>
   # ---
   ```
2. Revert `ralph/summarize_paper.yaml` to last successful version
3. Close the beads issue: `bd close <id> --reason="failed: <brief reason>"`
4. Commit the failure record:
   ```bash
   git add ralph/unsuccessful/
   git commit -m "record: failed experiment — <description>"
   ```

## Key Levers

**Prompt improvements (high impact):**
- System prompts: demand hedged, source-faithful language — never overstate, use paper's own phrasing
- Extraction prompts: capture named entities (people, institutions, examples), not just abstract concepts
- Synthesis prompt: include author full name with affiliation, all cited researchers, specific comparisons, concrete numbers
- Chain of density: pack maximum information into minimum words — every sentence must earn its place
- Faithfulness self-check: "Before including any claim, verify it is directly supported. Use source hedging (attack, reduce, address) not absolutes (eliminate, solve)"

**Pipeline structure (medium impact):**
- Adjust chunk_size/overlap for better per-chunk context
- Use sonnet for extraction (better named entity capture)
- Add dedicated named entity extraction step
- Add compress-and-verify final step
- Reduce output schema field descriptions to force conciseness

**Domain generalization (important for multi-paper):**
- Avoid prompts that assume a CS paper structure
- Extraction should work for social science, literary analysis, and technical papers alike
- Consider a paper-type detection step to adapt extraction prompts

**What NOT to do:**
- Do not add fields to the output schema (evaluation expects: executive_summary, core_thesis, main_arguments, key_evidence, conclusions)
- Do not remove required fields
- Do not change the input schema

## Phase 3: Analyze Each Iteration

After each run, read `ralph/results_all.json`:
- Which papers improved? Which regressed?
- Per-paper weaknesses: missed QA questions, faithfulness failures, verbosity
- Track trends: is the change helping all paper types or only some?

If a change helps one paper but hurts others, investigate and either tune the prompt or create a follow-up beads issue.

After every 2–3 experiments, update `ralph/research/experiment_log.md` with:
- Experiment description and before/after per-paper scores
- What worked and why
- At least 2 new experiment ideas as beads issues

## Session Close Protocol

```bash
bd sync
git add ralph/
git commit -m "wip: <description of session progress>"
bd sync
git push
```

## Target

- Average overall score across all papers >= **baseline_average × 1.5**
- No individual paper drops below 0.80 overall

Output `<promise>PIPELINE_OPTIMIZED</promise>` when the average meets the target, OR after 5 consecutive experiment iterations without improvement.
