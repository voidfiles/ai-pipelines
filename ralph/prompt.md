Your task is to iteratively improve the summarization pipeline at ralph/summarize_paper.yaml so it scores as high as possible on the evaluation pipeline at ralph/evaluate_summary.yaml.

ONLY modify ralph/summarize_paper.yaml. Do not touch the evaluation pipeline.

## Context

The pipeline summarizes academic papers. It is evaluated on:
1. **QA score** (weight 0.25 of overall): Keyphrases are extracted from the source paper, turned into yes/no questions, then checked against the summary. score = correct / total.
2. **Conciseness** (weight 0.25 of overall): How much shorter the summary is vs the source.
3. **Faithfulness** (weight 0.50 of overall): Summary is decomposed into atomic claims, each checked via NLI against the source. score = supported / total.

overall = (qa_score * 0.5 + conciseness * 0.5 + faithfulness) / 2

## Current baseline: overall 0.902

Weaknesses identified from the last evaluation run at e2e/evaluation.json:
- QA: 17/20. Missed questions about author affiliation (UNC Chapel Hill), David Parnas citation, and precise designer-vs-manager comparison. The summary lacks specific named
entities and precise attributions.
- Conciseness: 0.817. The summary is verbose. Shorter = better.
- Faithfulness: 99/102 claims. Three claims failed because the summary used absolute language ('eliminated accidental difficulties') where the paper used hedged language
('attack', 'remove much of'). Every word must be defensible from the source text.

## How to run

1. Copy the pipeline: cp ralph/summarize_paper.yaml e2e/summarize_paper.yaml
2. Run summarization: uv run python e2e/run.py
3. Run evaluation: uv run python e2e/run_evaluate.py
4. Read the scorecard from e2e/evaluation.json

## Strategy: research, then iterate

### Phase 1: Research (first iteration only)
Search the web for these specific techniques and apply the best ideas:
- 'Chain of Density' summarization prompting (Adams et al. 2023)
- Faithful summarization prompts that minimize hallucination
- Extract-then-abstract vs map-reduce summarization tradeoffs
- Prompt techniques for maximizing keyphrase coverage while staying concise

### Phase 2: Improve the pipeline (every iteration)
Read ralph/summarize_paper.yaml and e2e/evaluation.json, then make targeted improvements. Key levers:

**Prompt improvements (high impact):**
- System prompts should demand hedged, source-faithful language. Never overstate. Use the paper's own phrasing.
- Extraction prompts should capture named entities (people, institutions, specific examples) not just abstract concepts.
- The synthesis prompt should explicitly instruct: include author full name with affiliation, all cited researchers by name, specific comparisons made in the paper, and concrete
numbers/statistics.
- Add a 'chain of density' instruction: pack maximum information into minimum words. Every sentence must earn its place.
- Add a faithfulness self-check instruction: 'Before including any claim, verify it is directly supported by the extracted points. Use the source's own hedging language (e.g.,
attack, reduce, address) rather than absolutes (e.g., eliminate, solve, remove).'

**Pipeline structure improvements (medium impact):**
- Consider adjusting chunk_size/overlap for better context per chunk
- Consider using sonnet for extraction too (better at capturing named entities)
- Consider adding a dedicated 'named entity extraction' step
- Consider a 'compress and verify' final step that shortens while checking faithfulness
- Consider reducing the output schema to force conciseness (fewer fields, tighter descriptions)

**What NOT to do:**
- Do not add fields to the output schema (the evaluation expects: executive_summary, core_thesis, main_arguments, key_evidence, conclusions)
- Do not remove required fields
- Do not change the input schema

### Phase 3: Evaluate and analyze
After each run, read e2e/evaluation.json carefully. Look at:
- Which questions were answered NO (missed coverage)
- Which claims got verdict 0 (faithfulness failures)
- The conciseness score trend
- Compare to previous iteration's scores

If a change made things worse, revert it. If a change helped one metric but hurt another, find the right balance.

## Target
- overall >= 0.95
- qa_score >= 0.90
- faithfulness >= 0.98
- conciseness >= 0.85

Output <promise>PIPELINE_OPTIMIZED</promise> when you have achieved overall >= 0.95 OR you have completed 3 consecutive iterations without improvement.