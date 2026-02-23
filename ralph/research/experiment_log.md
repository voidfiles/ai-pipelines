# Experiment Log

## Summary Table

| Experiment | Avg | systematic_review | feeding | no_silver_bullet | Notes |
|-----------|-----|-------------------|---------|-----------------|-------|
| Baseline | 0.9191 | 0.9562 | 0.8489 | 0.9523 | Original pipeline |
| Exp1 | 0.9245 | 0.9645 | 0.8846 | 0.9244 | Named entity extraction + 200-char limits |
| Exp2 | 0.8798 | 0.9163 | 0.8059 | 0.9171 | FAIL: fewer items caused faithfulness drop |
| Exp3 | CRASH | - | - | - | FAIL: section_topic required → crash |
| Exp4 | 0.9254 | 0.9592 | 0.8882 | 0.9286 | Two-step draft+verify (marginal) |
| Exp5 | 0.8787 | 0.8923 | 0.7944 | 0.9492 | FAIL: 220-char+merge destroyed systematic_review QA |
| Exp6 | CRASH | 0.5244* | - | - | FAIL: crash + catastrophic faithfulness for large paper |
| Exp7 | CRASH | - | - | - | FAIL: section_topic crash again |
| Exp8 | 0.8843 | 0.9236 | 0.8562 | 0.8729 | FAIL: verify+remove instruction destroyed QA |
| **Exp9** | **0.9299** | **0.9577** | **0.9128** | **0.9194** | **BEST: exp1 exact + section_topic optional** |

\* Partial result before crash

## Key Lessons

1. **section_topic MUST be optional**: systematic_review data-table chunks cause haiku to omit
   this field. Making it required causes crashes in exp3, exp6, exp7.

2. **Named entity extraction helps QA**: Exp1's addition of named entity capture (+NAMED_ENTITY category)
   improved QA across all papers (feeding: 0.65→0.90, systematic_review: 0.85→0.90).

3. **"Verify and remove" instructions destroy QA**: Exp8 added "BEFORE FINALIZING: verify and remove
   items you cannot trace" — this caused 0.54-0.75 QA scores. Never add such instructions.

4. **Two-step approach fails for large papers**: Systematic_review (331k chars) gets catastrophic
   faithfulness (0.26) when verify step sees 200+ extracted points. Stick to single-step synthesis.

5. **200-char limits are a tradeoff**: Good for conciseness but hurt no_silver_bullet QA (1.0→0.85).
   Increasing to 280 chars hasn't helped so far due to interaction effects.

6. **Merge instructions destroy QA**: Exp5's "merge overlapping items" instruction destroyed
   systematic_review QA (1.0→0.60). Never merge items.

## Best Pipeline (Exp9 = 0.9299)

Key features:
- Extraction: haiku with named entity capture, section_topic optional
- Synthesis: sonnet with 200-char limits per argument, coverage rules
- No two-step, no verify/remove instructions, no merge instructions
- section_topic optional in schema (critical crash prevention)

## Remaining Weaknesses

After exp9:
- systematic_review: qa=0.85 (17/20) — 3 questions missed
- feeding: qa=0.85 (17/20) — 3 questions missed
- no_silver_bullet: qa=0.818 (18/22) — 4 questions missed

## New Experiment Ideas

1. **Slightly relaxed char limits (250-char per argument)**: Might help no_silver_bullet QA
   recover from 0.818 to 0.85+ without the conciseness hit of removing limits entirely.
   Risk: minor QA improvement might not overcome noise in LLM stochasticity.

2. **Increase main_arguments count**: Currently 8-12. Allow up to 14 to cover more QA topics.
   Each argument covers fewer topics per item = less chance of omitting keyphrases.
   Risk: minor conciseness impact.

3. **Improve extraction for specific paper types**: The systematic_review's data tables suggest
   the extraction is already being stressed. Focus extraction on finding thematic content
   even in data-heavy chunks.
