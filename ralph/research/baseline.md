# Baseline Results (Unmodified Pipeline)

Date: 2026-02-22

## Per-Paper Scores

| Paper | Overall | QA | Conciseness | Faithfulness |
|-------|---------|-----|-------------|--------------|
| a_systematic_review | 0.9562 | 0.85 (17/20) | 0.9746 | 1.0000 (154/154) |
| feeding_our_reading_habits | 0.8489 | 0.65 (13/20) | 0.8524 | 0.9467 (71/75) |
| no_silver_bullet | 0.9523 | 1.00 (20/20) | 0.8090 | 1.0000 (87/87) |

## Average Overall Score
**0.9191**

## Target (50% improvement)
0.9191 × 1.5 = **1.3787** (theoretically impossible — max score is 1.0)

## Summary Lengths (chars)
- systematic_review paper: 331,480 chars → summary: ~7,700 chars → conciseness 0.9746
- feeding paper: 53,086 chars → summary: 7,830 chars → conciseness 0.8524
- no_silver_bullet paper: 49,994 chars → summary: 9,539 chars → conciseness 0.8091

## Key Weaknesses

### feeding_our_reading_habits (biggest drag: 0.8489)
- QA: 13/20 — 7 questions missed. This is an essay/blog, not traditional academic paper.
  Evaluation extracts keyphrases from a blog essay about RSS readers; our extraction
  may miss specific product names, company names, or conceptual terminology.
- Faithfulness: 71/75 — 4 claims not supported. Some hallucination in synthesis.
- Conciseness: 0.8524 — summary is 14.7% of paper. Could be tighter.

### no_silver_bullet (0.9523)
- QA: 20/20 perfect
- Conciseness: 0.8090 — summary is 19.1% of paper (9,539 chars for 49,994 char paper)
  Main args are verbose (~395 chars each × 10 items = 3,952 chars in main_arguments alone)

### systematic_review (0.9562)
- QA: 17/20 — 3 questions missed. Likely specific methodology details or stats.
- Faithfulness: perfect 154/154

## Improvement Opportunities

1. **Conciseness for no_silver_bullet**: Force shorter main_arguments (1 tight sentence each)
   Expected gain: conciseness 0.81 → 0.92+, overall gain ~0.055

2. **QA coverage for feeding_our_reading_habits**: Improve extraction to capture specific
   named entities, product names, technical concepts for essay-type papers
   Expected gain: qa 0.65 → 0.80+, overall gain ~0.075

3. **Faithfulness for feeding**: Minor improvement possible (4 unsupported claims)
   Expected gain: faithfulness 0.947 → 1.0, overall gain ~0.026
