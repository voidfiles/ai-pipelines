# ai-pipelines

YAML-driven AI/LLM pipeline executor. Define multi-step workflows in YAML, run them with a single Python call.

## What it does

- Describe pipelines as YAML files: read files, chunk text, call LLMs, loop, transform data, evaluate outputs
- Steps reference each other by name using JSONata expressions
- LLM prompt templates use Jinja2
- Built-in LLM-as-judge evaluation with 7 scoring strategies
- Returns per-step timing and total cost

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Quick Start

```python
import asyncio
from ai_pipelines import load_pipeline, run_pipeline

async def main():
    pipeline = load_pipeline("my_pipeline.yaml")
    result = await run_pipeline(pipeline, {"text": "Hello world"})
    print(result.output)
    print(f"Cost: ${result.total_cost_usd:.4f}")

asyncio.run(main())
```

### Minimal pipeline YAML

```yaml
input:
  type: object
  properties:
    text: { type: string }
  required: [text]

steps:
  - kind: prompt
    name: summary
    model: haiku
    arguments: '{"text": input.text}'
    template: "Summarize this: {{ args.text }}"
```

## Pipeline YAML Reference

Every pipeline has `input` (JSON Schema) and `steps` (ordered list). Each step requires `kind` and `name`. All prior step results are available by name via JSONata expressions.

### Step types

| kind | What it does | Key fields |
|---|---|---|
| `read_file` | Read file contents as text | `arguments`: JSONata path to filename |
| `find_files` | Glob file discovery | `arguments`: base dir, `pattern`: glob |
| `transform` | Evaluate a JSONata expression | `arguments`: any JSONata expression |
| `chunk` | Split text into overlapping chunks | `arguments`: text source, `chunk_size` (default 4000), `overlap` (default 200) |
| `prompt` | LLM call with Jinja2 template | `model`, `template`, `arguments`, `output` (optional JSON schema), `system_prompt` |
| `for_each` | Loop over an array, run nested steps | `arguments`: array source, `steps`: list |
| `evaluate` | LLM-as-judge scoring | `strategy`, `arguments`, `model` (default haiku) |

### Data flow

- `input.field` — pipeline input fields
- `step_name.field` — prior step output
- Inside `for_each`, `item` is the current element and `item_index` its position
- `prompt` templates use `{{ args.field }}` where `args` is the resolved `arguments` dict

### Structured output

Add `output` with a JSON Schema to a `prompt` step to get a typed dict back:

```yaml
- kind: prompt
  name: result
  model: sonnet
  arguments: '{"doc": input.text}'
  template: "Extract key points from: {{ args.doc }}"
  output:
    type: object
    properties:
      points: { type: array, items: { type: string } }
    required: [points]
```

### Evaluate strategies

| strategy | Required argument keys | Score |
|---|---|---|
| `summarization` | `source`, `summary` | `(qa_score + conciseness) / 2` |
| `faithfulness` | `source`, `response` | `supported_claims / total_claims` |
| `hallucination` | `context`, `response` | `1 - contradicted / total` |
| `factual_accuracy` | `question`, `context`, `response` | mean of per-fact scores |
| `context_relevance` | `question`, `context` | full=1.0 / partial=0.5 / none=0.0 |
| `context_utilization` | `question`, `context`, `response` | full=1.0 / partial=0.5 / none=0.0 |
| `context_conciseness` | `question`, `context`, `concise_context` | full=1.0 / partial=0.5 / none=0.0 |

## Running Tests

```bash
just test
# or
uv run pytest
```

## Project Structure

```
src/ai_pipelines/
  __init__.py         Public API
  models.py           Pydantic models for all step types + discriminated union
  executor.py         Async orchestrator (run_pipeline)
  loader.py           YAML parsing + JSON Schema validation
  context.py          PipelineContext: scoped dict for step results
  expressions.py      JSONata evaluator
  templates.py        Jinja2 renderer
  validator.py        Static pre-flight validation
  errors.py           Exception hierarchy
  pipeline_logger.py  Structured JSON-lines logging
  steps/              One file per step kind
e2e/                  End-to-end example scripts and pipelines
tests/                pytest suite
```

## Public API

```python
from ai_pipelines import (
    configure_logging,
    load_pipeline,
    validate_pipeline,
    load_and_validate_pipeline,
    run_pipeline,
    validate_input,
)
```

All exceptions inherit from `PipelineError`:

| Exception | When raised |
|---|---|
| `PipelineLoadError` | YAML parse failure or bad structure |
| `ValidationError` | Input/output JSON Schema mismatch |
| `ExpressionError` | Invalid or failing JSONata expression |
| `StepExecutionError` | Any step fails at runtime |
| `LLMError` | LLM call fails or returns unparseable output |
