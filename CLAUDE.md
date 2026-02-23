# Claude Code Agent Guide

## Running Pipelines

**CRITICAL: Never run pipeline scripts inside a Claude Code session without `env -u CLAUDECODE`.**

Running `uv run python e2e/run.py` directly causes a nested session error. Always use:

```bash
env -u CLAUDECODE uv run python e2e/run.py
env -u CLAUDECODE uv run python e2e/run_evaluate.py
```

To test a modified ralph pipeline:
```bash
cp ralph/summarize_paper.yaml e2e/summarize_paper.yaml
env -u CLAUDECODE uv run python e2e/run.py
env -u CLAUDECODE uv run python e2e/run_evaluate.py
```

## Tests & Linting

```bash
just test          # run all tests
uv run pytest      # equivalent
just lint          # compile-check package
just sync          # uv sync --all-extras
```

## Task Tracking

Use **bd** (beads), NOT TodoWrite or TaskCreate.

```bash
bd ready                                 # find available work
bd show <id>                             # view issue details
bd update <id> --status in_progress      # claim work
bd close <id>                            # mark complete
bd sync                                  # sync with git
```

## Architecture

`load_pipeline(path)` → `PipelineDefinition` → `run_pipeline(pipeline, input)` → `PipelineResult`

### Key files

| File | Responsibility |
|---|---|
| `models.py` | All Pydantic shapes. No logic. Discriminated union on `kind`. |
| `executor.py` | `run_pipeline` + `execute_step` dispatch via `match/case` on step type. |
| `loader.py` | YAML → `PipelineDefinition`. Raises `PipelineLoadError` on bad input. |
| `context.py` | `PipelineContext`: scoped dict. `child()` creates isolated scope for `for_each`. |
| `expressions.py` | Thin wrapper over `jsonata-python`. All `arguments` fields evaluated here. |
| `templates.py` | Jinja2 with `StrictUndefined`. Variables available as `args`. Missing keys fail immediately. |
| `validator.py` | Static checks: duplicate names, bad JSONata, unresolved references, template/args mismatches. |
| `errors.py` | Exception hierarchy. All errors inherit `PipelineError`. |
| `sdk_patch.py` | Monkey-patches `claude-agent-sdk`. Applied at import time in `__init__.py` before other imports. |
| `steps/prompt.py` | LLM calls. Handles SDK structured output via both `StructuredOutput` tool block and `ResultMessage`. |
| `steps/evaluate.py` | 7 LLM-as-judge strategies dispatched via `_STRATEGIES` dict. |
| `steps/for_each.py` | Iterates array; binds `item` and `item_index` in child context per iteration. |

## Code Conventions

- Python 3.12+. Use `from __future__ import annotations` in every file.
- All step executors: `async def execute_<kind>(step, context) -> Any`.
- Steps don't import from each other — only from `models`, `context`, `expressions`, `errors`.
- `executor.py` is the only file that imports all step executors.
- Models are pure shapes (`models.py`). No business logic.
- JSONata for data access/transformation. Jinja2 only for prompt templates.
- Use `PipelineContext.child()` for loop isolation. Never mutate the parent context inside a loop.
- Inject `llm_fn` for testing LLM steps (see `tests/test_steps/test_prompt.py`).
- In `executor.py`: catch generic exceptions and re-raise as `StepExecutionError`. Let `PipelineError` subclasses pass through unwrapped.

## Adding a New Step Kind

1. Add a Pydantic model with `kind: Literal["your_kind"]` in `models.py`
2. Add it to the `Step` discriminated union in `models.py`
3. Create `steps/your_kind.py` with `async def execute_your_kind(step, context) -> Any`
4. Add a `case YourKindStep():` branch in `executor.py`'s `execute_step`
5. Add tests in `tests/test_steps/test_your_kind.py`

## Common Pitfalls

- **Nested session error**: forgetting `env -u CLAUDECODE` when running pipelines.
- **`for_each` on a string**: `read_file` returns plain text — iterating it yields characters. Use `chunk` first, then `for_each` on `.chunks`.
- **Duplicate step names**: context raises immediately; validator catches this statically.
- **Template/arguments mismatch**: `{{ args.key }}` in template requires `key` in the `arguments` object literal. `StrictUndefined` makes this a hard render error.
- **Reserved step names**: `input`, `item`, `item_index` cannot be step names.
- **Loop variable binding**: inside `for_each` nested steps, the current element is `item`, not the loop step's name.

## Session Completion

Work is NOT complete until pushed (see `AGENTS.md` for full checklist).

```bash
just test           # if code changed
bd close <id> ...   # close finished issues
git pull --rebase && bd sync && git push
git status          # must show "up to date with origin"
```
