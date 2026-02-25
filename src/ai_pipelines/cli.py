"""Command-line interface for ai-pipelines.

Enables execution via ``uvx ai-pipelines``, ``python -m ai_pipelines``,
or a plain ``ai-pipelines`` command after install.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# ── Human-readable help strings ──────────────────────────────────────────────
# Written to teach both humans and LLM agents how and when to use this tool.

_TOP_DESCRIPTION = """\
YAML-driven AI/LLM pipeline executor.

Loads a pipeline defined in YAML, resolves inputs, and executes each step in
order. Steps can call LLMs (prompt), read files (find_files, read_file),
transform data (transform, chunk), and loop over collections (for_each).

Use this tool when you need to RUN or VALIDATE a .yaml pipeline file.
Do NOT use it to: write pipeline YAML, query an LLM directly, list files,
or manage the filesystem outside of a pipeline run.
"""

_TOP_EPILOG = """\
For a machine-readable JSON description of this CLI (commands, argument
schemas, output shapes, examples):

  ai-pipelines schema

Quick examples:
  ai-pipelines run pipeline.yaml --input dir=/data
  ai-pipelines run pipeline.yaml --input-json inputs.json --output result.json
  ai-pipelines validate pipeline.yaml
"""

_RUN_DESCRIPTION = """\
Execute a pipeline YAML file end-to-end and emit results as JSON.

Loads the pipeline from YAML, merges inputs (--input flags take precedence
over --input-json), then runs every step in order. The final pipeline value
is returned in the "output" field.
"""

_RUN_EPILOG = """\
Output schema (JSON written to stdout, or to the --output file):

  {
    "output": <any>          -- final pipeline value; type depends on last step
    "step_results": [
      {
        "step_name": <str>,  -- name declared in the YAML
        "value":     <any>,  -- value stored in pipeline context after this step
        "duration_ms": <num> -- wall-clock milliseconds for this step
      }
    ],
    "total_duration_ms": <num>,  -- total wall-clock milliseconds
    "total_cost_usd":    <num>   -- total LLM cost (0.0 when no prompt steps ran)
  }

Input value parsing:
  --input KEY=VALUE tries JSON parsing first (so integers, booleans, arrays,
  and objects work without extra quoting), then falls back to a plain string.
  --input-json FILE must contain a JSON object (not an array or scalar).
  --input flags override keys from --input-json when both are supplied.

Examples:
  # Single string input
  ai-pipelines run summarize.yaml --input text="Hello world"

  # Integer input (auto-parsed from JSON)
  ai-pipelines run process.yaml --input dir=/tmp --input max_items=10

  # JSON array input
  ai-pipelines run pipeline.yaml --input 'ids=[1,2,3]'

  # Complex inputs from a JSON file, output written to a file
  ai-pipelines run pipeline.yaml --input-json inputs.json --output result.json

  # Enable step-level logging for debugging
  ai-pipelines run pipeline.yaml --input dir=/data --log-dir ./logs

Common errors:
  PipelineLoadError  -- YAML file not found or the schema is invalid
  StepExecutionError -- a step failed at runtime (use --log-dir for details)

This command does NOT:
  - Write or modify pipeline YAML files
  - List available pipelines or discover .yaml files
  - Interact with files outside of what the pipeline itself requests
"""

_VALIDATE_DESCRIPTION = """\
Statically validate a pipeline YAML file without executing it.

No LLM calls are made and no side effects occur. Checks for: duplicate step
names, unresolvable step references, malformed JSONata expressions, Jinja2
template/arguments mismatches, and JSON Schema violations.
"""

_VALIDATE_EPILOG = """\
Diagnostic output format (written to stderr on failure):
  [error]   step_name.field: message  -- blocks execution; must be fixed
  [warning] step_name.field: message  -- may cause issues at runtime

Exit codes:
  0 -- pipeline is valid; safe to pass to "ai-pipelines run"
  1 -- one or more errors found; do not attempt to run

Examples:
  ai-pipelines validate pipeline.yaml
  ai-pipelines validate pipeline.yaml && echo "OK to run"
  ai-pipelines validate pipeline.yaml && ai-pipelines run pipeline.yaml --input dir=/data
"""

_SCHEMA_DESCRIPTION = """\
Print a machine-readable JSON description of this CLI to stdout.

Designed for LLM agents and tooling that need to understand what commands
are available, what arguments they accept, and what output they produce.
Analogous to an OpenAPI spec or MCP inputSchema, but for this CLI.
Exits 0 and writes JSON to stdout; nothing is written to stderr.
"""


# ── Structured JSON schema (for `ai-pipelines schema`) ───────────────────────

def _cli_schema() -> dict[str, Any]:
    """Return a structured JSON description of the entire CLI."""
    return {
        "tool": "ai-pipelines",
        "description": (
            "YAML-driven AI/LLM pipeline executor. "
            "Runs or validates .yaml pipeline files that chain LLM calls, "
            "file I/O, and data-transformation steps."
        ),
        "when_to_use": (
            "Use this CLI when you have a .yaml pipeline file and want to "
            "execute it or check it for errors. "
            "Not a general-purpose LLM gateway or file manager."
        ),
        "not_for": [
            "Querying an LLM directly (no pipeline YAML needed for that)",
            "Writing or generating pipeline YAML files",
            "Listing available pipelines or discovering .yaml files",
            "Managing the filesystem outside of what a pipeline run requires",
        ],
        "commands": [
            {
                "name": "run",
                "description": (
                    "Execute a pipeline YAML file end-to-end and emit results as JSON. "
                    "Loads the pipeline, merges inputs, runs every step in order."
                ),
                "arguments": {
                    "pipeline": {
                        "type": "string",
                        "format": "file path",
                        "required": True,
                        "description": "Path to the pipeline YAML file to execute.",
                    },
                    "--input": {
                        "short": "-i",
                        "type": "string",
                        "format": "KEY=VALUE",
                        "required": False,
                        "repeatable": True,
                        "description": (
                            "Input key-value pair injected into the pipeline context. "
                            "VALUE is JSON-parsed first (so integers, booleans, arrays, "
                            "and objects are handled correctly); falls back to a plain "
                            "string if JSON parsing fails. "
                            "Repeatable for multiple inputs. "
                            "Overrides matching keys from --input-json."
                        ),
                        "examples": ["text=hello", "max_items=10", "ids=[1,2,3]"],
                    },
                    "--input-json": {
                        "type": "string",
                        "format": "file path",
                        "required": False,
                        "description": (
                            "Path to a JSON file whose top-level keys become pipeline "
                            "inputs. Use for complex or deeply-nested input data. "
                            "Must contain a JSON object, not an array or scalar."
                        ),
                    },
                    "--output": {
                        "short": "-o",
                        "type": "string",
                        "format": "file path",
                        "required": False,
                        "description": (
                            "Write the JSON result to this file instead of stdout. "
                            "A confirmation message is printed to stderr."
                        ),
                    },
                    "--log-dir": {
                        "type": "string",
                        "format": "directory path",
                        "required": False,
                        "description": (
                            "Directory for JSONL execution logs (pipeline.log). "
                            "Each line is a JSON event: step_start, step_complete, "
                            "or pipeline_complete. Use to debug failed runs."
                        ),
                    },
                },
                "output": {
                    "channel": "stdout (or the file given by --output)",
                    "format": "JSON object",
                    "schema": {
                        "output": "<any> — final pipeline value; type depends on the last step",
                        "step_results": [
                            {
                                "step_name": "<string> — name declared in YAML",
                                "value": "<any> — value stored in pipeline context",
                                "duration_ms": "<number> — wall-clock milliseconds",
                            }
                        ],
                        "total_duration_ms": "<number> — total wall-clock milliseconds",
                        "total_cost_usd": "<number> — total LLM spend (0.0 if no prompt steps ran)",
                    },
                },
                "exit_codes": {
                    "0": "success — JSON output has been written",
                    "1": "pipeline load error or step execution error",
                },
                "examples": [
                    {
                        "description": "Run with a single string input",
                        "command": "ai-pipelines run summarize.yaml --input text='Hello world'",
                    },
                    {
                        "description": "Run with integer and string inputs (integer auto-parsed from JSON)",
                        "command": "ai-pipelines run process.yaml --input dir=/tmp --input max_items=10",
                    },
                    {
                        "description": "Load complex inputs from a JSON file",
                        "command": "ai-pipelines run pipeline.yaml --input-json inputs.json",
                    },
                    {
                        "description": "Write output to a file instead of stdout",
                        "command": "ai-pipelines run pipeline.yaml --input dir=/data --output result.json",
                    },
                    {
                        "description": "Enable step-level logging for debugging a failing run",
                        "command": "ai-pipelines run pipeline.yaml --input dir=/data --log-dir ./logs",
                    },
                ],
            },
            {
                "name": "validate",
                "description": (
                    "Statically validate a pipeline YAML without executing it. "
                    "No LLM calls are made and no side effects occur. "
                    "Checks: duplicate step names, unresolvable references, "
                    "bad JSONata expressions, template/arguments mismatches."
                ),
                "arguments": {
                    "pipeline": {
                        "type": "string",
                        "format": "file path",
                        "required": True,
                        "description": "Path to the pipeline YAML file to validate.",
                    },
                },
                "output": {
                    "stdout_on_success": "Pipeline is valid (<N> steps)",
                    "stderr_on_failure": "[error|warning] step_name.field: message",
                    "format": "human-readable text",
                },
                "exit_codes": {
                    "0": "pipeline is valid — safe to pass to 'run'",
                    "1": "one or more errors found — do not attempt to run",
                },
                "examples": [
                    {
                        "description": "Validate a pipeline file",
                        "command": "ai-pipelines validate pipeline.yaml",
                    },
                    {
                        "description": "Validate then run as a single command",
                        "command": "ai-pipelines validate pipeline.yaml && ai-pipelines run pipeline.yaml --input dir=/data",
                    },
                ],
            },
            {
                "name": "schema",
                "description": (
                    "Print this machine-readable JSON schema to stdout. "
                    "Use to discover available commands, argument types, "
                    "output shapes, and usage examples."
                ),
                "arguments": {},
                "output": {
                    "channel": "stdout",
                    "format": "JSON object — this document",
                },
                "exit_codes": {"0": "always succeeds"},
                "examples": [
                    {
                        "description": "Print the CLI schema for LLM consumption",
                        "command": "ai-pipelines schema",
                    },
                ],
            },
        ],
        "pipeline_yaml_format": {
            "description": (
                "A pipeline YAML file has two top-level keys: "
                "'input' (JSON Schema describing expected inputs) and "
                "'steps' (ordered list of step objects)."
            ),
            "step_kinds": [
                {
                    "kind": "find_files",
                    "description": "Glob a directory for files matching a pattern. Returns a list of {path, name} objects.",
                    "required_fields": ["name", "arguments (JSONata → directory path)", "pattern (glob string)"],
                },
                {
                    "kind": "read_file",
                    "description": "Read a file's text content. Returns a plain string. NOT a list — iterate chunks, not raw text.",
                    "required_fields": ["name", "arguments (JSONata → file path)"],
                },
                {
                    "kind": "transform",
                    "description": "Evaluate a JSONata expression to reshape or compute over pipeline data.",
                    "required_fields": ["name", "arguments (JSONata expression)"],
                },
                {
                    "kind": "chunk",
                    "description": "Split a string into overlapping text chunks. Returns a list of {text, index} objects.",
                    "required_fields": ["name", "arguments (JSONata → string)"],
                    "optional_fields": ["chunk_size (default 4000)", "overlap (default 200)"],
                },
                {
                    "kind": "prompt",
                    "description": "Call an LLM with a Jinja2 template. Returns the model's text response (or structured output if 'output' is set).",
                    "required_fields": ["name", "template (Jinja2 string)"],
                    "optional_fields": ["model (haiku|sonnet|opus, default sonnet)", "arguments", "system_prompt", "output (JSON schema for structured output)"],
                },
                {
                    "kind": "evaluate",
                    "description": "LLM-as-judge quality assessment. Returns a dict with score and reasoning.",
                    "required_fields": ["name", "arguments (JSONata → dict)", "strategy"],
                    "strategies": ["summarization", "faithfulness", "hallucination", "context_relevance", "context_utilization", "factual_accuracy", "context_conciseness"],
                    "optional_fields": ["model (haiku|sonnet|opus, default haiku)"],
                },
                {
                    "kind": "for_each",
                    "description": "Iterate over an array, running nested steps per element. The current element is 'item'; its index is 'item_index'.",
                    "required_fields": ["name", "arguments (JSONata → array)", "steps (list of nested steps)"],
                    "note": "Inside nested steps, reference the current element as 'item', NOT by the for_each step's name.",
                },
            ],
            "reserved_step_names": ["input", "item", "item_index"],
            "common_mistakes": [
                "Applying for_each directly to a read_file result — read_file returns a plain string, not an array. Use 'chunk' first, then for_each on the chunks.",
                "Referencing a step by name before it appears in the steps list.",
                "Omitting input keys that the pipeline's 'input' schema marks as required.",
                "Using the for_each step's name inside its nested steps — use 'item' instead.",
            ],
        },
    }


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-pipelines",
        description=_TOP_DESCRIPTION,
        epilog=_TOP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ──────────────────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help="Execute a pipeline and emit results as JSON",
        description=_RUN_DESCRIPTION,
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_p.add_argument(
        "pipeline",
        type=Path,
        help="Path to the pipeline YAML file",
    )
    run_p.add_argument(
        "--input", "-i",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Input key=value pair injected into the pipeline context. "
            "VALUE is JSON-parsed first (so integers, booleans, arrays, and "
            "objects work without extra quoting), then falls back to a plain "
            "string. Repeatable; overrides keys from --input-json."
        ),
    )
    run_p.add_argument(
        "--input-json",
        type=Path,
        metavar="FILE",
        help=(
            "JSON file containing a top-level object whose keys become pipeline "
            "inputs. Use for complex or deeply-nested input data. "
            "Must be a JSON object, not an array."
        ),
    )
    run_p.add_argument(
        "--output", "-o",
        type=Path,
        metavar="FILE",
        help="Write JSON output to FILE instead of stdout.",
    )
    run_p.add_argument(
        "--log-dir",
        type=Path,
        metavar="DIR",
        help=(
            "Write JSONL execution logs to DIR/pipeline.log. Each line is a "
            "JSON event: step_start, step_complete, or pipeline_complete. "
            "Useful for debugging failed runs."
        ),
    )

    # ── validate ─────────────────────────────────────────────────────────────
    val_p = sub.add_parser(
        "validate",
        help="Statically validate a pipeline without executing it",
        description=_VALIDATE_DESCRIPTION,
        epilog=_VALIDATE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    val_p.add_argument(
        "pipeline",
        type=Path,
        help="Path to the pipeline YAML file to validate",
    )

    # ── schema ───────────────────────────────────────────────────────────────
    sub.add_parser(
        "schema",
        help="Print a machine-readable JSON schema of this CLI to stdout",
        description=_SCHEMA_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    return parser


# ── Command handlers ──────────────────────────────────────────────────────────

def _parse_inputs(raw: list[str], input_json: Path | None) -> dict[str, Any]:
    """Build the input dict from --input flags and/or --input-json."""
    data: dict[str, Any] = {}

    if input_json is not None:
        with open(input_json) as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            print(
                f"Error: --input-json must contain a JSON object, got {type(loaded).__name__}",
                file=sys.stderr,
            )
            sys.exit(1)
        data.update(loaded)

    for pair in raw:
        if "=" not in pair:
            print(f"Error: --input values must be KEY=VALUE, got {pair!r}", file=sys.stderr)
            sys.exit(1)
        key, value = pair.split("=", 1)
        # Try to parse as JSON for non-string values (numbers, booleans, arrays, objects)
        try:
            data[key] = json.loads(value)
        except json.JSONDecodeError:
            data[key] = value

    return data


async def _cmd_run(args: argparse.Namespace) -> int:
    from ai_pipelines import configure_logging, load_pipeline, run_pipeline

    if args.log_dir:
        configure_logging(args.log_dir)

    definition = load_pipeline(args.pipeline)
    input_data = _parse_inputs(args.input, args.input_json)
    result = await run_pipeline(definition, input_data)

    output_obj = {
        "output": result.output,
        "step_results": [
            {"step_name": sr.step_name, "value": sr.value, "duration_ms": sr.duration_ms}
            for sr in result.step_results
        ],
        "total_duration_ms": result.total_duration_ms,
        "total_cost_usd": result.total_cost_usd,
    }

    text = json.dumps(output_obj, indent=2)

    if args.output:
        args.output.write_text(text)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(text)

    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from ai_pipelines import load_and_validate_pipeline

    definition, result = load_and_validate_pipeline(args.pipeline)

    if result.ok:
        print(f"Pipeline is valid ({len(definition.steps)} steps)")
        return 0

    for d in result.diagnostics:
        print(f"[{d.severity.value}] {d.step_name}.{d.field}: {d.message}", file=sys.stderr)

    error_count = len(result.errors)
    warning_count = len(result.warnings)
    print(f"\n{error_count} error(s), {warning_count} warning(s)", file=sys.stderr)
    return 1


def _cmd_schema() -> int:
    print(json.dumps(_cli_schema(), indent=2))
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        sys.exit(asyncio.run(_cmd_run(args)))
    elif args.command == "validate":
        sys.exit(_cmd_validate(args))
    elif args.command == "schema":
        sys.exit(_cmd_schema())


if __name__ == "__main__":
    main()
