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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-pipelines",
        description="YAML-driven AI/LLM pipeline executor",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Execute a pipeline")
    run_p.add_argument("pipeline", type=Path, help="Path to the pipeline YAML file")
    run_p.add_argument(
        "--input", "-i",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Input key=value pair (repeatable)",
    )
    run_p.add_argument(
        "--input-json",
        type=Path,
        metavar="FILE",
        help="Read pipeline input from a JSON file",
    )
    run_p.add_argument(
        "--output", "-o",
        type=Path,
        metavar="FILE",
        help="Write JSON output to a file instead of stdout",
    )
    run_p.add_argument(
        "--log-dir",
        type=Path,
        metavar="DIR",
        help="Directory for pipeline execution logs",
    )

    # ── validate ──────────────────────────────────────────────────
    val_p = sub.add_parser("validate", help="Validate a pipeline without executing it")
    val_p.add_argument("pipeline", type=Path, help="Path to the pipeline YAML file")

    return parser


def _parse_inputs(raw: list[str], input_json: Path | None) -> dict[str, Any]:
    """Build the input dict from --input flags and/or --input-json."""
    data: dict[str, Any] = {}

    if input_json is not None:
        with open(input_json) as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            print(f"Error: --input-json must contain a JSON object, got {type(loaded).__name__}", file=sys.stderr)
            sys.exit(1)
        data.update(loaded)

    for pair in raw:
        if "=" not in pair:
            print(f"Error: --input values must be KEY=VALUE, got {pair!r}", file=sys.stderr)
            sys.exit(1)
        key, value = pair.split("=", 1)
        # Try to parse as JSON for non-string values
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


if __name__ == "__main__":
    main()
