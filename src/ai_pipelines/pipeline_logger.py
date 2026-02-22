"""Structured JSON logging for pipeline execution.

Writes JSON-lines to disk so agents and humans can debug pipeline
runs after the fact. Each log entry is a single JSON object on one line.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger("ai_pipelines")


def configure_logging(
    log_dir: str | Path, level: int = logging.DEBUG
) -> None:
    """Set up pipeline logging to write JSON-lines to a file.

    Args:
        log_dir: Directory to write ``pipeline.log`` into.
        level: Logging level (default: DEBUG).
    """
    log_path = Path(log_dir) / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(str(log_path))
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))

    _logger.addHandler(handler)
    _logger.setLevel(level)


def _log(event: dict[str, Any]) -> None:
    _logger.info(json.dumps(event, default=str))


def log_step_start(step_name: str, step_kind: str) -> None:
    _log({"event": "step_start", "step_name": step_name, "step_kind": step_kind})


def log_step_complete(step_name: str, duration_ms: float) -> None:
    _log({
        "event": "step_complete",
        "step_name": step_name,
        "duration_ms": round(duration_ms, 2),
    })


def log_llm_call(
    step_name: str, model: str, prompt_preview: str, cost_usd: float | None
) -> None:
    _log({
        "event": "llm_call",
        "step_name": step_name,
        "model": model,
        "prompt_preview": prompt_preview[:200],
        "cost_usd": cost_usd,
    })


def log_error(step_name: str, error: str) -> None:
    _log({"event": "error", "step_name": step_name, "error": error})
