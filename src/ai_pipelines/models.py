"""Pydantic models for pipeline definitions and results.

All data structures live here. No business logic, just shapes.
Steps use a discriminated union on the ``kind`` field so invalid
step configurations fail at parse time.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


# ── Step definitions ──────────────────────────────────────────────


class FindFilesStep(BaseModel):
    kind: Literal["find_files"]
    name: str
    arguments: str
    pattern: str


class ReadFileStep(BaseModel):
    kind: Literal["read_file"]
    name: str
    arguments: str


class TransformStep(BaseModel):
    kind: Literal["transform"]
    name: str
    arguments: str


class ChunkStep(BaseModel):
    kind: Literal["chunk"]
    name: str
    arguments: str
    chunk_size: int = 4000
    overlap: int = 200


class PromptStep(BaseModel):
    kind: Literal["prompt"]
    name: str
    arguments: str | None = None
    model: Literal["haiku", "sonnet", "opus"] = "sonnet"
    system_prompt: str | None = None
    template: str
    output: dict[str, Any] | None = None


class EvaluateStep(BaseModel):
    kind: Literal["evaluate"]
    name: str
    arguments: str  # required JSONata expression producing a dict
    strategy: Literal[
        "summarization",
        "faithfulness",
        "hallucination",
        "context_relevance",
        "context_utilization",
        "factual_accuracy",
        "context_conciseness",
    ]
    model: Literal["haiku", "sonnet", "opus"] = "haiku"


class ForEachStep(BaseModel):
    kind: Literal["for_each", "pipeline"]
    name: str
    arguments: str
    steps: list[Step]


# Discriminated union: Pydantic picks the right model based on `kind`
Step = Annotated[
    FindFilesStep
    | ReadFileStep
    | TransformStep
    | ChunkStep
    | PromptStep
    | EvaluateStep
    | ForEachStep,
    Field(discriminator="kind"),
]

# Resolve forward reference for recursive ForEachStep.steps
ForEachStep.model_rebuild()


# ── Pipeline definition ──────────────────────────────────────────


class PipelineDefinition(BaseModel):
    input: dict[str, Any]
    steps: list[Step]


# ── Runtime results ──────────────────────────────────────────────


class StepResult(BaseModel):
    step_name: str
    value: Any
    duration_ms: float
    cost_usd: float | None = None


class PipelineResult(BaseModel):
    output: Any
    step_results: list[StepResult]
    total_duration_ms: float
    total_cost_usd: float
