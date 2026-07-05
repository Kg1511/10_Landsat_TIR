"""Response models for the Landsat-9 dashboard API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: bool
    default_color_model: Literal["cnn", "gan", "transformer"]
    demo_mode: bool
    model_ready: bool
    missing: list[str] = Field(default_factory=list)


class InspectResponse(BaseModel):
    valid: bool
    filename: str
    shape: list[int]
    dtype: str
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    finite_count: int = 0
    message: str


class OutputFile(BaseModel):
    filename: str
    url: str | None = None
    exists: bool = False
    reason: str | None = None


class InferResponse(BaseModel):
    success: bool
    run_id: str
    demo_mode: bool
    input_summary: dict[str, Any]
    outputs: dict[str, OutputFile]
    metrics_or_stats: dict[str, Any]
    logs: list[str]
