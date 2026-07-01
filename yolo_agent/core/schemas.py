"""Pydantic schemas used by the optimization harness."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    """High-level metadata about an object-detection dataset."""

    name: str = "unnamed"
    path: Path | None = None
    image_count: int | None = Field(default=None, ge=0)
    class_count: int | None = Field(default=None, ge=0)
    average_objects_per_image: float | None = Field(default=None, ge=0.0)


class DeploymentConstraints(BaseModel):
    """Target runtime constraints used to choose a YOLO configuration."""

    target: Literal["edge", "server", "cloud", "unknown"] = "unknown"
    max_latency_ms: float | None = Field(default=None, gt=0.0)
    max_model_size_mb: float | None = Field(default=None, gt=0.0)
    preferred_export: Literal["onnx", "tensorrt", "openvino", "torchscript", "none"] = "none"


class AgentConfig(BaseModel):
    """Top-level configuration for a reproducible yolo-agent run."""

    project_name: str = "yolo-agent"
    experiment_root: Path = Path("experiments")
    random_seed: int = 42
    dataset: DatasetProfile = Field(default_factory=DatasetProfile)
    deployment: DeploymentConstraints = Field(default_factory=DeploymentConstraints)

