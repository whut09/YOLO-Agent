"""Task specification schemas for YOLO optimization scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


TaskType = Literal["detect", "segment", "obb"]
SceneType = Literal[
    "infrared_small_target",
    "industrial_defect",
    "traffic_edge",
    "drone_small_object",
    "generic",
]
DeviceType = Literal["cpu", "cuda", "edge_gpu", "npu", "tensorrt", "openvino", "unknown"]
MetricName = Literal[
    "map50",
    "map50_95",
    "precision",
    "recall",
    "f1",
    "fps",
    "latency_ms",
    "model_size_mb",
    "miss_cost",
    "false_alarm_cost",
]


class MetricPriority(BaseModel):
    """A metric and its relative importance for optimization."""

    name: MetricName
    weight: float = Field(default=1.0, gt=0.0)
    goal: Literal["maximize", "minimize"] = "maximize"


class ScenarioHint(BaseModel):
    """Human-authored hints that guide future planning without running training."""

    name: str
    description: str = ""
    suggested_model_size: Literal["n", "s", "m", "l", "x", "auto"] = "auto"
    notes: list[str] = Field(default_factory=list)


class DatasetSpec(BaseModel):
    """Dataset characteristics that influence model and training choices."""

    class_names: list[str] = Field(min_length=1)
    image_size: int | None = Field(default=None, gt=0)
    object_scale: Literal["tiny", "small", "mixed", "large", "unknown"] = "unknown"
    expected_image_count: int | None = Field(default=None, ge=0)
    imbalance: Literal["low", "medium", "high", "unknown"] = "unknown"


class DeploymentSpec(BaseModel):
    """Runtime constraints for the target deployment environment."""

    device_type: DeviceType = "unknown"
    target_fps: float | None = Field(default=None, gt=0.0)
    max_latency_ms: float | None = Field(default=None, gt=0.0)
    max_model_size_mb: float | None = Field(default=None, gt=0.0)


class TaskSpec(BaseModel):
    """Complete task profile consumed by YOLO Agent planning workflows."""

    task_type: TaskType = "detect"
    scene: SceneType = "generic"
    class_names: list[str] = Field(min_length=1)
    primary_metric: MetricPriority
    secondary_metrics: list[MetricPriority] = Field(default_factory=list)
    device_type: DeviceType = "unknown"
    target_fps: float | None = Field(default=None, gt=0.0)
    max_latency_ms: float | None = Field(default=None, gt=0.0)
    max_model_size_mb: float | None = Field(default=None, gt=0.0)
    miss_cost: float = Field(default=1.0, ge=0.0)
    false_alarm_cost: float = Field(default=1.0, ge=0.0)
    dataset: DatasetSpec | None = None
    deployment: DeploymentSpec | None = None
    scenario_hint: ScenarioHint | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TaskSpec":
        """Load and validate a task specification from YAML."""
        yaml_path = Path(path)
        with yaml_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Task spec YAML must contain a mapping: {yaml_path}")
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Write the task specification to YAML."""
        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = self.model_dump(mode="json", exclude_none=True)
        with yaml_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False)
