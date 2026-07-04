"""Component card schemas for the YOLO optimization registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.core.task_spec import TaskType


ComponentType = Literal[
    "backbone_block",
    "neck",
    "head",
    "bbox_loss",
    "cls_loss",
    "assigner",
    "augmentation",
    "optimizer",
]
FrameworkName = Literal["ultralytics", "mmyolo", "generic"]
ModelFamily = Literal["yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolo26", "generic"]


class Compatibility(BaseModel):
    """Structured compatibility notes for a component."""

    frameworks: list[FrameworkName] = Field(default_factory=lambda: ["generic"])
    tasks: list[TaskType] = Field(default_factory=lambda: ["detect"])
    model_families: list[ModelFamily] = Field(default_factory=lambda: ["generic"])
    requires: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)


class SearchSpace(BaseModel):
    """Tunable parameters exposed by a component card."""

    enabled: bool = True
    parameters: dict[str, list[str | int | float | bool]] = Field(default_factory=dict)
    default: dict[str, str | int | float | bool] = Field(default_factory=dict)


class EvidenceRequirement(BaseModel):
    """Minimum validation evidence required before recommending a component."""

    min_trials: int = Field(default=1, ge=0)
    required_metrics: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ComponentCard(BaseModel):
    """Metadata card describing a selectable YOLO component."""

    id: str
    name: str
    type: ComponentType
    description: str = ""
    target_problems: list[str] = Field(default_factory=list)
    compatible_frameworks: list[FrameworkName] = Field(default_factory=lambda: ["generic"])
    compatible_tasks: list[TaskType] = Field(default_factory=lambda: ["detect"])
    compatible_model_families: list[ModelFamily] = Field(default_factory=lambda: ["generic"])
    constraints: dict[str, Any] = Field(default_factory=dict)
    search_space: SearchSpace = Field(default_factory=SearchSpace)
    risks: list[str] = Field(default_factory=list)
    evidence_required: EvidenceRequirement = Field(default_factory=EvidenceRequirement)

    @property
    def compatibility(self) -> Compatibility:
        """Return a grouped compatibility view for consumers that prefer it."""
        return Compatibility(
            frameworks=self.compatible_frameworks,
            tasks=self.compatible_tasks,
            model_families=self.compatible_model_families,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ComponentCard":
        """Load and validate a component card from YAML."""
        yaml_path = Path(path)
        with yaml_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Component card YAML must contain a mapping: {yaml_path}")
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Serialize a component card to YAML."""
        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        with yaml_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False)
