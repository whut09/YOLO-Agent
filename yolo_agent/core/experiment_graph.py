"""Experiment graph schemas for reproducible candidate evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.candidate_generator import CandidateConfig


ExperimentStatus = Literal["planned", "running", "completed", "failed", "skipped"]


class Evidence(BaseModel):
    """Local evidence captured for a run."""

    run_id: str
    config_path: Path | None = None
    metrics_path: Path | None = None
    artifacts_dir: Path | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    artifacts: dict[str, Path] = Field(default_factory=dict)


class ExperimentNode(BaseModel):
    """A reproducible experiment node for one candidate."""

    node_id: str
    candidate_config: CandidateConfig
    data_version: str
    seed: int = 42
    command: str
    status: ExperimentStatus = "planned"
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    artifacts: dict[str, Path] = Field(default_factory=dict)
    parent_id: str | None = None
    changed_variables: dict[str, Any] = Field(default_factory=dict)


class ExperimentPlan(BaseModel):
    """A collection of reproducible experiment nodes."""

    plan_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    nodes: list[ExperimentNode] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_yaml(self, path: Path | str) -> None:
        """Serialize the experiment plan to YAML."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.model_dump(mode="json"), file, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentPlan":
        """Load an experiment plan from YAML."""
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Experiment plan YAML must contain a mapping: {input_path}")
        return cls.model_validate(data)

