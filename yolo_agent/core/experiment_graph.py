"""Experiment graph schemas for reproducible candidate evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer, model_validator

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.artifact_manifest import ArtifactManifestEntry
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.yaml_io import YAMLModelMixin


ExperimentStatus = Literal["planned", "running", "completed", "failed", "skipped"]


class Evidence(BaseModel):
    """Local evidence captured for a run."""

    run_id: str
    config_path: Path | None = None
    metrics_path: Path | None = None
    metric_records_path: Path | None = None
    artifact_manifest_path: Path | None = None
    artifacts_dir: Path | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    metric_records: list["MetricEvidence"] = Field(default_factory=list)
    artifact_manifest: list[ArtifactManifestEntry] = Field(default_factory=list)
    artifacts: dict[str, Path] = Field(default_factory=dict)


MetricValue = float | int | str | bool | None
METRIC_SCHEMA_VERSION = "1.0"

LOWER_IS_BETTER_METRICS = {
    "latency",
    "latency_ms",
    "model_size",
    "model_size_mb",
    "runtime_epoch_time_seconds",
    "batch_tuning_oom_trials",
    "false_negative_count",
    "false_positive_count",
    "localization_error_rate",
}


class MetricEvidence(BaseModel):
    """One metric observation tied to a candidate and experiment node."""

    candidate_id: str
    node_id: str
    dataset_version: str = "unversioned"
    split: str = "val"
    metric_name: str
    value: MetricValue
    source: str = "manual"
    verified: bool = True
    validator: str = "manual"
    source_artifact: Path | None = None
    metric_schema_version: str = METRIC_SCHEMA_VERSION
    higher_is_better: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_serializer("source_artifact")
    def serialize_source_artifact(self, value: Path | None) -> str | None:
        """Serialize source artifact paths portably."""
        return value.as_posix() if value is not None else None

    @model_validator(mode="after")
    def fill_metric_direction(self) -> "MetricEvidence":
        """Infer metric direction when it is not explicitly supplied."""
        if self.higher_is_better is None:
            self.higher_is_better = (
                self.metric_name not in LOWER_IS_BETTER_METRICS
                and not (
                    self.metric_name.startswith("batch_tuning_")
                    and self.metric_name.endswith("_oom")
                )
            )
        return self


class ExperimentNode(BaseModel):
    """A reproducible experiment node for one candidate."""

    node_id: str
    candidate_config: CandidateConfig
    data_version: str
    seed: int = 42
    command: str = ""
    command_spec: CommandSpec | None = None
    status: ExperimentStatus = "planned"
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    artifacts: dict[str, Path] = Field(default_factory=dict)
    parent_id: str | None = None
    changed_variables: dict[str, Any] = Field(default_factory=dict)


class ExperimentPlan(BaseModel, YAMLModelMixin):
    """A collection of reproducible experiment nodes."""

    plan_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    nodes: list[ExperimentNode] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def plan_hash(self) -> str:
        """Return a stable semantic hash for queue invalidation.

        ``created_at`` is intentionally excluded so rewriting the same plan
        does not invalidate a queue. Command specs, node definitions, and plan
        metadata remain included, so changes to profile, model, data,
        training config, or dry-run versus execute mode create a new hash.
        """
        payload = self.model_dump(mode="json", exclude={"created_at"})
        payload["metadata"] = {
            key: value
            for key, value in payload.get("metadata", {}).items()
            if key not in {"plan_hash", "created_at"}
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
