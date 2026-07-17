"""Machine-readable certification contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


CertificationLevel = Literal["mini_gpu_pilot", "full_coco_multi_seed"]
CertificationStatus = Literal["passed", "failed", "skipped", "running"]


class CertificationStage(BaseModel):
    stage_id: str
    status: CertificationStatus
    message: str = ""
    command: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class CertificationCapabilityClaim(BaseModel):
    capability_id: str
    local_reproduction: Literal["locally_pilot_reproduced", "confirmed_multi_seed"]
    certification_level: CertificationLevel


class CertificationObjectiveResult(BaseModel):
    objective_hash: str | None = None
    primary_metric: str = "map50_95"
    required_delta: float | None = None
    observed_delta: float | None = None
    confidence_interval_95: tuple[float, float] | None = None
    baseline_seeds: list[int] = Field(default_factory=list)
    candidate_seeds: list[int] = Field(default_factory=list)
    latency_regression: float | None = None
    model_size_regression: float | None = None
    passed: bool = False


class CertificationReport(BaseModel, YAMLModelMixin):
    schema_version: str = "gpu_certification.v1"
    certification_id: str
    level: CertificationLevel
    status: CertificationStatus
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str
    data_yaml: str
    device: str
    fixed_imgsz: int = Field(default=640, ge=640, le=640)
    environment: dict[str, Any] = Field(default_factory=dict)
    protocol_hash: str
    stages: list[CertificationStage] = Field(default_factory=list)
    paired_result_hashes: list[str] = Field(default_factory=list)
    asha_survivor: str | None = None
    objective: CertificationObjectiveResult | None = None
    capability_claims: list[CertificationCapabilityClaim] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "CertificationReport":
        required = {
            "environment",
            "train_entrypoint",
            "debug",
            "pilot_3_control",
            "pilot_3_candidates",
            "post_eval",
            "error_facts",
            "paired_delta",
            "asha_decision",
            "pilot_10",
        }
        completed = {stage.stage_id for stage in self.stages if stage.status == "passed"}
        if self.status == "passed" and not required.issubset(completed):
            missing = sorted(required - completed)
            raise ValueError(f"passed certification is missing required stages: {missing}")
        if self.status == "passed" and self.failures:
            raise ValueError("passed certification cannot contain failures")
        if self.level == "full_coco_multi_seed" and self.status == "passed":
            if self.objective is None or not self.objective.passed:
                raise ValueError("full COCO certification requires a passed objective")
            if len(set(self.objective.baseline_seeds)) < 3 or len(set(self.objective.candidate_seeds)) < 3:
                raise ValueError("full COCO certification requires three baseline and candidate seeds")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("certification report hash does not match its payload")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"report_hash", "generated_at"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()

    @classmethod
    def load_verified(cls, path: Path | str) -> "CertificationReport":
        return cls.from_yaml(path)
