"""Artifact contracts for YOLO26 loss and distillation CPU certification."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


QualityLossComponentId = Literal[
    "loss.quality.iou_aware_classification",
    "loss.quality.correlation",
    "loss.calibration.bpc",
    "loss.quality.pseudo_iou",
    "loss.quality.localization_aware",
    "loss.boundary_aware",
    "loss.localization.uncertainty_weighted",
    "loss.hard_negative_classification",
    "loss.class_balanced_focal",
]


class QualityLossCpuReport(BaseModel, YAMLModelMixin):
    """Independent golden-path evidence for one additive quality loss."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "quality_loss_cpu_golden_path.v1"
    component_id: QualityLossComponentId
    recipe_id: str
    status: Literal["passed", "failed"]
    protocol_hash: str
    runtime_payload_hash: str
    zero_control_payload_hash: str
    runtime_payload_path: Path
    zero_control_payload_path: Path | None = None
    runtime_evidence_path: Path | None = None
    zero_control_evidence_path: Path | None = None
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "QualityLossCpuReport":
        if self.status == "passed":
            required = {
                "atomic_recipe_verified",
                "trainer_bridge_called",
                "total_loss_changed",
                "student_backward",
                "zero_weight_native_equivalent",
                "paper_prior_not_local_evidence",
                "exact_reproduction_false",
            }
            missing = sorted(key for key in required if self.checks.get(key) is not True)
            if missing:
                raise ValueError(
                    "passed quality loss CPU report is missing checks: "
                    + ", ".join(missing)
                )
            if self.errors:
                raise ValueError("passed quality loss CPU report cannot contain errors")
            if self.zero_control_payload_path is None:
                raise ValueError("passed quality loss report requires zero-weight payload")
            if self.runtime_evidence_path is None or self.zero_control_evidence_path is None:
                raise ValueError("passed quality loss report requires runtime evidence")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("quality loss CPU report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload: dict[str, Any] = self.model_dump(
            mode="json",
            exclude={"generated_at", "report_hash"},
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


class DistillationCpuReport(BaseModel, YAMLModelMixin):
    """Golden-path evidence for YOLO26 teacher-student distillation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "yolo26_distillation_cpu_golden_path.v1"
    component_id: str = "distillation.yolo26_teacher_student"
    recipe_id: str = "yolo26n_distillation"
    status: Literal["passed", "failed"]
    protocol_hash: str
    runtime_payload_hash: str
    zero_control_payload_hash: str
    runtime_payload_path: Path
    zero_control_payload_path: Path | None = None
    runtime_evidence_path: Path | None = None
    zero_control_evidence_path: Path | None = None
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "DistillationCpuReport":
        if self.status == "passed":
            required = {
                "atomic_recipe_verified",
                "trainer_bridge_called",
                "total_loss_changed",
                "student_backward",
                "teacher_no_grad",
                "teacher_frozen_eval",
                "zero_weight_native_equivalent",
                "student_inference_graph_unchanged",
                "method_profiles_only",
                "exact_reproduction_false",
            }
            missing = sorted(key for key in required if self.checks.get(key) is not True)
            if missing:
                raise ValueError(
                    "passed distillation CPU report is missing checks: "
                    + ", ".join(missing)
                )
            if self.errors:
                raise ValueError("passed distillation CPU report cannot contain errors")
            if self.zero_control_payload_path is None:
                raise ValueError("passed distillation report requires zero-weight payload")
            if self.runtime_evidence_path is None or self.zero_control_evidence_path is None:
                raise ValueError("passed distillation report requires runtime evidence")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("distillation CPU report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload: dict[str, Any] = self.model_dump(
            mode="json",
            exclude={"generated_at", "report_hash"},
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


__all__ = [
    "DistillationCpuReport",
    "QualityLossComponentId",
    "QualityLossCpuReport",
]
