"""Artifact contracts for model-graph and assignment runtime certification."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


GraphComponentId = Literal[
    "head.p2_small_object",
    "neck.multi_scale_fusion",
    "neck.gold_gather_distribute",
    "neck.rtmdet_large_kernel",
    "neck.weighted_feature_pyramid",
    "neck.bidirectional_feature_fusion",
    "neck.lightweight",
    "block.reparameterized_convolution",
    "attention.channel",
    "attention.spatial",
    "neck.deformable_feature_aggregation",
]

AssignmentComponentId = Literal[
    "assigner.task_aligned",
    "assigner.optimal_transport",
    "assigner.dynamic_smooth_label",
    "assigner.task_aligned_weighting",
    "assigner.dynamic_topk",
    "assigner.quality_aware",
    "assigner.soft_label",
    "assigner.dual_path",
    "assigner.conflict_aware",
]


class GraphCpuReport(BaseModel, YAMLModelMixin):
    """Independent CPU golden-path result for one graph component."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "graph_cpu_golden_path.v1"
    component_id: GraphComponentId
    recipe_id: str
    status: Literal["passed", "failed"]
    protocol_hash: str
    runtime_payload_hash: str
    runtime_payload_path: Path
    manifest_path: Path | None = None
    checkpoint_audit_path: Path | None = None
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "GraphCpuReport":
        if self.status == "passed":
            required = {
                "atomic_recipe_verified",
                "real_forward",
                "native_loss_preserved",
                "backward",
                "amp",
                "partial_checkpoint_audit",
                "export",
                "resource_guard",
                "matched_control_required",
            }
            missing = sorted(key for key in required if self.checks.get(key) is not True)
            if missing:
                raise ValueError("passed graph report is missing checks: " + ", ".join(missing))
            if self.errors:
                raise ValueError("passed graph report cannot contain errors")
            if self.manifest_path is None:
                raise ValueError("passed graph report requires a graph manifest")
            if self.checkpoint_audit_path is None:
                raise ValueError("passed graph report requires checkpoint audit")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("graph report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"generated_at", "report_hash"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class AssignmentShadowCpuReport(BaseModel, YAMLModelMixin):
    """Shadow-only evidence before an assignment method can become active."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "assignment_shadow_cpu_golden_path.v1"
    component_id: AssignmentComponentId
    method: Literal[
        "tood_tal",
        "ota",
        "dsla",
        "task_aligned_weighting",
        "dynamic_topk",
        "quality_aware",
        "soft_label",
        "dual_path",
        "conflict_aware",
    ]
    recipe_id: str
    status: Literal["passed", "failed"]
    protocol_hash: str
    runtime_payload_hash: str
    runtime_payload_path: Path
    shadow_evidence_path: Path | None = None
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "AssignmentShadowCpuReport":
        if self.status == "passed":
            required = {
                "atomic_recipe_verified",
                "shadow_mode_only",
                "native_audit_verified",
                "positive_ratio_recorded",
                "conflict_rate_recorded",
                "matching_stability_recorded",
                "per_path_metrics_recorded",
                "native_loss_equivalent",
                "native_one_to_one_preserved",
                "matched_control_required",
                "active_pilot_blocked_until_explicit_gate",
            }
            missing = sorted(key for key in required if self.checks.get(key) is not True)
            if missing:
                raise ValueError(
                    "passed assignment report is missing checks: " + ", ".join(missing)
                )
            if self.errors:
                raise ValueError("passed assignment report cannot contain errors")
            if self.shadow_evidence_path is None:
                raise ValueError("passed assignment report requires shadow evidence")
            for key in (
                "baseline_positive_ratio",
                "candidate_positive_ratio",
                "conflict_rate",
                "matching_stability",
            ):
                if key not in self.metrics:
                    raise ValueError(f"passed assignment report requires metric: {key}")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("assignment report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"generated_at", "report_hash"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


__all__ = [
    "AssignmentComponentId",
    "AssignmentShadowCpuReport",
    "GraphComponentId",
    "GraphCpuReport",
]
