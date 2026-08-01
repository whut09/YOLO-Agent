"""Typed contracts for the paper-driven automatic optimization acceptance suite.

The generic GPU certification report covers several historical paths.  This
module describes the stricter paper path explicitly so a report cannot claim
paper reproduction when the snapshot, adapter identity, or paired evidence is
missing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


PaperAutoOptimizationStatus = Literal["passed", "failed", "skipped", "recovery"]
PaperAutoOptimizationStageId = Literal[
    "fresh_snapshot",
    "diagnosis",
    "method_profile",
    "certified_adapter",
    "matched_pilot_3_cohort",
    "candidate_control_post_eval",
    "complete_coco_error_facts",
    "paired_bootstrap_delta",
    "asha",
    "pilot_10",
    "policy_memory",
    "pilot_reproduced",
    "evidence_recovery",
]


class PaperProtocolIdentity(BaseModel):
    """Fields that must match between a candidate and its baseline control."""

    model_config = ConfigDict(extra="forbid")

    dataset_manifest_hash: str
    subset_manifest_hash: str
    seed: int
    epochs: int = Field(ge=1)
    batch_policy_hash: str
    ultralytics_version: str
    eval_protocol_hash: str
    imgsz: int = Field(default=640, ge=640, le=640)
    objective_hash: str
    protocol_hash: str

    def comparison_payload(self) -> dict[str, Any]:
        """Return only candidate/control fields; recipe changes are excluded."""
        return self.model_dump(mode="json", exclude={"protocol_hash", "objective_hash"})


class PaperRuntimeIdentity(BaseModel):
    """Frozen paper and adapter identity attached to a candidate observation."""

    model_config = ConfigDict(extra="forbid")

    paper_ids: list[str] = Field(min_length=1)
    component_id: Literal["sampling.small_object"] = "sampling.small_object"
    adapter_hash: str = Field(min_length=1)
    maturity: Literal[
        "gpu_certified",
        "pilot_reproduced",
        "full_reproduced",
        "confirmed_multi_seed",
    ]
    snapshot_hash: str = Field(min_length=1)
    runtime_payload_hash: str = Field(min_length=1)
    runtime_protocol_hash: str


class PaperAutoOptimizationStage(BaseModel):
    """One persisted stage in the acceptance state machine."""

    model_config = ConfigDict(extra="forbid")

    stage_id: PaperAutoOptimizationStageId
    status: PaperAutoOptimizationStatus
    message: str = ""
    command: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class PaperPairedDelta(BaseModel):
    """Promotion-facing paired result for the sampling recipe."""

    model_config = ConfigDict(extra="forbid")

    stage_id: Literal["pilot_3", "pilot_10"]
    baseline_id: str = ""
    candidate_id: str = "sampling.small_object"
    verified: bool = False
    protocol_match: bool = False
    ap_small_delta: float | None = None
    target_recall_delta: float | None = None
    false_negative_delta: float | None = None
    overall_map50_95_delta: float | None = None
    latency_delta_ms: float | None = None
    model_size_delta_mb: float | None = None
    paired_bootstrap_ci: tuple[float, float] | None = None
    rejection_reasons: list[str] = Field(default_factory=list)
    result_hash: str | None = None


class PaperAutoOptimizationReport(BaseModel, YAMLModelMixin):
    """Machine-readable final result of paper-driven optimization acceptance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_auto_optimization_acceptance.v1"
    acceptance_id: str
    status: PaperAutoOptimizationStatus
    execute_real_gpu: bool
    model: str
    device: str
    fixed_imgsz: int = Field(default=640, ge=640, le=640)
    recipe_id: Literal["sampling.small_object"] = "sampling.small_object"
    research_snapshot_hash: str | None = None
    research_snapshot_path: Path | None = None
    paper_ids: list[str] = Field(default_factory=list)
    component_id: str = "sampling.small_object"
    scalar_hpo_enabled: Literal[False] = False
    adapter_hash: str | None = None
    maturity: str | None = None
    runtime_payload_hash: str | None = None
    objective_hash: str | None = None
    protocol_hash: str | None = None
    stages: list[PaperAutoOptimizationStage] = Field(default_factory=list)
    protocol_identities: dict[str, PaperProtocolIdentity] = Field(default_factory=dict)
    paired_deltas: list[PaperPairedDelta] = Field(default_factory=list)
    asha_survivor: str | None = None
    policy_memory_path: Path | None = None
    pilot_reproduced: bool = False
    evidence_recovery_actions: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "PaperAutoOptimizationReport":
        if self.status == "passed":
            required = {
                "fresh_snapshot",
                "diagnosis",
                "method_profile",
                "certified_adapter",
                "matched_pilot_3_cohort",
                "candidate_control_post_eval",
                "complete_coco_error_facts",
                "paired_bootstrap_delta",
                "asha",
                "pilot_10",
                "policy_memory",
                "pilot_reproduced",
            }
            complete = {
                item.stage_id for item in self.stages if item.status == "passed"
            }
            missing = sorted(required - complete)
            if missing:
                raise ValueError(
                    "passed paper auto-optimization report is missing stages: "
                    + ", ".join(missing)
                )
            if self.failures or not self.pilot_reproduced:
                raise ValueError(
                    "passed paper auto-optimization report requires pilot_reproduced "
                    "and no failures"
                )
            if self.maturity not in {
                "gpu_certified",
                "pilot_reproduced",
                "full_reproduced",
                "confirmed_multi_seed",
            }:
                raise ValueError("passed report requires a certified adapter maturity")
            if len(self.paired_deltas) != 2 or not all(
                item.verified and item.protocol_match and not item.rejection_reasons
                for item in self.paired_deltas
            ):
                raise ValueError(
                    "passed report requires verified pilot_3 and pilot_10 paired deltas"
                )
            if not self.asha_survivor or self.asha_survivor != self.recipe_id:
                raise ValueError("passed report requires sampling recipe ASHA survivor")
        if self.status == "recovery" and not self.evidence_recovery_actions:
            raise ValueError("recovery report requires evidence recovery actions")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("paper auto-optimization report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"report_hash", "generated_at"},
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()


__all__ = [
    "PaperAutoOptimizationReport",
    "PaperAutoOptimizationStage",
    "PaperAutoOptimizationStageId",
    "PaperAutoOptimizationStatus",
    "PaperPairedDelta",
    "PaperProtocolIdentity",
    "PaperRuntimeIdentity",
]
