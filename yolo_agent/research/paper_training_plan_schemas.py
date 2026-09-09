"""Schemas for a prepared, non-executing paper training cohort."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


class PaperTrainingPlanRecord(BaseModel):
    """One paper execution identity and its paired queue nodes."""

    model_config = ConfigDict(extra="forbid")

    paper_ids: list[str] = Field(min_length=1)
    profile_ids: list[str] = Field(min_length=1)
    mechanism_ids: list[str] = Field(min_length=1)
    recipe_id: str
    recipe_version: str
    execution_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_node_id: str
    baseline_node_id: str
    asha_trial_id: str
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fidelity: str = "pilot_3"
    imgsz: int = Field(default=640, frozen=True)
    runtime_payload_path: str
    runtime_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_command: list[str] = Field(min_length=1)
    baseline_command: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_record(self) -> "PaperTrainingPlanRecord":
        self.paper_ids = sorted(set(self.paper_ids))
        self.profile_ids = sorted(set(self.profile_ids))
        self.mechanism_ids = sorted(set(self.mechanism_ids))
        if not all(item.strip() for item in self.paper_ids):
            raise ValueError("paper training plan requires non-empty paper IDs")
        if self.candidate_node_id == self.baseline_node_id:
            raise ValueError("candidate and matched baseline nodes must differ")
        return self


class PaperTrainingPlan(BaseModel, YAMLModelMixin):
    """A complete dry-run plan that can later be consumed by ``train``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_training_plan.v1"
    run_id: str
    run_dir: str
    inventory_path: str
    requirements_path: str
    assets_path: str
    readiness_path: str
    asha_path: str
    round_plan_path: str
    queue_path: str
    run_protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_papers: int = Field(ge=0)
    trainable_fingerprints: int = Field(ge=0)
    matched_controls_planned: int = Field(ge=0)
    asha_trials_registered: int = Field(ge=0)
    records: list[PaperTrainingPlanRecord] = Field(default_factory=list)
    dry_run_only: bool = True
    training_allowed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    plan_hash: str = ""

    @model_validator(mode="after")
    def validate_plan(self) -> "PaperTrainingPlan":
        if not self.dry_run_only:
            raise ValueError("paper training plan must be created as a dry-run artifact")
        fingerprints = [item.execution_fingerprint for item in self.records]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("paper training plan contains duplicate execution fingerprints")
        if self.trainable_fingerprints != len(fingerprints):
            raise ValueError("trainable fingerprint count does not match plan records")
        if self.matched_controls_planned != len(fingerprints):
            raise ValueError("every trainable fingerprint requires one matched control")
        if self.asha_trials_registered != len(fingerprints):
            raise ValueError("every trainable fingerprint requires one ASHA trial")
        if self.total_papers < len({paper_id for item in self.records for paper_id in item.paper_ids}):
            raise ValueError("total paper count cannot exclude plan provenance")
        if self.training_allowed != bool(fingerprints):
            raise ValueError("training_allowed must reflect prepared fingerprints")
        if self.plan_hash and self.plan_hash != self.calculate_hash():
            raise ValueError("paper training plan hash mismatch")
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"plan_hash", "created_at"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def with_hash(self) -> "PaperTrainingPlan":
        return self.model_copy(update={"plan_hash": self.calculate_hash()})


__all__ = ["PaperTrainingPlan", "PaperTrainingPlanRecord"]
