"""Schemas for the current, file-backed paper training cohort.

The cohort is a scheduling view over the complete paper inventory.  It does
not rewrite paper readiness and it never turns a CPU fixture into production
evidence.  A paper remains in this artifact even when its execution route is
blocked or needs an evidence bootstrap step.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


PaperTrainingCohortCategory = Literal[
    "coco_supervised_ready",
    "teacher_bootstrap_ready",
    "evidence_bootstrap_ready",
    "inference_only",
    "requires_external_domain",
    "implementation_blocked",
]

COHORT_CATEGORIES: tuple[PaperTrainingCohortCategory, ...] = (
    "coco_supervised_ready",
    "teacher_bootstrap_ready",
    "evidence_bootstrap_ready",
    "inference_only",
    "requires_external_domain",
    "implementation_blocked",
)


class PaperTrainingCohortRecord(BaseModel):
    """Current scheduling classification for exactly one paper identity."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    paper_ids: list[str] = Field(default_factory=list)
    profile_id: str
    method_profile_ids: list[str] = Field(default_factory=list)
    mechanism_ids: list[str] = Field(default_factory=list)
    recipe_ids: list[str] = Field(default_factory=list)
    execution_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    category: PaperTrainingCohortCategory
    training_candidate: bool = False
    asha_eligible: bool = False
    pre_registered: bool = False
    cpu_checks_passed: bool = False
    runtime_checks_passed: bool = False
    matched_control_plan_ready: bool = False
    matched_control_result_ready: bool = False
    teacher_ready: bool = False
    evidence_ready: bool = False
    inference_only: bool = False
    asha_trial_id: str | None = None
    blocker: str | None = None
    recovery_action: str | None = None

    @model_validator(mode="after")
    def validate_record(self) -> "PaperTrainingCohortRecord":
        if not self.paper_id.strip():
            raise ValueError("paper cohort record requires paper_id")
        if not self.paper_ids:
            self.paper_ids = [self.paper_id]
        if self.paper_id not in self.paper_ids:
            raise ValueError("paper_id must be included in paper_ids provenance")
        self.paper_ids = sorted(set(self.paper_ids))
        self.method_profile_ids = sorted(set(self.method_profile_ids))
        self.mechanism_ids = sorted(set(self.mechanism_ids))
        self.recipe_ids = sorted(set(self.recipe_ids))
        if self.category == "inference_only":
            if self.training_candidate or self.asha_eligible:
                raise ValueError("inference-only paper cannot enter training cohort")
            if not self.inference_only:
                raise ValueError("inference-only category requires inference_only=true")
        if self.category in {"requires_external_domain", "implementation_blocked"}:
            if self.training_candidate or self.asha_eligible:
                raise ValueError(f"{self.category} paper cannot be a training candidate")
        if self.category == "evidence_bootstrap_ready" and self.asha_eligible:
            raise ValueError("evidence bootstrap must complete before ASHA eligibility")
        if self.asha_eligible and not self.training_candidate:
            raise ValueError("ASHA eligibility requires a training candidate")
        if self.asha_eligible:
            if not self.cpu_checks_passed:
                raise ValueError("ASHA eligibility requires CPU checks")
            if not self.runtime_checks_passed:
                raise ValueError("ASHA eligibility requires runtime checks")
            if not self.matched_control_plan_ready:
                raise ValueError("ASHA eligibility requires a matched control plan")
            if self.inference_only:
                raise ValueError("inference-only paper cannot be ASHA eligible")
            if self.blocker:
                raise ValueError("ASHA-eligible paper cannot retain a blocker")
        if not self.training_candidate and not self.blocker:
            raise ValueError("non-candidate paper requires an exact blocker")
        return self


class PaperTrainingCohort(BaseModel, YAMLModelMixin):
    """Complete paper denominator plus the executable current-data subset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_training_cohort.v1"
    inventory_path: str
    requirements_path: str
    assets_path: str
    readiness_path: str
    asha_path: str
    inventory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirements_file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_registry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    readiness_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    paper_count: int = Field(ge=0)
    inventory_count: int = Field(ge=0)
    category_counts: dict[str, int] = Field(default_factory=dict)
    executable_fingerprint_count: int = Field(ge=0)
    training_cohort_fingerprints: list[str] = Field(default_factory=list)
    training_allowed: bool = False
    records: list[PaperTrainingCohortRecord] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cohort_hash: str = ""

    @model_validator(mode="after")
    def validate_cohort(self) -> "PaperTrainingCohort":
        ids = [item.paper_id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("paper training cohort contains duplicate paper IDs")
        if ids != sorted(ids):
            raise ValueError("paper training cohort records must be sorted by paper_id")
        if self.paper_count != len(ids) or self.inventory_count != len(ids):
            raise ValueError("cohort paper counts must equal record count")
        expected_counts = {
            category: sum(item.category == category for item in self.records)
            for category in COHORT_CATEGORIES
        }
        if self.category_counts != expected_counts:
            raise ValueError("cohort category counts do not match records")
        eligible = sorted(
            {
                item.execution_fingerprint
                for item in self.records
                if item.asha_eligible
            }
        )
        if self.training_cohort_fingerprints != eligible:
            raise ValueError("training cohort fingerprints do not match eligible records")
        if self.executable_fingerprint_count != len(eligible):
            raise ValueError("executable fingerprint count does not match records")
        if self.training_allowed != bool(eligible):
            raise ValueError("training_allowed must reflect executable fingerprints")
        if self.cohort_hash and self.cohort_hash != self.calculate_hash():
            raise ValueError("paper training cohort hash mismatch")
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"cohort_hash", "generated_at"},
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def with_hash(self) -> "PaperTrainingCohort":
        return self.model_copy(update={"cohort_hash": self.calculate_hash()})

    @property
    def executable_records(self) -> list[PaperTrainingCohortRecord]:
        return [item for item in self.records if item.asha_eligible]


__all__ = [
    "COHORT_CATEGORIES",
    "PaperTrainingCohort",
    "PaperTrainingCohortCategory",
    "PaperTrainingCohortRecord",
]
