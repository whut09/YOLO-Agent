"""Schemas for the frozen 83-paper engineering campaign.

The campaign plan is intentionally paper-shaped.  Shared runtime primitives
are recorded as reusable dependencies, but they never replace a paper entry
or turn a generic mapping into a paper implementation claim.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


PlanStatus = Literal["planned", "blocked_missing_evidence", "incompatible"]
EvidenceStatus = Literal["sufficient_for_planning", "blocked_missing_evidence"]
Difficulty = Literal["low", "medium", "high"]
ImplementationDomain = Literal[
    "data_quality",
    "annotation",
    "sampling",
    "augmentation",
    "preprocessing",
    "backbone",
    "neck",
    "feature_fusion",
    "head",
    "assignment",
    "bbox_loss",
    "classification_loss",
    "auxiliary_loss",
    "distillation",
    "domain_adaptation",
    "semi_supervised",
    "training_strategy",
    "optimizer",
    "regularization",
    "postprocess",
    "inference",
    "calibration",
    "active_learning",
    "evaluation",
    "other",
]


class Paper83EngineeringPlanEntry(BaseModel):
    """One complete engineering plan for one frozen paper identity."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_engineering_plan_entry.v1"
    paper_id: str
    title: str
    method_profile_id: str | None = None
    execution_fingerprint: str | None = None
    current_disposition: str | None = None
    implementation_domain: ImplementationDomain
    secondary_domains: list[str] = Field(default_factory=list)
    paper_mechanism_summary: str
    paper_specific_mechanism_ids: list[str] = Field(default_factory=list)
    shared_primitives_available: list[str] = Field(default_factory=list)
    existing_component_ids: list[str] = Field(default_factory=list)
    existing_adapter_ids: list[str] = Field(default_factory=list)
    paper_specific_missing_parts: list[str] = Field(default_factory=list)
    runtime_insertion_points: list[str] = Field(default_factory=list)
    implementation_strategy: str
    unit_test_plan: list[str] = Field(default_factory=list)
    smoke_test_plan: list[str] = Field(default_factory=list)
    compatibility_test_plan: list[str] = Field(default_factory=list)
    required_assets: list[str] = Field(default_factory=list)
    evidence_status: EvidenceStatus
    status: PlanStatus = "planned"
    known_facts: list[str] = Field(default_factory=list)
    unknown_facts: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    estimated_files: int = Field(ge=1)
    blockers: list[str] = Field(default_factory=list)
    dependency_papers_or_primitives: list[str] = Field(default_factory=list)
    source_locations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_plan_boundary(self) -> "Paper83EngineeringPlanEntry":
        if not self.paper_id.strip():
            raise ValueError("engineering plan entry requires paper_id")
        if not self.title.strip():
            raise ValueError("engineering plan entry requires title")
        if not self.paper_mechanism_summary.strip():
            raise ValueError("engineering plan entry requires mechanism summary")
        if not self.implementation_strategy.strip():
            raise ValueError("engineering plan entry requires implementation strategy")
        if self.status == "blocked_missing_evidence":
            if self.evidence_status != "blocked_missing_evidence":
                raise ValueError(
                    "blocked_missing_evidence plan must have matching evidence_status"
                )
            if not self.unknown_facts:
                raise ValueError(
                    "blocked_missing_evidence plan must list unknown_facts"
                )
            if not self.required_evidence:
                raise ValueError(
                    "blocked_missing_evidence plan must list required_evidence"
                )
            if not self.blockers:
                raise ValueError("blocked_missing_evidence plan must list blockers")
        if self.status == "incompatible" and not self.blockers:
            raise ValueError("incompatible plan must list a blocker")
        if self.evidence_status == "sufficient_for_planning" and self.status == "blocked_missing_evidence":
            raise ValueError("sufficient evidence cannot use blocked status")
        return self


class Paper83EngineeringBatch(BaseModel):
    """A deterministic 4-8 paper implementation batch."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_engineering_batch.v1"
    batch_id: str
    focus_domain: str
    shared_primitives: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(min_length=4, max_length=8)
    paper_count: int = Field(ge=4, le=8)
    rationale: str
    plan_identity_hash: str = ""

    @model_validator(mode="after")
    def validate_batch(self) -> "Paper83EngineeringBatch":
        if not self.batch_id.strip() or not self.focus_domain.strip():
            raise ValueError("engineering batch requires identity and focus domain")
        if self.paper_ids != sorted(set(self.paper_ids)):
            raise ValueError("engineering batch paper IDs must be sorted and unique")
        if self.paper_count != len(self.paper_ids):
            raise ValueError("engineering batch paper_count does not match paper_ids")
        if not self.rationale.strip():
            raise ValueError("engineering batch requires rationale")
        return self


class Paper83EngineeringPlan(BaseModel, YAMLModelMixin):
    """Complete plan and batch membership for the frozen campaign."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_engineering_plan.v1"
    manifest_path: str
    manifest_membership_hash: str
    manifest_paper_ids: list[str] = Field(default_factory=list)
    paper_count: int = Field(ge=0)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    papers: list[Paper83EngineeringPlanEntry] = Field(default_factory=list)
    batches: list[Paper83EngineeringBatch] = Field(default_factory=list)
    batch_directory: str
    summary: dict[str, int] = Field(default_factory=dict)
    plan_identity_hash: str = ""

    @model_validator(mode="after")
    def validate_plan(self) -> "Paper83EngineeringPlan":
        ids = [item.paper_id for item in self.papers]
        if ids != sorted(set(ids)):
            raise ValueError("engineering plan papers must be sorted and unique")
        if self.manifest_paper_ids and ids != self.manifest_paper_ids:
            raise ValueError("engineering plan membership differs from manifest")
        if self.paper_count != len(ids):
            raise ValueError("paper_count must equal plan entry count")
        batch_ids = [paper_id for batch in self.batches for paper_id in batch.paper_ids]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("engineering batch membership contains duplicates")
        if self.batches and sorted(batch_ids) != sorted(ids):
            raise ValueError("engineering batches must cover every paper exactly once")
        if self.plan_identity_hash and self.plan_identity_hash != self.calculate_hash():
            raise ValueError("engineering plan identity hash mismatch")
        return self

    @property
    def blocked_count(self) -> int:
        return sum(item.status == "blocked_missing_evidence" for item in self.papers)

    @property
    def actionable_count(self) -> int:
        return sum(item.status == "planned" for item in self.papers)

    @property
    def incompatible_count(self) -> int:
        return sum(item.status == "incompatible" for item in self.papers)

    def calculate_hash(self) -> str:
        """Hash all planning content while excluding the self-referential field."""

        payload = self.model_dump(
            mode="json",
            exclude={"plan_identity_hash"},
        )
        for batch in payload["batches"]:
            batch.pop("plan_identity_hash", None)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def with_hash(self) -> "Paper83EngineeringPlan":
        digest = self.calculate_hash()
        return self.model_copy(
            update={
                "plan_identity_hash": digest,
                "batches": [
                    batch.model_copy(update={"plan_identity_hash": digest})
                    for batch in self.batches
                ],
            }
        )


__all__ = [
    "Difficulty",
    "EvidenceStatus",
    "ImplementationDomain",
    "Paper83EngineeringBatch",
    "Paper83EngineeringPlan",
    "Paper83EngineeringPlanEntry",
    "PlanStatus",
]
