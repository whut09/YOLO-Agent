"""Persistent schemas for paper-level and execution-level proposal coverage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


ProposalDisposition = Literal[
    "queued",
    "already_tested",
    "evidence_recovery",
    "implementation_request",
    "incompatible",
    "blocked_runtime",
    "deferred_budget",
]

CoverageBoundary = Literal[
    "inventory",
    "planner",
    "critic",
    "materialization_input",
    "round_execution_plan",
    "runtime_readiness",
    "asha_registration",
    "candidate_terminal",
]


class PaperProposalStageEvent(BaseModel):
    """One immutable routing decision observed at a candidate boundary."""

    model_config = ConfigDict(extra="forbid")

    source_stage: str
    boundary: CoverageBoundary | None = None
    disposition: ProposalDisposition
    reason_codes: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    execution_fingerprint: str | None = None
    candidate_id: str | None = None
    asha_trial_id: str | None = None
    node_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaperProposalDisposition(BaseModel):
    """Current auditable disposition of one canonical execution proposal."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_proposal_disposition.v2"
    run_id: str
    round_index: int = 0
    paper_id: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    profile_id: str | None = None
    method_profile_ids: list[str] = Field(default_factory=list)
    paper_specific_mechanism_id: str | None = None
    recipe_id: str
    recipe_version: str
    canonical_component_ids: list[str] = Field(default_factory=list)
    combination_id: str | None = None
    combination_fingerprint: str | None = None
    coupling_reason: str | None = None
    coupling_source_papers: list[str] = Field(default_factory=list)
    internal_ablation_plan: list[dict[str, object]] = Field(default_factory=list)
    execution_fingerprint: str | None = None
    protocol_hash: str | None = None
    dataset_manifest_hash: str | None = None
    candidate_id: str | None = None
    asha_trial_id: str | None = None
    node_id: str | None = None
    source_stage: str
    disposition: ProposalDisposition
    reason_codes: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    required_adapters: list[str] = Field(default_factory=list)
    matched_error_fact_ids: list[str] = Field(default_factory=list)
    budget_rank: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage_history: list[PaperProposalStageEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_disposition(self) -> "PaperProposalDisposition":
        self.paper_ids = sorted(set(self.paper_ids) | ({self.paper_id} if self.paper_id else set()))
        self.method_profile_ids = sorted(
            set(self.method_profile_ids) | ({self.profile_id} if self.profile_id else set())
        )
        self.canonical_component_ids = sorted(set(self.canonical_component_ids))
        if self.disposition != "queued" and not self.reason_codes:
            raise ValueError("non-queued proposal dispositions require reason_codes")
        if self.disposition == "evidence_recovery" and not self.required_evidence:
            raise ValueError("evidence_recovery requires required_evidence")
        if self.disposition == "implementation_request" and not self.required_adapters:
            raise ValueError("implementation_request requires required_adapters")
        if self.disposition in {"queued", "deferred_budget"} and not self.execution_fingerprint:
            raise ValueError("queued/deferred proposals require execution_fingerprint")
        if self.disposition == "deferred_budget" and not self.asha_trial_id:
            raise ValueError("deferred_budget requires a recoverable ASHA trial identity")
        return self


class PaperCoverageStageEvent(BaseModel):
    """One paper's immutable state at one required routing boundary."""

    model_config = ConfigDict(extra="forbid")

    boundary: CoverageBoundary
    source_stage: str
    disposition: ProposalDisposition
    reason_codes: list[str] = Field(default_factory=list)
    recipe_id: str
    recipe_version: str
    execution_fingerprint: str
    asha_trial_id: str | None = None
    node_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaperCoverageDisposition(BaseModel):
    """Exactly one current routing disposition for one compatible paper."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    profile_id: str
    method_profile_ids: list[str] = Field(default_factory=list)
    paper_specific_mechanism_id: str | None = None
    recipe_id: str
    recipe_version: str
    canonical_component_ids: list[str] = Field(default_factory=list)
    protocol_hash: str
    dataset_manifest_hash: str
    execution_fingerprint: str
    source_stage: str
    disposition: ProposalDisposition
    reason_codes: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    required_adapters: list[str] = Field(default_factory=list)
    matched_error_fact_ids: list[str] = Field(default_factory=list)
    budget_rank: int | None = None
    asha_trial_id: str | None = None
    node_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage_history: list[PaperCoverageStageEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_current_state(self) -> "PaperCoverageDisposition":
        if not self.paper_id or not self.profile_id:
            raise ValueError("paper coverage requires paper_id and profile_id")
        if not self.recipe_id or not self.recipe_version or not self.execution_fingerprint:
            raise ValueError("paper coverage requires recoverable recipe identity")
        if self.disposition != "queued" and not self.reason_codes:
            raise ValueError("non-queued paper dispositions require reason_codes")
        if self.disposition == "evidence_recovery" and not self.required_evidence:
            raise ValueError("paper evidence recovery requires required_evidence")
        if self.disposition == "implementation_request" and not self.required_adapters:
            raise ValueError("paper implementation request requires required_adapters")
        if self.disposition == "deferred_budget" and not self.asha_trial_id:
            raise ValueError("deferred paper requires a recoverable ASHA trial identity")
        return self


class PaperCandidateCoverage(BaseModel, YAMLModelMixin):
    """Paper denominator plus de-duplicated execution proposal records."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_candidate_coverage.v2"
    run_id: str
    protocol_hash: str = "unknown"
    dataset_manifest_hash: str = "unknown"
    inventory_hash: str | None = None
    expected_paper_count: int = 0
    paper_coverage: list[PaperCoverageDisposition] = Field(default_factory=list)
    records: list[PaperProposalDisposition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_paper_denominator(self) -> "PaperCandidateCoverage":
        paper_ids = [item.paper_id for item in self.paper_coverage]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("paper candidate coverage has duplicate paper dispositions")
        if self.expected_paper_count and len(paper_ids) != self.expected_paper_count:
            raise ValueError(
                "paper candidate coverage denominator mismatch: "
                f"expected {self.expected_paper_count}, found {len(paper_ids)}"
            )
        known = set(paper_ids)
        referenced = {paper_id for record in self.records for paper_id in record.paper_ids}
        if known and not referenced.issubset(known):
            raise ValueError(
                "proposal records reference papers outside the frozen inventory: "
                + ", ".join(sorted(referenced - known))
            )
        return self

    @property
    def current_by_fingerprint(self) -> dict[str, PaperProposalDisposition]:
        return {
            record.execution_fingerprint: record
            for record in self.records
            if record.execution_fingerprint
        }

    @property
    def current_by_paper(self) -> dict[str, PaperCoverageDisposition]:
        return {record.paper_id: record for record in self.paper_coverage}

    @property
    def disposition_counts(self) -> dict[str, int]:
        source = self.paper_coverage or self.records
        counts: dict[str, int] = {}
        for record in source:
            counts[record.disposition] = counts.get(record.disposition, 0) + 1
        return dict(sorted(counts.items()))


__all__ = [
    "CoverageBoundary",
    "PaperCandidateCoverage",
    "PaperCoverageDisposition",
    "PaperCoverageStageEvent",
    "PaperProposalDisposition",
    "PaperProposalStageEvent",
    "ProposalDisposition",
]
