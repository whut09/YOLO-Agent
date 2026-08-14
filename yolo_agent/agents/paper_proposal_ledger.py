"""Auditable coverage ledger for paper proposals and executable candidates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

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


class PaperProposalDisposition(BaseModel):
    """Current auditable disposition of one canonical execution proposal."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_proposal_disposition.v1"
    run_id: str
    round_index: int = 0
    paper_ids: list[str] = Field(default_factory=list)
    method_profile_ids: list[str] = Field(default_factory=list)
    recipe_id: str
    recipe_version: str
    canonical_component_ids: list[str] = Field(min_length=1)
    combination_id: str | None = None
    execution_fingerprint: str | None = None
    source_stage: str
    disposition: ProposalDisposition
    reason_codes: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    required_adapters: list[str] = Field(default_factory=list)
    matched_error_fact_ids: list[str] = Field(default_factory=list)
    budget_rank: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_disposition(self) -> "PaperProposalDisposition":
        if self.disposition != "queued" and not self.reason_codes:
            raise ValueError("non-queued proposal dispositions require reason_codes")
        if self.disposition == "evidence_recovery" and not self.required_evidence:
            raise ValueError("evidence_recovery requires required_evidence")
        if self.disposition == "implementation_request" and not self.required_adapters:
            raise ValueError("implementation_request requires required_adapters")
        if self.disposition == "queued" and not self.execution_fingerprint:
            raise ValueError("queued proposals require execution_fingerprint")
        return self


class PaperCandidateCoverage(BaseModel, YAMLModelMixin):
    """Reconciled proposal inventory written beside every optimization run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_candidate_coverage.v1"
    run_id: str
    protocol_hash: str = "unknown"
    records: list[PaperProposalDisposition] = Field(default_factory=list)

    @property
    def current_by_fingerprint(self) -> dict[str, PaperProposalDisposition]:
        result: dict[str, PaperProposalDisposition] = {}
        for record in self.records:
            if record.execution_fingerprint:
                result[record.execution_fingerprint] = record
        return result

    @property
    def disposition_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.disposition] = counts.get(record.disposition, 0) + 1
        return dict(sorted(counts.items()))


class PaperCandidateCoverageLedger:
    """Upsert-only ledger with a strict no-silent-drop reconciliation check."""

    def __init__(self, path: Path | str, *, run_id: str, protocol_hash: str = "unknown") -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.protocol_hash = protocol_hash

    def read(self) -> PaperCandidateCoverage:
        if not self.path.is_file():
            return PaperCandidateCoverage(run_id=self.run_id, protocol_hash=self.protocol_hash)
        return PaperCandidateCoverage.from_yaml(self.path)

    def upsert(self, record: PaperProposalDisposition) -> PaperProposalDisposition:
        coverage = self.read()
        key = _record_key(record)
        retained = [item for item in coverage.records if _record_key(item) != key]
        retained.append(record)
        coverage = PaperCandidateCoverage(
            run_id=self.run_id,
            protocol_hash=self.protocol_hash,
            records=sorted(retained, key=_record_key),
        )
        coverage.to_yaml(self.path, sort_keys=False)
        return record

    def upsert_many(self, records: Iterable[PaperProposalDisposition]) -> PaperCandidateCoverage:
        coverage = self.read()
        by_key = {_record_key(item): item for item in coverage.records}
        for record in records:
            by_key[_record_key(record)] = record
        result = PaperCandidateCoverage(
            run_id=self.run_id,
            protocol_hash=self.protocol_hash,
            records=[by_key[key] for key in sorted(by_key)],
        )
        result.to_yaml(self.path, sort_keys=False)
        return result

    def reconcile(self, expected_keys: Iterable[str]) -> None:
        """Raise when any proposal key was omitted by a downstream stage."""
        actual = {_record_key(item) for item in self.read().records}
        missing = sorted(set(expected_keys) - actual)
        if missing:
            raise RuntimeError(
                "paper proposal coverage has silent drops: " + ", ".join(missing)
            )


def _record_key(record: PaperProposalDisposition) -> str:
    return record.execution_fingerprint or ":".join(
        [
            record.recipe_id,
            record.recipe_version,
            "+".join(sorted(record.canonical_component_ids)),
            record.combination_id or "atomic",
        ]
    )


def planned_recipe_disposition(
    *,
    run_id: str,
    round_index: int,
    recipe_id: str,
    recipe_version: str,
    component_ids: list[str],
    decision: str,
    reasons: list[str],
    related_papers: list[str] | None = None,
    required_evidence: list[str] | None = None,
    required_adapters: list[str] | None = None,
    execution_fingerprint: str | None = None,
    budget_rank: int | None = None,
) -> PaperProposalDisposition:
    """Translate planner decisions into the stable user-facing disposition set."""
    mapping: dict[str, ProposalDisposition] = {
        "selected": "queued",
        "deferred": "deferred_budget",
        "needs_evidence": "evidence_recovery",
        "implementation_proposal": "implementation_request",
        "rejected": "incompatible",
    }
    disposition = mapping.get(decision, "blocked_runtime")
    evidence = list(required_evidence or [])
    adapters = list(required_adapters or [])
    normalized_reasons = list(dict.fromkeys(reasons))
    if disposition == "evidence_recovery" and not evidence:
        evidence = ["recipe_bound_error_facts"]
    if disposition == "implementation_request" and not adapters:
        adapters = [f"adapter_for:{component}" for component in component_ids]
    if disposition == "queued":
        normalized_reasons = []
    return PaperProposalDisposition(
        run_id=run_id,
        round_index=round_index,
        paper_ids=sorted(set(related_papers or [])),
        recipe_id=recipe_id,
        recipe_version=recipe_version,
        canonical_component_ids=sorted(set(component_ids)),
        execution_fingerprint=execution_fingerprint,
        source_stage="paper_recipe_planner",
        disposition=disposition,
        reason_codes=normalized_reasons or (["eligible_for_pilot"] if disposition == "queued" else ["unspecified"]),
        required_evidence=evidence,
        required_adapters=adapters,
        budget_rank=budget_rank,
    )


__all__ = [
    "PaperCandidateCoverage",
    "PaperCandidateCoverageLedger",
    "PaperProposalDisposition",
    "ProposalDisposition",
    "planned_recipe_disposition",
]
