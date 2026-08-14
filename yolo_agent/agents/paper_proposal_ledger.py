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


class PaperProposalStageEvent(BaseModel):
    """One immutable routing decision observed at a candidate boundary."""

    model_config = ConfigDict(extra="forbid")

    source_stage: str
    disposition: ProposalDisposition
    reason_codes: list[str] = Field(default_factory=list)
    candidate_id: str | None = None
    node_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    candidate_id: str | None = None
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
        coverage = PaperCandidateCoverage.from_yaml(self.path)
        if coverage.run_id != self.run_id:
            raise RuntimeError(
                "paper proposal coverage run mismatch: "
                f"expected {self.run_id}, found {coverage.run_id}"
            )
        if (
            self.protocol_hash != "unknown"
            and coverage.protocol_hash != "unknown"
            and coverage.protocol_hash != self.protocol_hash
        ):
            raise RuntimeError(
                "paper proposal coverage protocol mismatch: "
                f"expected {self.protocol_hash}, found {coverage.protocol_hash}"
            )
        return coverage

    def upsert(self, record: PaperProposalDisposition) -> PaperProposalDisposition:
        coverage = self.read()
        record = _with_current_stage_event(record)
        key = _record_key(record)
        existing = next(
            (item for item in coverage.records if _record_key(item) == key),
            None,
        )
        merged = _merge_record(existing, record) if existing is not None else record
        retained = [item for item in coverage.records if _record_key(item) != key]
        retained.append(merged)
        coverage = PaperCandidateCoverage(
            run_id=self.run_id,
            protocol_hash=self.protocol_hash,
            records=sorted(retained, key=_record_key),
        )
        coverage.to_yaml(self.path, sort_keys=False)
        return merged

    def upsert_many(self, records: Iterable[PaperProposalDisposition]) -> PaperCandidateCoverage:
        coverage = self.read()
        by_key = {_record_key(item): item for item in coverage.records}
        for raw_record in records:
            record = _with_current_stage_event(raw_record)
            key = _record_key(record)
            existing = by_key.get(key)
            by_key[key] = _merge_record(existing, record) if existing is not None else record
        result = PaperCandidateCoverage(
            run_id=self.run_id,
            protocol_hash=self.protocol_hash,
            records=[by_key[key] for key in sorted(by_key)],
        )
        result.to_yaml(self.path, sort_keys=False)
        return result

    def update_disposition(
        self,
        *,
        execution_fingerprint: str,
        disposition: ProposalDisposition,
        reason_codes: list[str],
        source_stage: str,
        candidate_id: str | None = None,
        node_id: str | None = None,
        required_evidence: list[str] | None = None,
        required_adapters: list[str] | None = None,
    ) -> PaperProposalDisposition | None:
        """Update one downstream stage without creating an untracked proposal."""
        coverage = self.read()
        updated: PaperProposalDisposition | None = None
        records: list[PaperProposalDisposition] = []
        for record in coverage.records:
            if record.execution_fingerprint != execution_fingerprint:
                records.append(record)
                continue
            evidence = list(required_evidence or record.required_evidence)
            adapters = list(required_adapters or record.required_adapters)
            if disposition == "evidence_recovery" and not evidence:
                evidence = list(dict.fromkeys(reason_codes)) or ["recipe_bound_error_facts"]
            if disposition == "implementation_request" and not adapters:
                adapters = [
                    f"adapter_for:{component_id}"
                    for component_id in record.canonical_component_ids
                ]
            updated = PaperProposalDisposition.model_validate(
                {
                    **record.model_dump(mode="python"),
                    "disposition": disposition,
                    "reason_codes": list(dict.fromkeys(reason_codes)),
                    "source_stage": source_stage,
                    "candidate_id": candidate_id or record.candidate_id,
                    "node_id": node_id or record.node_id,
                    "required_evidence": evidence,
                    "required_adapters": adapters,
                }
            )
            updated = _merge_record(record, _with_current_stage_event(updated))
            records.append(updated)
        if updated is None:
            return None
        PaperCandidateCoverage(
            run_id=self.run_id,
            protocol_hash=self.protocol_hash,
            records=sorted(records, key=_record_key),
        ).to_yaml(self.path, sort_keys=False)
        return updated

    def update_candidate_disposition(
        self,
        *,
        candidate_id: str,
        disposition: ProposalDisposition,
        reason_codes: list[str],
        source_stage: str,
        node_id: str | None = None,
    ) -> PaperProposalDisposition | None:
        """Update a materialized candidate after planner identity is known."""
        coverage = self.read()
        matches = [item for item in coverage.records if item.candidate_id == candidate_id]
        if not matches:
            return None
        return self.update_disposition(
            execution_fingerprint=matches[0].execution_fingerprint or "",
            disposition=disposition,
            reason_codes=reason_codes,
            source_stage=source_stage,
            candidate_id=candidate_id,
            node_id=node_id,
        )

    def ensure_runtime_candidate(
        self,
        *,
        candidate_id: str,
        recipe_id: str,
        recipe_version: str,
        component_ids: list[str],
        execution_fingerprint: str,
        disposition: ProposalDisposition,
        reason_codes: list[str],
        source_stage: str,
        node_id: str | None = None,
        required_evidence: list[str] | None = None,
        required_adapters: list[str] | None = None,
    ) -> PaperProposalDisposition:
        """Create a runtime candidate record when an upstream stage omitted it."""
        existing = self.read().current_by_fingerprint.get(execution_fingerprint)
        if existing is not None:
            return self.update_disposition(
                execution_fingerprint=execution_fingerprint,
                disposition=disposition,
                reason_codes=reason_codes,
                source_stage=source_stage,
                candidate_id=candidate_id,
                node_id=node_id,
                required_evidence=required_evidence,
                required_adapters=required_adapters,
            ) or existing
        evidence = list(required_evidence or [])
        adapters = list(required_adapters or [])
        if disposition == "evidence_recovery" and not evidence:
            evidence = ["runtime_candidate_error_facts"]
        if disposition == "implementation_request" and not adapters:
            adapters = [f"adapter_for:{component_id}" for component_id in component_ids]
        return self.upsert(PaperProposalDisposition(
            run_id=self.run_id,
            paper_ids=[],
            recipe_id=recipe_id,
            recipe_version=recipe_version,
            canonical_component_ids=sorted(set(component_ids)),
            execution_fingerprint=execution_fingerprint,
            candidate_id=candidate_id,
            node_id=node_id,
            source_stage=source_stage,
            disposition=disposition,
            reason_codes=list(dict.fromkeys(reason_codes)),
            required_evidence=evidence,
            required_adapters=adapters,
        ))

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


def _merge_record(
    existing: PaperProposalDisposition,
    incoming: PaperProposalDisposition,
) -> PaperProposalDisposition:
    """Merge compatible provenance while rejecting ambiguous runtime identity."""
    identity_fields = (
        "run_id",
        "recipe_id",
        "recipe_version",
        "combination_id",
    )
    conflicts = [
        field
        for field in identity_fields
        if getattr(existing, field) != getattr(incoming, field)
    ]
    if set(existing.canonical_component_ids) != set(incoming.canonical_component_ids):
        conflicts.append("canonical_component_ids")
    if (
        existing.execution_fingerprint
        and incoming.execution_fingerprint
        and existing.execution_fingerprint != incoming.execution_fingerprint
    ):
        conflicts.append("execution_fingerprint")
    if (
        existing.candidate_id
        and incoming.candidate_id
        and existing.candidate_id != incoming.candidate_id
    ):
        conflicts.append("candidate_id")
    if conflicts:
        fingerprint = incoming.execution_fingerprint or _record_key(incoming)
        raise RuntimeError(
            "paper proposal fingerprint identity conflict "
            f"for {fingerprint}: {', '.join(sorted(set(conflicts)))}"
        )

    def merged_values(left: list[str], right: list[str]) -> list[str]:
        return sorted(set(left) | set(right))

    existing = _with_current_stage_event(existing)
    incoming = _with_current_stage_event(incoming)
    history_by_key = {
        _stage_event_key(event): event
        for event in [*existing.stage_history, *incoming.stage_history]
    }
    return incoming.model_copy(
        update={
            "paper_ids": merged_values(existing.paper_ids, incoming.paper_ids),
            "method_profile_ids": merged_values(
                existing.method_profile_ids,
                incoming.method_profile_ids,
            ),
            "candidate_id": incoming.candidate_id or existing.candidate_id,
            "node_id": incoming.node_id or existing.node_id,
            "required_evidence": merged_values(
                existing.required_evidence,
                incoming.required_evidence,
            ),
            "required_adapters": merged_values(
                existing.required_adapters,
                incoming.required_adapters,
            ),
            "matched_error_fact_ids": merged_values(
                existing.matched_error_fact_ids,
                incoming.matched_error_fact_ids,
            ),
            "budget_rank": (
                incoming.budget_rank
                if incoming.budget_rank is not None
                else existing.budget_rank
            ),
            "created_at": min(existing.created_at, incoming.created_at),
            "stage_history": sorted(
                history_by_key.values(),
                key=lambda event: (event.created_at, _stage_event_key(event)),
            ),
        }
    )


def _with_current_stage_event(
    record: PaperProposalDisposition,
) -> PaperProposalDisposition:
    event = PaperProposalStageEvent(
        source_stage=record.source_stage,
        disposition=record.disposition,
        reason_codes=list(record.reason_codes),
        candidate_id=record.candidate_id,
        node_id=record.node_id,
    )
    existing_keys = {_stage_event_key(item) for item in record.stage_history}
    if _stage_event_key(event) in existing_keys:
        return record
    return record.model_copy(update={"stage_history": [*record.stage_history, event]})


def _stage_event_key(event: PaperProposalStageEvent) -> tuple[object, ...]:
    return (
        event.source_stage,
        event.disposition,
        tuple(event.reason_codes),
        event.candidate_id,
        event.node_id,
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
    method_profile_ids: list[str] | None = None,
    required_evidence: list[str] | None = None,
    required_adapters: list[str] | None = None,
    execution_fingerprint: str | None = None,
    candidate_id: str | None = None,
    combination_id: str | None = None,
    budget_rank: int | None = None,
    source_stage: str = "paper_recipe_planner",
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
        method_profile_ids=sorted(set(method_profile_ids or [])),
        recipe_id=recipe_id,
        recipe_version=recipe_version,
        canonical_component_ids=sorted(set(component_ids)),
        combination_id=combination_id,
        execution_fingerprint=execution_fingerprint,
        candidate_id=candidate_id,
        source_stage=source_stage,
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
    "PaperProposalStageEvent",
    "ProposalDisposition",
    "planned_recipe_disposition",
]
