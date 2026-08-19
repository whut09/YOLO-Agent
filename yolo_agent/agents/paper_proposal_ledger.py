"""Auditable coverage ledger for paper proposals and executable candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from yolo_agent.agents.paper_proposal_schemas import (
    CoverageBoundary,
    PaperCandidateCoverage,
    PaperCoverageDisposition,
    PaperCoverageStageEvent,
    PaperProposalDisposition,
    PaperProposalStageEvent,
    ProposalDisposition,
)
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)


class PaperCandidateCoverageLedger:
    """Upsert-only ledger with a strict no-silent-drop reconciliation check."""

    def __init__(
        self,
        path: Path | str,
        *,
        run_id: str,
        protocol_hash: str = "unknown",
        dataset_manifest_hash: str = "unknown",
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.protocol_hash = protocol_hash
        self.dataset_manifest_hash = dataset_manifest_hash

    def read(self) -> PaperCandidateCoverage:
        if not self.path.is_file():
            return PaperCandidateCoverage(
                run_id=self.run_id,
                protocol_hash=self.protocol_hash,
                dataset_manifest_hash=self.dataset_manifest_hash,
            )
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
        coverage = _coverage_with_records(coverage, retained)
        coverage = _project_records_to_papers(coverage, [merged])
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
        result = _coverage_with_records(
            coverage,
            [by_key[key] for key in sorted(by_key)],
        )
        result = _project_records_to_papers(result, by_key.values())
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
            asha_trial_id = record.asha_trial_id
            if disposition == "deferred_budget" and not asha_trial_id:
                asha_trial_id = _reserved_asha_trial_id(
                    self.run_id,
                    record.execution_fingerprint or _record_key(record),
                )
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
                    "asha_trial_id": asha_trial_id,
                }
            )
            updated = _merge_record(record, _with_current_stage_event(updated))
            records.append(updated)
        if updated is None:
            return None
        result = _project_records_to_papers(
            _coverage_with_records(coverage, records),
            [updated],
        )
        result.to_yaml(self.path, sort_keys=False)
        return updated

    def seed_inventory(self, inventory: PaperExecutionInventory) -> PaperCandidateCoverage:
        """Freeze the complete compatible-paper denominator into this run ledger."""
        coverage = self.read()
        if coverage.inventory_hash and coverage.inventory_hash != inventory.inventory_hash:
            raise RuntimeError("paper proposal inventory hash mismatch")
        if coverage.paper_coverage:
            actual = set(coverage.current_by_paper)
            expected = {item.paper_id for item in inventory.records}
            if actual != expected:
                raise RuntimeError(
                    "paper proposal inventory denominator changed: "
                    + ", ".join(sorted(actual ^ expected))
                )
            return coverage

        papers = [_paper_coverage_from_inventory(item, self) for item in inventory.records]
        result = coverage.model_copy(
            update={
                "dataset_manifest_hash": self.dataset_manifest_hash,
                "inventory_hash": inventory.inventory_hash,
                "expected_paper_count": inventory.compatible_paper_count,
                "paper_coverage": papers,
            }
        )
        result.to_yaml(self.path, sort_keys=False)
        return result

    def seal_boundary(self, boundary: CoverageBoundary) -> PaperCandidateCoverage:
        """Write one state event for every inventoried paper at a boundary."""
        coverage = self.read()
        if not coverage.paper_coverage:
            return coverage
        papers = []
        for paper in coverage.paper_coverage:
            if any(event.boundary == boundary for event in paper.stage_history):
                papers.append(paper)
                continue
            event = _paper_stage_event(
                paper,
                boundary=boundary,
                source_stage=f"{boundary}_carry_forward",
            )
            papers.append(
                paper.model_copy(
                    update={"stage_history": [*paper.stage_history, event]}
                )
            )
        result = coverage.model_copy(update={"paper_coverage": papers})
        result.to_yaml(self.path, sort_keys=False)
        self.assert_boundary_complete(boundary)
        return result

    def assert_boundary_complete(self, boundary: CoverageBoundary) -> None:
        """Fail when a required boundary silently omits an inventoried paper."""
        coverage = self.read()
        missing = sorted(
            paper.paper_id
            for paper in coverage.paper_coverage
            if not any(event.boundary == boundary for event in paper.stage_history)
        )
        if missing:
            raise RuntimeError(
                f"paper proposal {boundary} boundary has silent drops: "
                + ", ".join(missing)
            )

    def reconcile_papers(self, expected_paper_ids: Iterable[str]) -> None:
        """Fail when the persisted paper denominator differs from inventory."""
        actual = set(self.read().current_by_paper)
        expected = set(expected_paper_ids)
        if actual != expected:
            raise RuntimeError(
                "paper proposal coverage has paper-level silent drops: "
                + ", ".join(sorted(actual ^ expected))
            )

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
        combination_id: str | None = None,
        coupling_reason: str | None = None,
        coupling_source_papers: list[str] | None = None,
        internal_ablation_plan: list[dict[str, object]] | None = None,
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
            combination_id=combination_id,
            combination_fingerprint=execution_fingerprint,
            coupling_reason=coupling_reason,
            coupling_source_papers=sorted(set(coupling_source_papers or [])),
            internal_ablation_plan=list(internal_ablation_plan or []),
            execution_fingerprint=execution_fingerprint,
            asha_trial_id=(
                _reserved_asha_trial_id(self.run_id, execution_fingerprint)
                if disposition == "deferred_budget"
                else None
            ),
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


def _coverage_with_records(
    coverage: PaperCandidateCoverage,
    records: Iterable[PaperProposalDisposition],
) -> PaperCandidateCoverage:
    """Replace execution records without dropping the frozen paper denominator."""
    return coverage.model_copy(
        update={"records": sorted(records, key=_record_key)}
    )


_BOUNDARY_BY_STAGE: dict[str, CoverageBoundary] = {
    "paper_execution_inventory": "inventory",
    "paper_recipe_planner": "planner",
    "recipe_critic": "critic",
    "materialization_input": "materialization_input",
    "round_execution_plan": "round_execution_plan",
    "runtime_readiness": "runtime_readiness",
    "asha_registration": "asha_registration",
    "asha_execution": "candidate_terminal",
    "candidate_completion": "candidate_terminal",
    "candidate_failure": "candidate_terminal",
}

_DISPOSITION_PRIORITY: dict[ProposalDisposition, int] = {
    "queued": 7,
    "deferred_budget": 6,
    "already_tested": 5,
    "evidence_recovery": 4,
    "implementation_request": 3,
    "blocked_runtime": 2,
    "incompatible": 1,
}


def _project_records_to_papers(
    coverage: PaperCandidateCoverage,
    records: Iterable[PaperProposalDisposition],
) -> PaperCandidateCoverage:
    """Project merged execution decisions onto exactly one row per paper."""
    if not coverage.paper_coverage:
        return coverage
    by_paper = coverage.current_by_paper
    for record in records:
        boundary = _BOUNDARY_BY_STAGE.get(record.source_stage)
        if boundary is None:
            continue
        for paper_id in record.paper_ids:
            current = by_paper.get(paper_id)
            if current is None:
                raise RuntimeError(
                    f"paper proposal references non-inventory paper: {paper_id}"
                )
            event = PaperCoverageStageEvent(
                boundary=boundary,
                source_stage=record.source_stage,
                disposition=record.disposition,
                reason_codes=list(record.reason_codes),
                recipe_id=record.recipe_id,
                recipe_version=record.recipe_version,
                execution_fingerprint=record.execution_fingerprint or _record_key(record),
                asha_trial_id=record.asha_trial_id,
                node_id=record.node_id,
            )
            event_key = _paper_event_key(event)
            history = list(current.stage_history)
            if event_key not in {_paper_event_key(item) for item in history}:
                history.append(event)
            same_boundary = [
                item for item in history if item.boundary == boundary
            ]
            winner = max(
                same_boundary,
                key=lambda item: (
                    _DISPOSITION_PRIORITY[item.disposition],
                    item.execution_fingerprint,
                ),
            )
            update: dict[str, object] = {"stage_history": history}
            if winner == event:
                update.update(
                    {
                        "recipe_id": record.recipe_id,
                        "recipe_version": record.recipe_version,
                        "canonical_component_ids": record.canonical_component_ids,
                        "protocol_hash": record.protocol_hash or coverage.protocol_hash,
                        "dataset_manifest_hash": (
                            record.dataset_manifest_hash
                            or coverage.dataset_manifest_hash
                        ),
                        "execution_fingerprint": event.execution_fingerprint,
                        "source_stage": record.source_stage,
                        "disposition": record.disposition,
                        "reason_codes": record.reason_codes,
                        "required_evidence": record.required_evidence,
                        "required_adapters": record.required_adapters,
                        "matched_error_fact_ids": record.matched_error_fact_ids,
                        "budget_rank": record.budget_rank,
                        "asha_trial_id": record.asha_trial_id,
                        "node_id": record.node_id,
                    }
                )
            by_paper[paper_id] = current.model_copy(update=update)
    return coverage.model_copy(
        update={"paper_coverage": [by_paper[key] for key in sorted(by_paper)]}
    )


def _paper_stage_event(
    paper: PaperCoverageDisposition,
    *,
    boundary: CoverageBoundary,
    source_stage: str,
) -> PaperCoverageStageEvent:
    return PaperCoverageStageEvent(
        boundary=boundary,
        source_stage=source_stage,
        disposition=paper.disposition,
        reason_codes=list(paper.reason_codes),
        recipe_id=paper.recipe_id,
        recipe_version=paper.recipe_version,
        execution_fingerprint=paper.execution_fingerprint,
        asha_trial_id=paper.asha_trial_id,
        node_id=paper.node_id,
    )


def _paper_event_key(event: PaperCoverageStageEvent) -> tuple[object, ...]:
    return (
        event.boundary,
        event.source_stage,
        event.disposition,
        tuple(event.reason_codes),
        event.recipe_id,
        event.recipe_version,
        event.execution_fingerprint,
        event.asha_trial_id,
        event.node_id,
    )


def _paper_coverage_from_inventory(
    item: PaperExecutionSpec,
    ledger: PaperCandidateCoverageLedger,
) -> PaperCoverageDisposition:
    disposition: ProposalDisposition = (
        "queued" if item.current_disposition == "runtime_ready" else item.current_disposition
    )  # type: ignore[assignment]
    mechanism = next(iter(item.paper_specific_mechanism_ids), None)
    recipe_id = next(
        iter(item.recipe_ids),
        "implementation:" + (mechanism or next(iter(item.canonical_component_ids), item.paper_id)),
    )
    reason_code = f"inventory_{disposition}"
    required_adapters = sorted(
        {
            resolution.required_adapter
            for resolution in item.paper_mechanism_resolutions
            if resolution.required_adapter
        }
    )
    if disposition == "implementation_request" and not required_adapters:
        required_adapters = [
            f"adapter_for:{component_id}"
            for component_id in item.canonical_component_ids
        ] or [f"paper_specific_adapter:{item.paper_id}"]
    required_evidence = list(item.required_evidence)
    if disposition == "evidence_recovery" and not required_evidence:
        required_evidence = ["paper_specific_mechanism_evidence"]
    trial_id = (
        _reserved_asha_trial_id(ledger.run_id, item.execution_fingerprint)
        if disposition == "deferred_budget"
        else None
    )
    event = PaperCoverageStageEvent(
        boundary="inventory",
        source_stage="paper_execution_inventory",
        disposition=disposition,
        reason_codes=[reason_code],
        recipe_id=recipe_id,
        recipe_version="inventory.v1",
        execution_fingerprint=item.execution_fingerprint,
        asha_trial_id=trial_id,
    )
    return PaperCoverageDisposition(
        paper_id=item.paper_id,
        profile_id=item.profile_id,
        method_profile_ids=[item.profile_id],
        paper_specific_mechanism_id=mechanism,
        recipe_id=recipe_id,
        recipe_version="inventory.v1",
        canonical_component_ids=item.canonical_component_ids,
        protocol_hash=ledger.protocol_hash,
        dataset_manifest_hash=ledger.dataset_manifest_hash,
        execution_fingerprint=item.execution_fingerprint,
        source_stage=event.source_stage,
        disposition=disposition,
        reason_codes=event.reason_codes,
        required_evidence=required_evidence,
        required_adapters=required_adapters,
        matched_error_fact_ids=item.matched_error_fact_ids,
        asha_trial_id=trial_id,
        stage_history=[event],
    )


def _reserved_asha_trial_id(run_id: str, execution_fingerprint: str) -> str:
    """Return the stable ASHA identity reserved before budget allocation."""
    return f"{run_id}:paper:{execution_fingerprint}"


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
    for field in (
        "paper_id",
        "profile_id",
        "paper_specific_mechanism_id",
        "protocol_hash",
        "dataset_manifest_hash",
        "asha_trial_id",
    ):
        left = getattr(existing, field)
        right = getattr(incoming, field)
        if left is not None and right is not None and left != right:
            conflicts.append(field)
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
            "asha_trial_id": incoming.asha_trial_id or existing.asha_trial_id,
            "protocol_hash": incoming.protocol_hash or existing.protocol_hash,
            "dataset_manifest_hash": (
                incoming.dataset_manifest_hash
                or existing.dataset_manifest_hash
            ),
            "paper_id": incoming.paper_id or existing.paper_id,
            "profile_id": incoming.profile_id or existing.profile_id,
            "paper_specific_mechanism_id": (
                incoming.paper_specific_mechanism_id
                or existing.paper_specific_mechanism_id
            ),
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
            "combination_fingerprint": (
                incoming.combination_fingerprint
                or existing.combination_fingerprint
            ),
            "coupling_reason": incoming.coupling_reason or existing.coupling_reason,
            "coupling_source_papers": merged_values(
                existing.coupling_source_papers,
                incoming.coupling_source_papers,
            ),
            "internal_ablation_plan": (
                incoming.internal_ablation_plan
                or existing.internal_ablation_plan
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
        boundary=_BOUNDARY_BY_STAGE.get(record.source_stage),
        disposition=record.disposition,
        reason_codes=list(record.reason_codes),
        paper_ids=list(record.paper_ids),
        execution_fingerprint=record.execution_fingerprint,
        candidate_id=record.candidate_id,
        asha_trial_id=record.asha_trial_id,
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
        tuple(event.paper_ids),
        event.execution_fingerprint,
        event.candidate_id,
        event.asha_trial_id,
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
    matched_error_fact_ids: list[str] | None = None,
    execution_fingerprint: str | None = None,
    candidate_id: str | None = None,
    combination_id: str | None = None,
    combination_fingerprint: str | None = None,
    coupling_reason: str | None = None,
    coupling_source_papers: list[str] | None = None,
    internal_ablation_plan: list[dict[str, object]] | None = None,
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
        "already_tested": "already_tested",
        "blocked_runtime": "blocked_runtime",
        "incompatible": "incompatible",
        "evidence_recovery": "evidence_recovery",
        "deferred_budget": "deferred_budget",
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
    trial_id = (
        _reserved_asha_trial_id(run_id, execution_fingerprint)
        if disposition == "deferred_budget" and execution_fingerprint
        else None
    )
    return PaperProposalDisposition(
        run_id=run_id,
        round_index=round_index,
        paper_ids=sorted(set(related_papers or [])),
        method_profile_ids=sorted(set(method_profile_ids or [])),
        recipe_id=recipe_id,
        recipe_version=recipe_version,
        canonical_component_ids=sorted(set(component_ids)),
        combination_id=combination_id,
        combination_fingerprint=combination_fingerprint or execution_fingerprint,
        coupling_reason=coupling_reason,
        coupling_source_papers=sorted(set(coupling_source_papers or [])),
        internal_ablation_plan=list(internal_ablation_plan or []),
        execution_fingerprint=execution_fingerprint,
        asha_trial_id=trial_id,
        candidate_id=candidate_id,
        source_stage=source_stage,
        disposition=disposition,
        reason_codes=normalized_reasons or (["eligible_for_pilot"] if disposition == "queued" else ["unspecified"]),
        required_evidence=evidence,
        required_adapters=adapters,
        matched_error_fact_ids=sorted(set(matched_error_fact_ids or [])),
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
