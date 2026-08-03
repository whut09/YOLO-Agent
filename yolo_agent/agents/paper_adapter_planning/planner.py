"""Diagnosis-driven implementation queue for Awesome catalog components."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable

from yolo_agent.agents.paper_adapter_planning.diagnosis import (
    build_adapter_diagnosis_context,
    match_component_to_diagnosis,
)
from yolo_agent.agents.paper_adapter_planning.diversity import (
    assess_implementation_diversity,
)
from yolo_agent.agents.paper_adapter_planning.fingerprints import (
    component_family,
    implementation_fingerprint,
)
from yolo_agent.agents.paper_adapter_planning.local_evidence import (
    assess_local_component_evidence,
)
from yolo_agent.agents.paper_adapter_planning.policy import PaperAdapterPlanningPolicy
from yolo_agent.agents.paper_adapter_planning.requests import build_implementation_request
from yolo_agent.agents.paper_adapter_planning.runtime_hooks import assess_runtime_hook
from yolo_agent.agents.paper_adapter_planning.schemas import (
    AdapterImplementationEstimate,
    ImplementationHistoryRecord,
    PaperAdapterImplementationPlan,
    PaperAdapterQueueItem,
    RuntimeHookAvailability,
)
from yolo_agent.agents.paper_adapter_planning.scoring import score_adapter_implementation
from yolo_agent.agents.paper_adapter_planning.tracks import classify_implementation_track
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.policy_memory import PolicyMemoryRecord
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.mechanism_clusters import (
    AdapterCoverageOpportunity,
    PaperMechanismClusterReport,
)
from yolo_agent.research.schemas import PaperRecord


_ACTIONABLE_TRACKS = {
    "ready_to_materialize",
    "implementation_queue",
    "shadow_evaluation_queue",
}


class PaperAdapterImplementationPlanner:
    """Rank implementation work; never create source code or training nodes."""

    def __init__(
        self,
        resolver: ComponentAliasResolver,
        policy: PaperAdapterPlanningPolicy | None = None,
    ) -> None:
        self.resolver = resolver
        self.policy = policy or PaperAdapterPlanningPolicy()

    def plan(
        self,
        *,
        papers: Iterable[PaperRecord],
        error_facts: Iterable[ErrorFact],
        implementation_estimates: Iterable[AdapterImplementationEstimate] = (),
        runtime_hooks: Iterable[RuntimeHookAvailability] = (),
        local_evidence: Iterable[PolicyMemoryRecord] = (),
        history: Iterable[ImplementationHistoryRecord] = (),
        current_round: int = 0,
        dataset_signature: str | None = None,
        mechanism_report: PaperMechanismClusterReport | None = None,
    ) -> PaperAdapterImplementationPlan:
        paper_list = list(papers)
        diagnosis = build_adapter_diagnosis_context(error_facts)
        estimates = {item.component_id: item for item in implementation_estimates}
        hooks = list(runtime_hooks)
        evidence = list(local_evidence)
        history_records = list(history)
        mechanism_context = _mechanism_context_by_paper(mechanism_report)
        candidates: list[PaperAdapterQueueItem] = []
        for paper in paper_list:
            for paper_component_id in paper.component_ids:
                resolution = self.resolver.resolve(
                    paper_component_id,
                    source_paper_ids=[paper.paper_id],
                )
                if not resolution.resolved:
                    candidates.append(_unresolved_item(paper, paper_component_id))
                    continue
                for mapping in resolution.mappings:
                    estimate = estimates.get(mapping.canonical_component_id) or AdapterImplementationEstimate(
                        component_id=mapping.canonical_component_id,
                        required_runtime_hook=(
                            mapping.insertion_point if mapping.insertion_point != "unknown" else None
                        ),
                    )
                    match = match_component_to_diagnosis(mapping, diagnosis)
                    hook = assess_runtime_hook(
                        estimate,
                        hooks,
                        verified_weight=self.policy.runtime_hook_weight,
                    )
                    local = assess_local_component_evidence(
                        mapping.canonical_component_id,
                        evidence,
                        dataset_signature=dataset_signature,
                        positive_max_weight=self.policy.local_positive_max_weight,
                        negative_max_penalty=self.policy.local_negative_max_penalty,
                    )
                    classification = classify_implementation_track(paper, mapping, estimate)
                    if diagnosis.source_fact_count == 0 and classification.track in _ACTIONABLE_TRACKS:
                        classification = classification.model_copy(
                            update={
                                "track": "insufficient_information",
                                "reasons": [*classification.reasons, "current_error_facts_required"],
                            }
                        )
                    elif classification.track == "ready_to_materialize" and not hook.verified:
                        classification = classification.model_copy(
                            update={
                                "track": "implementation_queue",
                                "reasons": [*classification.reasons, "verified_runtime_hook_required"],
                            }
                        )
                    score = score_adapter_implementation(
                        paper=paper,
                        mapping=mapping,
                        estimate=estimate,
                        diagnosis_context=diagnosis,
                        diagnosis_match=match,
                        runtime_hook=hook,
                        local_evidence=local,
                        policy=self.policy,
                        mechanism_confidence=(
                            mechanism_context[paper.paper_id][2]
                            if paper.paper_id in mechanism_context
                            else mapping.alias_confidence
                        ),
                    )
                    track_identity = (
                        "yolo26" if classification.track in _ACTIONABLE_TRACKS else classification.track
                    )
                    fingerprint = implementation_fingerprint(
                        component_id=mapping.canonical_component_id,
                        insertion_point=mapping.insertion_point,
                        required_runtime_hook=estimate.required_runtime_hook,
                        detector_family=track_identity,
                    )
                    request = (
                        build_implementation_request(
                            fingerprint=fingerprint,
                            mapping=mapping,
                            paper_ids=[paper.paper_id],
                            estimate=estimate,
                        )
                        if classification.track == "implementation_queue"
                        else None
                    )
                    candidates.append(PaperAdapterQueueItem(
                        component_id=mapping.canonical_component_id,
                        component_family=component_family(
                            mapping.canonical_component_id,
                            mapping.category,
                        ),
                        mechanism_cluster_id=(
                            mechanism_context[paper.paper_id][0]
                            if paper.paper_id in mechanism_context
                            else None
                        ),
                        adapter_family=(
                            mechanism_context[paper.paper_id][1]
                            if paper.paper_id in mechanism_context
                            else None
                        ),
                        canonical_component_ids=[mapping.canonical_component_id],
                        covered_paper_count=1,
                        paper_ids=[paper.paper_id],
                        paper_year=paper.year,
                        official_code_available=bool(paper.official_code_url),
                        source_license=paper.code_license or "unknown",
                        yolo26_compatibility=mapping.yolo26_compatibility,
                        implementation_status=mapping.implementation_status,
                        insertion_point=mapping.insertion_point,
                        diagnosis_targets=match.targets,
                        score=score.total,
                        score_breakdown=score.breakdown,
                        reasons=list(dict.fromkeys([
                            *classification.reasons,
                            *score.reasons,
                        ])),
                        fingerprint=fingerprint,
                        track=classification.track,
                        implementation_request=request,
                        metadata={
                            "paper_title": paper.title,
                            "paper_applicability": paper.applicability,
                            "runtime_hook_available": hook.available,
                            "runtime_hook_verified": hook.verified,
                            "local_evidence_records": local.record_count,
                            "canonical_mechanism_confidence": (
                                mechanism_context[paper.paper_id][2]
                                if paper.paper_id in mechanism_context
                                else mapping.alias_confidence
                            ),
                        },
                    ))
        consolidated = _consolidate_candidates(candidates, policy=self.policy)
        guarded = [
            _apply_diversity(
                item,
                history=history_records,
                current_round=current_round,
                cooldown_rounds=self.policy.family_cooldown_rounds,
            )
            for item in consolidated
        ]
        return _build_plan(
            guarded,
            current_round=current_round,
            mechanism_opportunities=(
                _implementation_opportunities(mechanism_report)
                if mechanism_report is not None
                else []
            ),
        )


def _unresolved_item(paper: PaperRecord, component_id: str) -> PaperAdapterQueueItem:
    fingerprint = implementation_fingerprint(
        component_id=component_id,
        insertion_point="unknown",
        required_runtime_hook=None,
        detector_family="insufficient_information",
    )
    return PaperAdapterQueueItem(
        component_id=component_id,
        component_family="unknown",
        paper_ids=[paper.paper_id],
        paper_year=paper.year,
        official_code_available=bool(paper.official_code_url),
        source_license=paper.code_license or "unknown",
        yolo26_compatibility="unknown",
        implementation_status="metadata_only",
        insertion_point="unknown",
        reasons=["component_alias_unresolved", "canonical_contract_required"],
        fingerprint=fingerprint,
        track="insufficient_information",
        metadata={"paper_title": paper.title, "paper_applicability": paper.applicability},
    )


def _consolidate_candidates(
    candidates: list[PaperAdapterQueueItem],
    *,
    policy: PaperAdapterPlanningPolicy,
) -> list[PaperAdapterQueueItem]:
    grouped: dict[str, list[PaperAdapterQueueItem]] = defaultdict(list)
    for item in candidates:
        grouped[item.fingerprint].append(item)
    output: list[PaperAdapterQueueItem] = []
    for fingerprint, items in sorted(grouped.items()):
        selected = max(items, key=lambda item: (item.score, item.paper_year, item.component_id))
        paper_ids = sorted({paper_id for item in items for paper_id in item.paper_ids})
        reasons = list(dict.fromkeys(reason for item in items for reason in item.reasons))
        request = selected.implementation_request
        if request is not None:
            request = request.model_copy(update={"paper_ids": paper_ids})
        coverage_score = min(
            math.log2(len(paper_ids) + 1) * policy.paper_coverage_log_weight,
            policy.paper_coverage_max_weight,
        )
        score_breakdown = dict(selected.score_breakdown)
        score_breakdown["paper_coverage"] = coverage_score
        canonical_component_ids = sorted({
            component_id
            for item in items
            for component_id in item.canonical_component_ids
        })
        output.append(selected.model_copy(update={
            "paper_ids": paper_ids,
            "covered_paper_count": len(paper_ids),
            "canonical_component_ids": canonical_component_ids,
            "official_code_available": any(item.official_code_available for item in items),
            "score": selected.score + coverage_score,
            "score_breakdown": score_breakdown,
            "reasons": reasons,
            "implementation_request": request,
        }))
    return output


def _apply_diversity(
    item: PaperAdapterQueueItem,
    *,
    history: list[ImplementationHistoryRecord],
    current_round: int,
    cooldown_rounds: int,
) -> PaperAdapterQueueItem:
    if item.track not in _ACTIONABLE_TRACKS:
        return item
    assessment = assess_implementation_diversity(
        fingerprint=item.fingerprint,
        component_family=item.component_family,
        current_round=current_round,
        history=history,
        cooldown_rounds=cooldown_rounds,
    )
    if not assessment.deferred:
        return item
    return item.model_copy(update={
        "track": "deferred",
        "reasons": [*item.reasons, assessment.reason],
        "implementation_request": None,
    })


def _build_plan(
    items: list[PaperAdapterQueueItem],
    *,
    current_round: int,
    mechanism_opportunities: list[AdapterCoverageOpportunity] | None = None,
) -> PaperAdapterImplementationPlan:
    mechanism_opportunities = mechanism_opportunities or []
    queues: dict[str, list[PaperAdapterQueueItem]] = {
        name: []
        for name in (
            "ready_to_materialize",
            "implementation_queue",
            "shadow_evaluation_queue",
            "incompatible",
            "separate_detector_family",
            "insufficient_information",
            "deferred",
        )
    }
    for item in items:
        queues[item.track].append(item)
    for values in queues.values():
        values.sort(key=lambda item: (-item.score, item.component_id, item.fingerprint))
    summary = {name: len(values) for name, values in queues.items()}
    hash_payload = {
        "current_round": current_round,
        "queues": {
            name: [item.model_dump(mode="json") for item in values]
            for name, values in queues.items()
        },
        "auto_code_generation": False,
        "mechanism_opportunities": [
            item.model_dump(mode="json") for item in mechanism_opportunities
        ],
    }
    plan_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PaperAdapterImplementationPlan(
        current_round=current_round,
        **queues,
        summary=summary,
        mechanism_opportunities=mechanism_opportunities,
        plan_hash=plan_hash,
    )


def _implementation_opportunities(
    report: PaperMechanismClusterReport,
) -> list[AdapterCoverageOpportunity]:
    return [
        item
        for item in report.implementation_opportunities
        if item.implementation_status == "adapter_required"
    ]


def _mechanism_context_by_paper(
    report: PaperMechanismClusterReport | None,
) -> dict[str, tuple[str, str | None, float]]:
    if report is None:
        return {}
    context: dict[str, tuple[str, str | None, float]] = {}
    for match in report.matches:
        if match.cluster_id is None or match.match_type == "unresolved" or match.conflicts:
            continue
        candidate = (
            match.cluster_id,
            match.adapter_family,
            match.confidence_score,
        )
        current = context.get(match.paper_id)
        if current is None or (candidate[2], candidate[0]) > (current[2], current[0]):
            context[match.paper_id] = candidate
    return context


__all__ = ["PaperAdapterImplementationPlanner"]
