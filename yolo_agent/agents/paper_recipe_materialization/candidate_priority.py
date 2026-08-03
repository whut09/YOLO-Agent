"""Deterministic capacity score for certified paper recipe candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any

from yolo_agent.agents.paper_adapter_planning.local_evidence import (
    assess_local_component_evidence,
)
from yolo_agent.agents.paper_recipe_materialization.schemas import (
    PaperCandidatePriority,
    PaperRecipeCandidateInput,
)
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.policy_memory import PolicyMemoryRecord
from yolo_agent.research.component_aliases import normalize_component_id


_IMPLEMENTATION_COST = {"low": 8.0, "medium": 2.0, "high": -10.0, "unknown": -2.0}
_GPU_COST = {"low": 4.0, "medium": 0.0, "high": -8.0, "unknown": -1.0}
_DEPLOYMENT_COST = {"low": 3.0, "medium": 0.0, "high": -6.0, "unknown": -1.0}


def rank_materialized_candidate(
    item: PaperRecipeCandidateInput,
    *,
    current_error_facts: Iterable[ErrorFact],
    local_evidence: Iterable[PolicyMemoryRecord | dict[str, Any]],
    runtime_execution_ready: bool,
) -> PaperCandidatePriority:
    """Score a candidate without granting queue or training authority."""
    facts = list(current_error_facts)
    records = [record for record in local_evidence if isinstance(record, PolicyMemoryRecord)]
    context = item.planning_context
    covered_paper_ids = sorted({*item.prior.paper_ids, *context.covered_paper_ids})
    mechanism_confidence = (
        context.canonical_mechanism_confidence
        if context.canonical_mechanism_confidence is not None
        else item.prior.confidence
    )
    diagnosis_ratio = _diagnosis_match_ratio(item, facts)
    local_score, local_reasons = _local_posterior_score(
        item,
        facts=facts,
        records=records,
    )
    runtime_hook_score = _runtime_hook_score(
        runtime_execution_ready=runtime_execution_ready,
        hook_available=context.runtime_hook_available,
    )
    breakdown = {
        "current_error_facts": diagnosis_ratio * 30.0,
        "paper_coverage": min(math.log2(len(covered_paper_ids) + 1) * 12.0, 36.0),
        "canonical_mechanism_confidence": mechanism_confidence * 15.0,
        "runtime_hook": runtime_hook_score,
        "implementation_cost": _IMPLEMENTATION_COST[context.implementation_cost],
        "gpu_cost": _GPU_COST[context.expected_gpu_cost],
        "latency_cost": _DEPLOYMENT_COST[context.expected_latency_cost],
        "model_size_cost": _DEPLOYMENT_COST[context.expected_model_size_cost],
        "local_posterior": local_score,
    }
    reasons = [
        f"diagnosis_match_ratio:{diagnosis_ratio:.6f}",
        f"covered_compatible_papers:{len(covered_paper_ids)}",
        *local_reasons,
    ]
    if context.runtime_hook_available is False:
        reasons.append("runtime_hook_prior_unavailable")
    if local_score < 0:
        reasons.append("local_negative_posterior_has_priority_over_paper_prior")
    return PaperCandidatePriority(
        score=sum(breakdown.values()),
        breakdown=breakdown,
        candidate_fingerprint=_candidate_fingerprint(item),
        covered_paper_count=len(covered_paper_ids),
        covered_paper_ids=covered_paper_ids,
        mechanism_cluster_id=context.mechanism_cluster_id,
        canonical_mechanism_confidence=mechanism_confidence,
        reasons=list(dict.fromkeys(reasons)),
    )


def _diagnosis_match_ratio(
    item: PaperRecipeCandidateInput,
    facts: list[ErrorFact],
) -> float:
    current_terms = [_error_fact_terms(fact) for fact in facts]
    targets = [_target_terms(target) for target in item.prior.target_error_facts]
    if not current_terms or not targets:
        return 0.0
    matched = sum(
        1
        for target in targets
        if target and any(target.intersection(observed) for observed in current_terms)
    )
    return matched / len(targets)


def _error_fact_terms(fact: ErrorFact) -> set[str]:
    return {
        normalize_component_id(str(value))
        for value in (
            fact.fact_type,
            fact.subject,
            fact.class_name,
            fact.area,
            fact.metric_name,
        )
        if value
    }


def _target_terms(target: dict[str, Any]) -> set[str]:
    return {
        normalize_component_id(str(target[key]))
        for key in ("fact_type", "subject", "class_name", "area", "metric_name")
        if target.get(key)
    }


def _local_posterior_score(
    item: PaperRecipeCandidateInput,
    *,
    facts: list[ErrorFact],
    records: list[PolicyMemoryRecord],
) -> tuple[float, list[str]]:
    dataset_signature = next(
        (
            fact.dataset_manifest_sha256 or fact.dataset_version
            for fact in facts
            if fact.dataset_manifest_sha256 or fact.dataset_version
        ),
        None,
    )
    assessments = [
        assess_local_component_evidence(
            component_id,
            records,
            dataset_signature=dataset_signature,
            positive_max_weight=20.0,
            negative_max_penalty=-40.0,
        )
        for component_id in item.prior.component_ids
    ]
    if not assessments:
        return 0.0, ["no_local_component_evidence"]
    return (
        sum(assessment.score for assessment in assessments) / len(assessments),
        list(dict.fromkeys(
            reason
            for assessment in assessments
            for reason in assessment.reasons
        )),
    )


def _runtime_hook_score(
    *,
    runtime_execution_ready: bool,
    hook_available: bool | None,
) -> float:
    if not runtime_execution_ready or hook_available is False:
        return -12.0
    if hook_available is True:
        return 12.0
    return 8.0


def _candidate_fingerprint(item: PaperRecipeCandidateInput) -> str:
    payload = {
        "component_ids": sorted(item.prior.component_ids),
        "changed_variables": sorted(item.prior.suggested_changed_variables),
        "snapshot_hash": item.prior.research_snapshot_hash,
        "baseline_protocol": item.prior.baseline_protocol,
        "coupling_reason": item.prior.coupling_reason,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


__all__ = ["rank_materialized_candidate"]
