"""Transparent ranking score for paper adapter implementation work."""

from __future__ import annotations

from pydantic import BaseModel, Field

from yolo_agent.agents.paper_adapter_planning.diagnosis import (
    AdapterDiagnosisContext,
    AdapterDiagnosisMatch,
)
from yolo_agent.agents.paper_adapter_planning.local_evidence import (
    LocalImplementationEvidenceAssessment,
)
from yolo_agent.agents.paper_adapter_planning.policy import PaperAdapterPlanningPolicy
from yolo_agent.agents.paper_adapter_planning.runtime_hooks import RuntimeHookAssessment
from yolo_agent.agents.paper_adapter_planning.schemas import AdapterImplementationEstimate
from yolo_agent.research.component_aliases import ResolvedComponentAlias
from yolo_agent.research.schemas import PaperRecord


class ImplementationPriorityScore(BaseModel):
    total: float
    breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


def score_adapter_implementation(
    *,
    paper: PaperRecord,
    mapping: ResolvedComponentAlias,
    estimate: AdapterImplementationEstimate,
    diagnosis_context: AdapterDiagnosisContext,
    diagnosis_match: AdapterDiagnosisMatch,
    runtime_hook: RuntimeHookAssessment,
    local_evidence: LocalImplementationEvidenceAssessment,
    policy: PaperAdapterPlanningPolicy,
) -> ImplementationPriorityScore:
    breakdown: dict[str, float] = {}
    reasons = [*diagnosis_match.reasons, *runtime_hook.reasons, *local_evidence.reasons]
    breakdown["diagnosis"] = policy.diagnosis_match_weight if diagnosis_match.matched else 0.0
    breakdown["small_object_priority"] = (
        policy.ap_small_priority_weights.get(mapping.canonical_component_id, 0.0)
        if diagnosis_context.small_object_false_negative
        else 0.0
    )
    breakdown["compatibility"] = {
        "compatible": policy.compatibility_weight,
        "adapter_required": policy.compatibility_weight * 0.35,
        "unknown": 0.0,
        "incompatible": -policy.compatibility_weight,
    }[mapping.yolo26_compatibility]
    breakdown["runtime_hook"] = runtime_hook.score
    breakdown["official_code"] = policy.official_code_weight if paper.official_code_url else 0.0
    known_license = bool(paper.code_license and paper.code_license.strip().lower() != "unknown")
    breakdown["source_license"] = (
        policy.known_license_weight if known_license else policy.unknown_license_penalty
    )
    age = max(0, policy.current_year - paper.year)
    freshness_ratio = max(0.0, 1.0 - age / policy.freshness_window_years)
    breakdown["paper_freshness"] = freshness_ratio * policy.freshness_max_weight
    breakdown["implementation_cost"] = policy.implementation_cost_weights[
        estimate.implementation_cost
    ]
    breakdown["latency_cost"] = policy.deployment_cost_weights[
        estimate.expected_latency_cost
    ]
    breakdown["model_size_cost"] = policy.deployment_cost_weights[
        estimate.expected_model_size_cost
    ]
    breakdown["local_evidence"] = local_evidence.score
    if local_evidence.negative_evidence:
        reasons.append("local_negative_evidence_outweighs_freshness_prior")
    if paper.applicability == "direct_adapter_candidate":
        reasons.append("direct_adapter_candidate_is_prior_not_execution_status")
    return ImplementationPriorityScore(
        total=sum(breakdown.values()),
        breakdown=breakdown,
        reasons=list(dict.fromkeys(reasons)),
    )


__all__ = ["ImplementationPriorityScore", "score_adapter_implementation"]
