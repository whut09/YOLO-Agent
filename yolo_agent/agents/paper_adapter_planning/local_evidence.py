"""Local posterior scoring with stronger negative-evidence authority than freshness."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from yolo_agent.core.policy_memory import CONFIDENCE_RANK, PolicyMemoryRecord
from yolo_agent.research.component_aliases import normalize_component_id


class LocalImplementationEvidenceAssessment(BaseModel):
    score: float = 0.0
    record_count: int = 0
    negative_evidence: bool = False
    reasons: list[str] = Field(default_factory=list)


def assess_local_component_evidence(
    component_id: str,
    records: Iterable[PolicyMemoryRecord],
    *,
    dataset_signature: str | None,
    positive_max_weight: float,
    negative_max_penalty: float,
) -> LocalImplementationEvidenceAssessment:
    matching = [item for item in records if _record_matches_component(item, component_id)]
    if not matching:
        return LocalImplementationEvidenceAssessment(reasons=["no_local_component_evidence"])
    weighted_effects: list[tuple[float, float]] = []
    reasons: list[str] = []
    for record in matching:
        confidence_weight = (CONFIDENCE_RANK[record.confidence] + 1) / 3.0
        seed_weight = min(max(record.seed_count, 1) / 3.0, 1.0)
        dataset_weight = _dataset_weight(record, dataset_signature)
        weight = confidence_weight * seed_weight * dataset_weight
        effect = _effect_sign(record)
        weighted_effects.append((effect, weight))
        reasons.append(
            f"local_evidence:{record.evidence_status}:{record.confidence}:seeds={record.seed_count}"
        )
    denominator = sum(weight for _, weight in weighted_effects)
    mean_effect = (
        sum(effect * weight for effect, weight in weighted_effects) / denominator
        if denominator > 0
        else 0.0
    )
    if mean_effect < 0:
        score = abs(mean_effect) * negative_max_penalty
    else:
        score = mean_effect * positive_max_weight
    return LocalImplementationEvidenceAssessment(
        score=score,
        record_count=len(matching),
        negative_evidence=mean_effect < 0,
        reasons=reasons,
    )


def _record_matches_component(record: PolicyMemoryRecord, component_id: str) -> bool:
    target = normalize_component_id(component_id)
    fingerprint = record.action_fingerprint
    if fingerprint is not None and any(
        normalize_component_id(item) == target for item in fingerprint.component_ids
    ):
        return True
    return target in normalize_component_id(record.action)


def _dataset_weight(record: PolicyMemoryRecord, dataset_signature: str | None) -> float:
    if dataset_signature is None:
        return 1.0
    fingerprint = record.action_fingerprint
    observed = fingerprint.dataset_signature if fingerprint is not None else record.dataset_version
    return 1.0 if observed == dataset_signature else 0.25


def _effect_sign(record: PolicyMemoryRecord) -> float:
    if record.failure_reason or record.evidence_status == "failed":
        return -1.0
    effect = record.effect_delta
    if effect is None:
        effect = record.full_delta
    if effect is None:
        effect = record.pilot_10_delta
    if effect is None:
        effect = record.pilot_3_delta
    if effect is None or effect == 0:
        return 0.0
    return 1.0 if effect > 0 else -1.0


__all__ = [
    "LocalImplementationEvidenceAssessment",
    "assess_local_component_evidence",
]
