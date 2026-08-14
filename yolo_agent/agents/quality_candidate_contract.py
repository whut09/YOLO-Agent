"""Canonical routing contract for YOLO26 quality-alignment candidates."""

from __future__ import annotations

from collections.abc import Iterable

from yolo_agent.agents.candidate_generator import CandidateEvaluationContract


QUALITY_COMPONENT_IDS = frozenset(
    {
        "loss.quality.correlation",
        "loss.quality.pseudo_iou",
    }
)
QUALITY_CHANGED_VARIABLES = {
    "loss.quality.correlation": "loss.correlation.weight",
    "loss.quality.pseudo_iou": "loss.pseudo_iou.weight",
}
QUALITY_EVIDENCE_ARTIFACTS = {
    "loss.quality.correlation": "auxiliary_loss_correlation_evidence",
    "loss.quality.pseudo_iou": "auxiliary_loss_pseudo_iou_evidence",
}
QUALITY_LOCALIZATION_METRICS = frozenset(
    {
        "ap75",
        "confidence_iou_correlation",
        "localization_error_rate",
        "confidence_localization_mismatch",
    }
)


def quality_evaluation_contract_errors(
    components: Iterable[str],
    contract: CandidateEvaluationContract,
) -> list[str]:
    """Return stable errors for a quality candidate missing evaluation guards."""
    if not QUALITY_COMPONENT_IDS.intersection(components):
        return []
    errors: list[str] = []
    if contract.primary_metric != "map50_95":
        errors.append("quality_primary_metric_must_be_map50_95")
    if not QUALITY_LOCALIZATION_METRICS.intersection(contract.evaluation_metrics):
        errors.append("quality_localization_metric_missing")
    guards = {*contract.stop_conditions, *contract.promotion_requirements}
    if not any("latency_guard" in item for item in guards):
        errors.append("quality_latency_guard_missing")
    if not any("model_size_guard" in item for item in guards):
        errors.append("quality_model_size_guard_missing")
    return errors


__all__ = [
    "QUALITY_CHANGED_VARIABLES",
    "QUALITY_COMPONENT_IDS",
    "QUALITY_EVIDENCE_ARTIFACTS",
    "QUALITY_LOCALIZATION_METRICS",
    "quality_evaluation_contract_errors",
]
