"""Diagnosis-bound promotion policy for the sampling acceptance recipe."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.paper_auto_optimization_schemas import PaperPairedDelta
from yolo_agent.certification.runner import (
    BackendEvaluation,
    CertificationRecipe,
    _promotion_result,
)
from yolo_agent.certification.schemas import CertificationPromotionResult
from yolo_agent.core.paired_bootstrap import PairedBootstrapReport
from yolo_agent.core.paired_experiment import PairedExperimentResult


SAMPLING_ACCEPTANCE_RECIPE = CertificationRecipe(
    recipe_id="small_object_sampling",
    changed_variable="data.sampling_policy",
    overrides={
        "data.sampling_policy": {
            "small_object_boost": 2.0,
            "class_balance": True,
            "rare_class_boost": 1.5,
            "fn_heavy_class_ids": [0],
            "target_class_ids": [0],
            "max_oversampling_ratio": 3.0,
        }
    },
    execution_class="component_adapter",
    target_class_names=["object"],
    primary_metric="ap_small",
)


def evaluate_sampling_promotion(
    *,
    stage_id: str,
    paired: PairedExperimentResult,
    control: BackendEvaluation,
    candidate: BackendEvaluation,
    bootstrap: PairedBootstrapReport,
    paired_result_path: Path | str,
) -> tuple[CertificationPromotionResult, PaperPairedDelta]:
    """Evaluate the primary diagnosis metrics and persist the paired result."""
    promotion = _promotion_result(
        stage_id,
        SAMPLING_ACCEPTANCE_RECIPE,
        paired,
        control,
        candidate,
        bootstrap,
    )
    path = paired.to_json(paired_result_path)
    recall_key = "per_class_ar/object"
    fn_key = "false_negative/object"
    summary = PaperPairedDelta(
        stage_id=stage_id,  # type: ignore[arg-type]
        baseline_id=str(paired.baseline_candidate_id or ""),
        candidate_id=paired.candidate_id,
        verified=paired.verified,
        protocol_match=paired.protocol_match_status == "matched",
        ap_small_delta=promotion.metric_deltas.get("ap_small"),
        target_recall_delta=promotion.metric_deltas.get(recall_key),
        false_negative_delta=promotion.error_fact_deltas.get(fn_key),
        overall_map50_95_delta=promotion.metric_deltas.get("map50_95"),
        latency_delta_ms=(
            paired.latency_delta.paired_delta
            if paired.latency_delta is not None
            else None
        ),
        model_size_delta_mb=(
            paired.model_size_delta.paired_delta
            if paired.model_size_delta is not None
            else None
        ),
        paired_bootstrap_ci=(
            (
                paired.paired_bootstrap_ci.confidence_interval_low,
                paired.paired_bootstrap_ci.confidence_interval_high,
            )
            if paired.paired_bootstrap_ci is not None
            else None
        ),
        rejection_reasons=list(promotion.rejection_reasons),
        result_hash=paired.result_hash,
    )
    if not path.is_file():
        raise RuntimeError("paired experiment result was not persisted")
    return promotion, summary


__all__ = [
    "SAMPLING_ACCEPTANCE_RECIPE",
    "evaluate_sampling_promotion",
]
