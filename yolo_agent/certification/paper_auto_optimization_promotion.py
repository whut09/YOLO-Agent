"""Diagnosis-bound promotion policies for paper acceptance recipes."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.paper_auto_optimization_schemas import PaperPairedDelta
from yolo_agent.certification.paper_auto_optimization_tracks import (
    PAPER_ACCEPTANCE_RECIPES,
    PaperAcceptanceRecipe,
)
from yolo_agent.certification.runner import (
    BackendEvaluation,
    CertificationRecipe,
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
    """Compatibility wrapper for the original sampling-only suite."""
    recipe = next(
        item for item in PAPER_ACCEPTANCE_RECIPES if item.track_id == "sampling"
    )
    return evaluate_paper_recipe_promotion(
        recipe=recipe,
        stage_id=stage_id,
        paired=paired,
        control=control,
        candidate=candidate,
        bootstrap=bootstrap,
        paired_result_path=paired_result_path,
    )


def evaluate_paper_recipe_promotion(
    *,
    recipe: PaperAcceptanceRecipe,
    stage_id: str,
    paired: PairedExperimentResult,
    control: BackendEvaluation,
    candidate: BackendEvaluation,
    bootstrap: PairedBootstrapReport,
    paired_result_path: Path | str,
) -> tuple[CertificationPromotionResult, PaperPairedDelta]:
    """Require matched target facts and recipe-specific metric improvement."""
    metric_deltas = {
        name: paired.metric_deltas[name].paired_delta
        for name in recipe.target_metrics
        if name in paired.metric_deltas
    }
    missing_metrics = sorted(set(recipe.target_metrics) - set(metric_deltas))
    error_deltas = {
        item.fact_key: item.effect_delta for item in paired.target_error_fact_deltas
    }
    target_facts_improved = bool(error_deltas) and all(
        value > 0 for value in error_deltas.values()
    )
    map_delta = paired.metric_deltas.get("map50_95")
    latency_regression = _relative_regression(
        control.latency_ms,
        candidate.latency_ms,
    )
    size_regression = _relative_regression(
        control.model_size_mb,
        candidate.model_size_mb,
    )
    bootstrap_not_regressed = bool(
        bootstrap.status == "completed"
        and bootstrap.overall is not None
        and bootstrap.overall.direction != "stable_regression"
        and not bootstrap.stable_regressed_classes
    )
    checks = {
        "protocol_matched": paired.protocol_match_status == "matched" and paired.verified,
        "target_metrics_complete": not missing_metrics,
        "primary_metric_improved": metric_deltas.get(recipe.primary_metric, 0.0) > 0,
        "target_metrics_improved": bool(metric_deltas)
        and all(value > 0 for value in metric_deltas.values()),
        "target_error_facts_improved": target_facts_improved,
        "overall_map_guard": bool(
            map_delta is not None
            and map_delta.paired_delta >= -recipe.max_map_regression
        ),
        "latency_guard": latency_regression <= recipe.max_latency_regression,
        "model_size_guard": size_regression <= recipe.max_model_size_regression,
        "paired_bootstrap_not_regressed": bootstrap_not_regressed,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    promotion = CertificationPromotionResult(
        stage_id=stage_id,  # type: ignore[arg-type]
        passed=all(checks.values()),
        primary_metric=recipe.primary_metric,
        metric_deltas=metric_deltas,
        error_fact_deltas=error_deltas,
        guard_regressions={
            "latency": latency_regression,
            "model_size": size_regression,
        },
        checks=checks,
        rejection_reasons=reasons,
    )
    path = paired.to_json(paired_result_path)
    recall_key = "per_class_ar/object"
    fn_delta = next(
        (
            item.effect_delta
            for item in paired.target_error_fact_deltas
            if item.fact_type == "false_negative_heavy_class"
        ),
        None,
    )
    summary = PaperPairedDelta(
        stage_id=stage_id,  # type: ignore[arg-type]
        track_id=recipe.track_id,
        recipe_id=recipe.recipe_id,
        component_id=recipe.component_id,
        component_family=recipe.component_family,
        primary_metric=recipe.primary_metric,
        baseline_id=str(paired.baseline_candidate_id or ""),
        candidate_id=paired.candidate_id,
        verified=paired.verified,
        protocol_match=paired.protocol_match_status == "matched",
        ap_small_delta=promotion.metric_deltas.get("ap_small"),
        target_recall_delta=promotion.metric_deltas.get(recall_key),
        false_negative_delta=fn_delta,
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
        metric_deltas=dict(promotion.metric_deltas),
        target_error_fact_deltas=dict(promotion.error_fact_deltas),
        rejection_reasons=list(promotion.rejection_reasons),
        result_hash=paired.result_hash,
    )
    if not path.is_file():
        raise RuntimeError("paired experiment result was not persisted")
    return promotion, summary


def _relative_regression(control: float, candidate: float) -> float:
    return (candidate - control) / control if control > 0 else float("inf")


__all__ = [
    "SAMPLING_ACCEPTANCE_RECIPE",
    "evaluate_paper_recipe_promotion",
    "evaluate_sampling_promotion",
]
