from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.paper_auto_optimization_promotion import (
    evaluate_paper_recipe_promotion,
)
from yolo_agent.certification.paper_auto_optimization_tracks import acceptance_recipe
from yolo_agent.certification.runner import BackendEvaluation
from yolo_agent.core.matched_baseline import (
    MatchedBaselineControl,
    MatchedBaselineKey,
    PairedMetricDelta,
)
from yolo_agent.core.paired_bootstrap import (
    PairedBootstrapConfig,
    PairedBootstrapMetric,
    PairedBootstrapReport,
)
from yolo_agent.core.paired_experiment import (
    PairedErrorFactDelta,
    PairedExperimentResult,
)


def _key() -> MatchedBaselineKey:
    return MatchedBaselineKey(
        dataset_manifest_sha256="dataset",
        protocol_hash="protocol",
        subset_manifest_sha256="subset",
        seed="1",
        epochs=3,
        fidelity="pilot_3",
        batch_policy_hash="batch",
        ultralytics_version="8.4.0",
        imgsz=640,
        eval_protocol_hash="eval",
        split="val2017",
    )


def _metric(name: str, delta: float) -> PairedMetricDelta:
    key = _key()
    return PairedMetricDelta(
        metric_name=name,
        baseline_value=0.3,
        candidate_value=0.3 + delta,
        paired_delta=delta,
        effect_delta=delta,
        higher_is_better=True,
        baseline_run_id="run",
        baseline_candidate_id="baseline",
        baseline_node_id="baseline-node",
        candidate_run_id="run",
        candidate_id="loss.quality.correlation",
        candidate_node_id="candidate-node",
        baseline_source="test",
        candidate_source="test",
        match_key=key,
        match_key_hash=key.match_key_hash,
    )


def _paired(error_effect: float) -> PairedExperimentResult:
    key = _key()
    metrics = {
        "map50_95": _metric("map50_95", 0.02),
        "per_class_ap/object": _metric("per_class_ap/object", 0.03),
        "latency_ms": _metric("latency_ms", 0.0),
        "model_size_mb": _metric("model_size_mb", 0.0),
    }
    return PairedExperimentResult(
        run_id="run",
        candidate_id="loss.quality.correlation",
        candidate_node_id="candidate-node",
        baseline_candidate_id="baseline",
        baseline_node_id="baseline-node",
        protocol_match_status="matched",
        matched_control=MatchedBaselineControl(
            candidate_run_id="run",
            candidate_id="loss.quality.correlation",
            candidate_node_id="candidate-node",
            baseline_run_id="run",
            baseline_candidate_id="baseline",
            baseline_node_id="baseline-node",
            match_key=key,
            matched=True,
            status="matched",
        ),
        metric_deltas=metrics,
        target_error_fact_deltas=[
            PairedErrorFactDelta(
                fact_key="localization_heavy_class|object|||localization_error",
                fact_type="localization_heavy_class",
                subject="object",
                metric_name="localization_error",
                baseline_value=4,
                candidate_value=4 - error_effect,
                paired_delta=-error_effect,
                effect_delta=error_effect,
                higher_is_better=False,
                improved=error_effect > 0,
                baseline_node_id="baseline-node",
                candidate_node_id="candidate-node",
                match_key_hash=key.match_key_hash,
            )
        ],
        latency_delta=metrics["latency_ms"],
        model_size_delta=metrics["model_size_mb"],
        verified=True,
    )


def _bootstrap(tmp_path: Path) -> PairedBootstrapReport:
    return PairedBootstrapReport(
        status="completed",
        baseline_predictions=tmp_path / "baseline.json",
        candidate_predictions=tmp_path / "candidate.json",
        ground_truth=tmp_path / "ground_truth.json",
        baseline_predictions_sha256="baseline",
        candidate_predictions_sha256="candidate",
        ground_truth_sha256="ground-truth",
        matched_image_count=4,
        protocol_hash="protocol",
        config=PairedBootstrapConfig(minimum_images=4),
        overall=PairedBootstrapMetric(
            metric_name="diagnostic_map50",
            baseline_value=0.3,
            candidate_value=0.32,
            observed_delta=0.02,
            confidence_interval_low=0.0,
            confidence_interval_high=0.04,
            probability_improvement=0.9,
            direction="inconclusive",
        ),
    )


def _evaluation(tmp_path: Path, latency: float = 10.0) -> BackendEvaluation:
    return BackendEvaluation(
        eval_path=tmp_path / "eval.json",
        predictions_path=tmp_path / "predictions.json",
        error_report_path=tmp_path / "errors.json",
        latency_ms=latency,
        model_size_mb=5.0,
    )


def test_recipe_promotion_uses_localization_error_target(tmp_path: Path) -> None:
    promotion, summary = evaluate_paper_recipe_promotion(
        recipe=acceptance_recipe("loss.quality.correlation"),
        stage_id="pilot_3",
        paired=_paired(1.0),
        control=_evaluation(tmp_path),
        candidate=_evaluation(tmp_path),
        bootstrap=_bootstrap(tmp_path),
        paired_result_path=tmp_path / "paired.json",
    )

    assert promotion.passed
    assert promotion.checks["target_error_facts_improved"]
    assert summary.track_id == "auxiliary_loss"
    assert summary.target_error_fact_deltas


def test_recipe_promotion_rejects_non_improving_target_fact(tmp_path: Path) -> None:
    promotion, _ = evaluate_paper_recipe_promotion(
        recipe=acceptance_recipe("loss.quality.correlation"),
        stage_id="pilot_3",
        paired=_paired(-1.0),
        control=_evaluation(tmp_path),
        candidate=_evaluation(tmp_path),
        bootstrap=_bootstrap(tmp_path),
        paired_result_path=tmp_path / "paired.json",
    )

    assert not promotion.passed
    assert "target_error_facts_improved" in promotion.rejection_reasons
