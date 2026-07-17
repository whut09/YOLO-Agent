"""Small verified paired-result fixtures shared by decision tests."""

from __future__ import annotations

from yolo_agent.core.matched_baseline import (
    MatchedBaselineControl,
    MatchedBaselineKey,
    PairedMetricDelta,
)
from yolo_agent.core.paired_experiment import PairedErrorFactDelta, PairedExperimentResult


def verified_paired_result(
    *,
    candidate_id: str,
    node_id: str,
    delta: float,
    run_id: str = "child",
    target_improved: bool = False,
    baseline_value: float = 0.40,
    target_baseline_value: float = 0.20,
    target_delta: float = 0.02,
    latency_baseline: float = 10.0,
    latency_delta: float = 0.0,
    model_size_baseline: float = 5.0,
    model_size_delta: float = 0.0,
    protocol_hash: str = "protocol-640",
) -> PairedExperimentResult:
    """Build a complete exact-pair result without filesystem or GPU dependencies."""
    key = MatchedBaselineKey(
        dataset_manifest_sha256="dataset",
        protocol_hash=protocol_hash,
        subset_manifest_sha256="subset",
        seed="42",
        epochs=10,
        fidelity="pilot_10",
        batch_policy_hash="batch-48",
        ultralytics_version="9.0.0",
        imgsz=640,
        eval_protocol_hash="coco-post-eval-v1",
        split="val2017",
    )
    control = MatchedBaselineControl(
        candidate_run_id=run_id,
        candidate_id=candidate_id,
        candidate_node_id=node_id,
        baseline_run_id=run_id,
        baseline_candidate_id="baseline",
        baseline_node_id="node_baseline",
        match_key=key,
        matched=True,
        status="matched",
    )

    def metric(metric_name: str, base: float, change: float) -> PairedMetricDelta:
        return PairedMetricDelta(
            metric_name=metric_name,
            baseline_value=base,
            candidate_value=base + change,
            paired_delta=change,
            effect_delta=change,
            higher_is_better=True,
            baseline_run_id=run_id,
            baseline_candidate_id="baseline",
            baseline_node_id="node_baseline",
            candidate_run_id=run_id,
            candidate_id=candidate_id,
            candidate_node_id=node_id,
            baseline_source="test:baseline",
            candidate_source="test:candidate",
            match_key=key,
            match_key_hash=key.match_key_hash,
        )

    primary = metric("map50_95", baseline_value, delta)
    latency = metric("latency_ms", latency_baseline, latency_delta)
    size = metric("model_size_mb", model_size_baseline, model_size_delta)
    facts = []
    if target_improved:
        facts.append(
            PairedErrorFactDelta(
                fact_key="area_metric|small|||small|ap_small",
                fact_type="area_metric",
                subject="small",
                metric_name="ap_small",
                baseline_value=target_baseline_value,
                candidate_value=target_baseline_value + target_delta,
                paired_delta=target_delta,
                effect_delta=target_delta,
                higher_is_better=True,
                improved=True,
                baseline_node_id="node_baseline",
                candidate_node_id=node_id,
                match_key_hash=key.match_key_hash,
            )
        )
    return PairedExperimentResult(
        run_id=run_id,
        candidate_id=candidate_id,
        candidate_node_id=node_id,
        baseline_candidate_id="baseline",
        baseline_node_id="node_baseline",
        protocol_match_status="matched",
        matched_control=control,
        metric_deltas={"map50_95": primary, "latency_ms": latency, "model_size_mb": size},
        target_error_fact_deltas=facts,
        latency_delta=latency,
        model_size_delta=size,
        verified=True,
    )
