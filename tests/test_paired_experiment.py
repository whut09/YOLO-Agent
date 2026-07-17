"""Verified paired experiment result tests."""

from __future__ import annotations

import pytest

from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.paired_experiment import build_paired_experiment_result


IDENTITY = {
    "run_id": "run-1",
    "origin_run_id": "run-1",
    "inheritance_depth": 0,
    "dataset_manifest_sha256": "dataset",
    "protocol_hash": "protocol",
    "subset_manifest_sha256": "subset",
    "seed": 42,
    "epochs": 10,
    "fidelity": "pilot_10",
    "batch_policy_hash": "batch",
    "ultralytics_version": "9.0.0",
    "imgsz": 640,
    "eval_protocol_hash": "eval",
}


def _metric(metric_name: str, value: float | int | str, *, baseline: bool = False, **overrides: object) -> MetricEvidence:
    values = {
        **IDENTITY,
        "candidate_id": "baseline" if baseline else "candidate",
        "node_id": "node_baseline" if baseline else "node_candidate",
        "evidence_role": "baseline_reference" if baseline else "current_observation",
        "split": "runtime" if metric_name in {"latency_ms", "model_size_mb"} else "val2017",
        "metric_name": metric_name,
        "value": value,
        "source": "test",
        "verified": True,
    }
    values.update(overrides)
    return MetricEvidence.model_validate(values)


def _fact(value: float, *, baseline: bool = False, **overrides: object) -> ErrorFact:
    values = {
        **IDENTITY,
        "candidate_id": "baseline" if baseline else "candidate",
        "node_id": "node_baseline" if baseline else "node_candidate",
        "evidence_role": "baseline_reference" if baseline else "current_observation",
        "split": "val2017",
        "fact_type": "area_metric",
        "subject": "small",
        "area": "small",
        "metric_name": "ap_small",
        "value": value,
    }
    values.update(overrides)
    return ErrorFact.model_validate(values)


def _records() -> list[MetricEvidence]:
    return [
        _metric("map50_95", 0.40, baseline=True),
        _metric("map50_95", 0.42),
        _metric("latency_ms", 10.0, baseline=True),
        _metric("latency_ms", 10.4),
        _metric("model_size_mb", 5.0, baseline=True),
        _metric("model_size_mb", 5.2),
        _metric("bootstrap/diagnostic_map50_ci_low", 0.005),
        _metric("bootstrap/diagnostic_map50_ci_high", 0.031),
        _metric("bootstrap/diagnostic_map50_probability_improvement", 0.97),
        _metric("bootstrap/diagnostic_map50_direction", "stable_improvement"),
        _metric("bootstrap/matched_image_count", 5000),
    ]


def test_normal_current_run_pair_builds_verified_result() -> None:
    result = build_paired_experiment_result(
        run_id="run-1",
        candidate_id="candidate",
        candidate_node_id="node_candidate",
        metric_records=_records(),
        error_facts=[_fact(0.20, baseline=True), _fact(0.23)],
        target_error_facts=[{"fact_type": "area_metric", "subject": "small", "area": "small", "metric_name": "ap_small"}],
    )

    assert result.verified is True
    assert result.protocol_match_status == "matched"
    assert result.metric_deltas["map50_95"].paired_delta == pytest.approx(0.02)
    assert result.latency_delta is not None and result.latency_delta.paired_delta == pytest.approx(0.4)
    assert result.model_size_delta is not None and result.model_size_delta.paired_delta == pytest.approx(0.2)
    assert result.target_error_fact_deltas[0].effect_delta == pytest.approx(0.03)
    assert result.paired_bootstrap_ci is not None
    assert result.paired_bootstrap_ci.confidence_interval_low == pytest.approx(0.005)


def test_protocol_mismatch_blocks_paired_result() -> None:
    records = [
        record.model_copy(update={"protocol_hash": "other"}) if record.evidence_role == "baseline_reference" else record
        for record in _records()
    ]
    result = build_paired_experiment_result(
        run_id="run-1",
        candidate_id="candidate",
        candidate_node_id="node_candidate",
        metric_records=records,
        error_facts=[_fact(0.20, baseline=True, protocol_hash="other"), _fact(0.23)],
        target_error_facts=[{"fact_type": "area_metric", "subject": "small", "area": "small", "metric_name": "ap_small"}],
    )

    assert result.verified is False
    assert result.protocol_match_status == "mismatch"
    assert "protocol_hash_mismatch" in result.blockers


def test_inherited_baseline_reference_cannot_pollute_pair() -> None:
    records = [
        record.model_copy(update={"origin_run_id": "parent", "inheritance_depth": 1})
        if record.evidence_role == "baseline_reference"
        else record
        for record in _records()
    ]
    result = build_paired_experiment_result(
        run_id="run-1",
        candidate_id="candidate",
        candidate_node_id="node_candidate",
        metric_records=records,
        error_facts=[
            _fact(0.20, baseline=True, origin_run_id="parent", inheritance_depth=1),
            _fact(0.23),
        ],
        target_error_facts=[{"fact_type": "area_metric", "subject": "small", "area": "small", "metric_name": "ap_small"}],
    )

    assert result.verified is False
    assert result.metric_deltas == {}
    assert "baseline_not_current_run" in result.blockers
