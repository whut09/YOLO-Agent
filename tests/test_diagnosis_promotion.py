"""Diagnosis-bound promotion gate tests."""

from __future__ import annotations

from yolo_agent.agents.diagnosis_promotion import DiagnosisPromotionGate
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import MetricEvidence


MATCHED = {
    "dataset_manifest_sha256": "dataset",
    "subset_manifest_sha256": "subset",
    "seed": 42,
    "epochs": 10,
    "fidelity": "pilot_10",
    "batch_policy_hash": "batch",
    "ultralytics_version": "9.0.0",
    "imgsz": 640,
    "eval_protocol_hash": "eval",
    "split": "val2017",
}


def _metric(candidate: str, node: str, name: str, value: float, *, baseline: bool = False) -> MetricEvidence:
    return MetricEvidence(
        run_id="run",
        candidate_id=candidate,
        node_id=node,
        metric_name=name,
        value=value,
        evidence_role="baseline_reference" if baseline else "current_observation",
        source="test",
        validator="test",
        **MATCHED,
    )


def _fact(
    candidate: str,
    node: str,
    fact_type: str,
    class_name: str,
    *,
    value: float | None = None,
    count: int | None = None,
    baseline: bool = False,
) -> ErrorFact:
    return ErrorFact(
        run_id="run",
        candidate_id=candidate,
        node_id=node,
        fact_type=fact_type,  # type: ignore[arg-type]
        subject=class_name,
        class_name=class_name,
        metric_name="per_class_ap" if value is not None else "false_negative_count",
        value=value,
        count=count,
        evidence_role="baseline_reference" if baseline else "current_observation",
        dataset_version="coco2017",
        **MATCHED,
    )


def _records(*, ap_small: float = 0.225, map_value: float = 0.398, latency: float = 10.3, size: float = 5.2) -> list[MetricEvidence]:
    return [
        _metric("baseline", "control", "ap_small", 0.21, baseline=True),
        _metric("candidate", "candidate-node", "ap_small", ap_small),
        _metric("baseline", "control", "per_class_ap/bottle", 0.20, baseline=True),
        _metric("candidate", "candidate-node", "per_class_ap/bottle", 0.215),
        _metric("baseline", "control", "map50_95", 0.40, baseline=True),
        _metric("candidate", "candidate-node", "map50_95", map_value),
        _metric("baseline", "control", "latency_ms", 10.0, baseline=True),
        _metric("candidate", "candidate-node", "latency_ms", latency),
        _metric("baseline", "control", "model_size_mb", 5.0, baseline=True),
        _metric("candidate", "candidate-node", "model_size_mb", size),
    ]


def _facts(*, candidate_fn: int = 90) -> list[ErrorFact]:
    return [
        _fact("baseline", "control", "false_negative_heavy_class", "bottle", count=100, baseline=True),
        _fact("candidate", "candidate-node", "false_negative_heavy_class", "bottle", count=candidate_fn),
    ]


def _targets() -> list[dict[str, object]]:
    return [
        {"fact_type": "area_metric", "subject": "small", "area": "small", "metric_name": "ap_small"},
        {"fact_type": "false_negative_heavy_class", "subject": "bottle", "class_name": "bottle"},
    ]


def _bootstrap_direction(metric_name: str, value: str) -> MetricEvidence:
    return MetricEvidence(
        run_id="run", candidate_id="candidate", node_id="candidate-node",
        metric_name=metric_name, value=value, source="paired_bootstrap",
        validator="paired_image_bootstrap", evidence_role="current_observation",
        dataset_version="coco2017", **MATCHED,
    )


def test_small_object_promotion_requires_all_diagnosis_guards() -> None:
    result = DiagnosisPromotionGate().evaluate(
        candidate_id="candidate",
        node_id="candidate-node",
        target_error_facts=_targets(),
        metric_records=_records(),
        error_facts=_facts(),
    )

    assert result.allowed is True
    assert {check.status for check in result.checks} == {"passed"}
    assert result.target_metric == "ap_small"
    assert result.related_classes == ["bottle"]


def test_arbitrary_map_gain_cannot_replace_target_ap_small_gain() -> None:
    result = DiagnosisPromotionGate().evaluate(
        candidate_id="candidate",
        node_id="candidate-node",
        target_error_facts=_targets(),
        metric_records=_records(ap_small=0.2101, map_value=0.41),
        error_facts=_facts(),
    )

    assert result.allowed is False
    assert any(reason.startswith("target_metric_improvement:") for reason in result.rejection_reasons)


def test_small_object_promotion_rejects_missing_bound_classes_and_fn() -> None:
    result = DiagnosisPromotionGate().evaluate(
        candidate_id="candidate",
        node_id="candidate-node",
        target_error_facts=[{"fact_type": "area_metric", "subject": "small", "metric_name": "ap_small"}],
        metric_records=_records(),
        error_facts=[],
    )

    assert result.allowed is False
    assert any(reason.startswith("related_class_ap:") for reason in result.rejection_reasons)
    assert any(reason.startswith("false_negative_reduction:") for reason in result.rejection_reasons)


def test_small_object_promotion_rejects_fn_or_resource_regression() -> None:
    fn_result = DiagnosisPromotionGate().evaluate(
        candidate_id="candidate",
        node_id="candidate-node",
        target_error_facts=_targets(),
        metric_records=_records(),
        error_facts=_facts(candidate_fn=101),
    )
    resource_result = DiagnosisPromotionGate().evaluate(
        candidate_id="candidate",
        node_id="candidate-node",
        target_error_facts=_targets(),
        metric_records=_records(latency=11.0),
        error_facts=_facts(),
    )

    assert fn_result.allowed is False
    assert any(reason.startswith("false_negative_reduction:") for reason in fn_result.rejection_reasons)
    assert resource_result.allowed is False
    assert any(reason.startswith("latency_guard:") for reason in resource_result.rejection_reasons)


def test_measurement_noise_blocks_tiny_ap_small_gain() -> None:
    records = _records(ap_small=0.212)
    records.extend(
        [
            _metric("baseline-2", "control-2", "ap_small", 0.215, baseline=True).model_copy(update={"seed": 43}),
            _metric("baseline-3", "control-3", "ap_small", 0.205, baseline=True).model_copy(update={"seed": 44}),
        ]
    )

    result = DiagnosisPromotionGate().evaluate(
        candidate_id="candidate",
        node_id="candidate-node",
        target_error_facts=_targets(),
        metric_records=records,
        error_facts=_facts(),
    )

    assert result.allowed is False
    assert result.measurement_noise["ap_small"] > 0.002
    assert any(reason.startswith("target_metric_improvement:") for reason in result.rejection_reasons)


def test_paired_bootstrap_rejects_stable_target_class_regression() -> None:
    records = _records()
    records.extend(
        [
            _bootstrap_direction("bootstrap/diagnostic_map50_direction", "inconclusive"),
            _bootstrap_direction("bootstrap/class_ap50/bottle/direction", "stable_regression"),
        ]
    )
    result = DiagnosisPromotionGate().evaluate(
        candidate_id="candidate", node_id="candidate-node",
        target_error_facts=_targets(), metric_records=records, error_facts=_facts(),
    )
    assert result.allowed is False
    assert any(reason.startswith("paired_bootstrap_guard:") for reason in result.rejection_reasons)


def test_paired_bootstrap_stable_improvement_remains_single_seed_guard_evidence() -> None:
    records = _records()
    records.extend(
        [
            _bootstrap_direction("bootstrap/diagnostic_map50_direction", "stable_improvement"),
            _bootstrap_direction("bootstrap/class_ap50/bottle/direction", "stable_improvement"),
            _bootstrap_direction("bootstrap/single_seed_only", "true"),
        ]
    )
    result = DiagnosisPromotionGate().evaluate(
        candidate_id="candidate", node_id="candidate-node",
        target_error_facts=_targets(), metric_records=records, error_facts=_facts(),
    )
    check = next(item for item in result.checks if item.check_id == "paired_bootstrap_guard")
    assert result.allowed is True
    assert check.status == "passed"
    assert "stable target-class improvement" in check.reason
