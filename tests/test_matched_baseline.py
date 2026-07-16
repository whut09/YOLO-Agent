"""Strict matched baseline control tests."""

from __future__ import annotations

import pytest

from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.matched_baseline import paired_metric_delta


def _record(*, role: str, value: float, **overrides: object) -> MetricEvidence:
    values = {
        "candidate_id": "baseline" if role == "baseline_reference" else "candidate",
        "node_id": "node_baseline" if role == "baseline_reference" else "node_candidate",
        "run_id": "run-1",
        "origin_run_id": "baseline-run" if role == "baseline_reference" else "run-1",
        "evidence_role": role,
        "inheritance_depth": 1 if role == "baseline_reference" else 0,
        "dataset_manifest_sha256": "dataset-sha",
        "subset_manifest_sha256": "subset-sha",
        "seed": 42,
        "epochs": 10,
        "fidelity": "pilot_10",
        "batch_policy_hash": "batch-policy",
        "ultralytics_version": "9.0.0",
        "imgsz": 640,
        "eval_protocol_hash": "eval-protocol",
        "split": "val2017",
        "metric_name": "map50_95",
        "value": value,
        "source": "test",
        "verified": True,
    }
    values.update(overrides)
    return MetricEvidence.model_validate(values)


def test_exact_match_produces_paired_delta() -> None:
    control, delta = paired_metric_delta(
        _record(role="current_observation", value=0.42),
        [_record(role="baseline_reference", value=0.40)],
    )
    assert control.matched is True
    assert delta is not None
    assert delta.paired_delta == pytest.approx(0.02)
    assert delta.effect_delta == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("subset_manifest_sha256", "other", "subset_manifest_sha256_mismatch"),
        ("seed", 7, "seed_mismatch"),
        ("epochs", 3, "epochs_mismatch"),
        ("fidelity", "pilot_3", "fidelity_mismatch"),
        ("batch_policy_hash", "other", "batch_policy_hash_mismatch"),
        ("ultralytics_version", "8.3.0", "ultralytics_version_mismatch"),
        ("imgsz", 1280, "baseline_missing_imgsz_not_fixed_640"),
        ("eval_protocol_hash", "other", "eval_protocol_hash_mismatch"),
    ],
)
def test_any_control_identity_mismatch_blocks_delta(field: str, value: object, reason: str) -> None:
    control, delta = paired_metric_delta(
        _record(role="current_observation", value=0.42),
        [_record(role="baseline_reference", value=0.40, **{field: value})],
    )
    assert delta is None
    assert control.status == "needs_matched_baseline"
    assert reason in control.mismatch_reasons


def test_inherited_context_cannot_masquerade_as_control() -> None:
    inherited = _record(
        role="inherited_context",
        value=0.40,
        candidate_id="baseline",
        origin_run_id="parent",
        inheritance_depth=1,
    )
    control, delta = paired_metric_delta(_record(role="current_observation", value=0.42), [inherited])
    assert delta is None
    assert "baseline_not_explicit_reference" in control.mismatch_reasons


def test_missing_dimension_never_falls_back_to_absolute_subtraction() -> None:
    candidate = _record(role="current_observation", value=0.42, subset_manifest_sha256=None)
    control, delta = paired_metric_delta(candidate, [_record(role="baseline_reference", value=0.40)])
    assert delta is None
    assert control.missing_dimensions == ["subset_manifest_sha256"]


def test_even_matching_non_640_runs_are_not_comparable() -> None:
    control, delta = paired_metric_delta(
        _record(role="current_observation", value=0.42, imgsz=1280),
        [_record(role="baseline_reference", value=0.40, imgsz=1280)],
    )
    assert delta is None
    assert "imgsz_not_fixed_640" in control.missing_dimensions
