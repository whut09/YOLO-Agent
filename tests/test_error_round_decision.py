"""Tests for the TaskSpec-driven round decision engine (Prompt-18J §2/§6)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yolo_agent.core.detection_error_delta import (
    ResourceSnapshot,
    build_detection_error_delta,
)
from yolo_agent.core.detection_error_profile import (
    DetectionErrorProfile,
    FalseNegativeFacts,
    FalsePositiveFacts,
    GlobalMetrics,
    ScaleMetrics,
)
from yolo_agent.core.error_round_decision import decide_next_round
from yolo_agent.core.task_spec import DatasetSpec, MetricPriority, TaskSpec


def _spec(**overrides) -> TaskSpec:
    kwargs: dict = {
        "class_names": ["person", "car"],
        "primary_metric": MetricPriority(name="map50_95", weight=1.0),
        "max_latency_ms": None,
        "max_model_size_mb": None,
    }
    kwargs.update(overrides)
    return TaskSpec(**kwargs)


def _profile(
    profile_id: str,
    candidate_id: str,
    *,
    map50_95: float,
    ap_small: float = 0.30,
    fn_total: int = 10,
    fp_total: int = 8,
    resources: ResourceSnapshot | None = None,
) -> DetectionErrorProfile:
    return DetectionErrorProfile(
        profile_id=profile_id,
        run_id="run-18j",
        candidate_id=candidate_id,
        gt_artifact="gt.json",
        dataset_manifest_hash="manifest-1",
        global_=GlobalMetrics(
            map50=map50_95 + 0.15,
            map50_95=map50_95,
            precision=0.7,
            recall=0.6,
        ),
        scale=ScaleMetrics(ap_small=ap_small, ap_medium=0.5, ap_large=0.6),
        false_negative=FalseNegativeFacts(total=fn_total),
        false_positive=FalsePositiveFacts(total=fp_total),
        resources=resources,
    )


def test_small_object_gain_with_fp_and_latency_regression_is_refused_promotion() -> None:
    """The §2 example: mAP up, AP_small up, but FP and latency regress.

    The engine must see the accuracy gain *and* the regressions, and refuse
    a plain promote when the latency breaches the TaskSpec ceiling.
    """
    parent = _profile(
        "prof-parent",
        "baseline",
        map50_95=0.300,
        ap_small=0.250,
        fn_total=40,
        fp_total=50,
        resources=ResourceSnapshot(latency_ms=10.0, params=3_000_000),
    )
    candidate = _profile(
        "prof-cand",
        "cand",
        map50_95=0.303,
        ap_small=0.297,
        fn_total=32,
        fp_total=56,
        resources=ResourceSnapshot(latency_ms=12.4, params=3_050_000),
    )
    delta = build_detection_error_delta(candidate, parent)

    # +24% latency with a 12ms ceiling and a genuine small-object gain:
    decision = decide_next_round(
        delta, _spec(max_latency_ms=12.0), budget_remaining=1
    )

    assert decision.decision == "rollback"
    codes = [item.code for item in decision.objections]
    assert "latency_constraint_violated" in codes
    # The engine still acknowledges the real gains it is rolling back.
    assert any("scale.ap_small" in item for item in decision.acknowledgements)


def test_objective_gain_with_soft_regressions_refines() -> None:
    """FP regression without a TaskSpec breach must refine, not promote."""
    parent = _profile(
        "prof-parent", "baseline", map50_95=0.300, fp_total=50
    )
    candidate = _profile(
        "prof-cand", "cand", map50_95=0.310, fp_total=56
    )
    delta = build_detection_error_delta(candidate, parent)

    decision = decide_next_round(delta, _spec())

    assert decision.decision == "refine"
    assert any(
        item.code == "non_objective_regressions" for item in decision.objections
    )


def test_clean_objective_gain_promotes() -> None:
    parent = _profile(
        "prof-parent", "baseline", map50_95=0.300, fp_total=50, fn_total=40
    )
    candidate = _profile(
        "prof-cand", "cand", map50_95=0.310, fp_total=46, fn_total=36
    )
    delta = build_detection_error_delta(candidate, parent)

    decision = decide_next_round(delta, _spec())

    assert decision.decision == "promote"
    assert decision.consumes_budget


def test_latency_breach_without_objective_gain_rejects() -> None:
    parent = _profile(
        "prof-parent", "baseline", map50_95=0.300,
        resources=ResourceSnapshot(latency_ms=10.0),
    )
    candidate = _profile(
        "prof-cand", "cand", map50_95=0.290,
        resources=ResourceSnapshot(latency_ms=13.0),
    )
    delta = build_detection_error_delta(candidate, parent)

    decision = decide_next_round(delta, _spec(max_latency_ms=12.0))

    assert decision.decision == "reject"
    assert not decision.consumes_budget


def test_unmatched_delta_can_only_collect_evidence() -> None:
    parent = _profile("prof-parent", "baseline", map50_95=0.30)
    candidate = _profile("prof-cand", "cand", map50_95=0.31)
    delta = build_detection_error_delta(candidate, parent)
    broken = delta.model_copy(update={"matched_evaluation": False})

    decision = decide_next_round(broken, _spec())

    assert decision.decision == "collect_evidence"


def test_missing_objective_metric_collects_evidence() -> None:
    parent = _profile("prof-parent", "baseline", map50_95=0.30)
    candidate = _profile("prof-cand", "cand", map50_95=0.31)
    delta = build_detection_error_delta(candidate, parent)
    # Drop the global section so the primary metric cannot be found.
    stripped = delta.model_copy(
        update={"sections": [s for s in delta.sections if s.section != "global"]}
    )

    decision = decide_next_round(stripped, _spec())

    assert decision.decision == "collect_evidence"


def test_budget_and_exhaustion_stops() -> None:
    parent = _profile("prof-parent", "baseline", map50_95=0.30)
    candidate = _profile("prof-cand", "cand", map50_95=0.31)
    delta = build_detection_error_delta(candidate, parent)

    assert decide_next_round(delta, _spec(), budget_remaining=0).decision == (
        "stop_budget"
    )
    assert decide_next_round(
        delta, _spec(), rounds_exhausted=True
    ).decision == "stop_exhausted"


def test_objective_drop_routes_to_data_side_actions() -> None:
    parent = _profile("prof-parent", "baseline", map50_95=0.31, fn_total=30)
    candidate = _profile("prof-cand", "cand", map50_95=0.29, fn_total=30)
    delta = build_detection_error_delta(candidate, parent)

    imbalanced = _spec(dataset=DatasetSpec(class_names=["a"], imbalance="high"))
    balanced = _spec(dataset=DatasetSpec(class_names=["a"], imbalance="low"))

    assert decide_next_round(delta, imbalanced).decision == "collect_data"
    assert decide_next_round(delta, balanced).decision == "request_annotation"


def test_decision_model_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        from yolo_agent.core.error_round_decision import RoundDecision

        RoundDecision(
            run_id="r",
            candidate_id="c",
            parent_candidate_id="p",
            decision="promote",
            unknown_field="x",
        )
