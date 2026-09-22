"""The observed-trace → next-decision closer (Prompt-18J §7)."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.detection_error_delta import build_detection_error_delta
from yolo_agent.core.detection_error_profile import (
    DetectionErrorProfile,
    FalseNegativeFacts,
    FalsePositiveFacts,
    GlobalMetrics,
    LocalizationFacts,
    PerClassMetrics,
    ScaleMetrics,
)
from yolo_agent.core.error_decision_trace import build_error_decision_trace
from yolo_agent.core.error_profile_evidence_gate import (
    evaluate_error_profile_evidence,
)
from yolo_agent.core.error_trace_next import close_round
from yolo_agent.core.experiment_memory import ExperimentMemory
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def _spec() -> TaskSpec:
    return TaskSpec(
        class_names=["person"],
        primary_metric=MetricPriority(name="map50_95"),
    )


def _complete_profile(
    profile_id: str,
    candidate_id: str,
    *,
    map50_95: float,
    fn_total: int,
    fp_total: int,
) -> DetectionErrorProfile:
    """A profile with every section the evidence gate demands."""
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
        scale=ScaleMetrics(
            ap_small=0.3, ap_medium=0.5, ap_large=0.6, recall_small=0.4
        ),
        per_class=[
            PerClassMetrics(
                category_id=1,
                name="person",
                ap=map50_95,
                ap50=map50_95 + 0.1,
                precision=0.7,
                recall=0.6,
                support=10,
            )
        ],
        false_negative=FalseNegativeFacts(total=fn_total),
        false_positive=FalsePositiveFacts(total=fp_total),
        localization=LocalizationFacts(
            matched_iou_distribution={"0.75-0.85": 4},
            mean_matched_iou=0.82,
            localization_error_count=0,
            ap50_vs_ap75_gap=0.1,
        ),
        confidence=_confidence(),
    )


def _confidence():
    from yolo_agent.core.detection_error_profile import ConfidenceFacts

    return ConfidenceFacts(tp_confidence_histogram={"0.5-0.6": 3})


def test_close_round_records_observation_and_next_decision(tmp_path: Path) -> None:
    parent = _complete_profile(
        "prof-parent", "baseline", map50_95=0.300, fn_total=40, fp_total=50
    )
    candidate = _complete_profile(
        "prof-cand", "cand", map50_95=0.310, fn_total=36, fp_total=46
    )
    delta = build_detection_error_delta(candidate, parent)
    gate = evaluate_error_profile_evidence(
        baseline_profile=parent, candidate_profile=candidate, delta=delta
    )
    trace = build_error_decision_trace(candidate, gate, delta=delta)

    result = close_round(
        trace,
        delta,
        _spec(),
        ExperimentMemory(),
        artifact_path=tmp_path / "decision_trace.yaml",
    )

    assert result.trace.status == "observed"
    assert result.trace.observed is not None
    assert result.next_decision.decision == "promote"
    # §7: the written trace itself carries next_decision.
    assert result.trace.next_decision == "promote"

    # The artifact is written and reloadable.
    assert (tmp_path / "decision_trace.yaml").is_file()
    from yolo_agent.core.error_round_artifacts import (
        load_decision_trace_artifact,
    )
    written = load_decision_trace_artifact(
        tmp_path / "decision_trace.yaml"
    )
    assert written.next_decision == "promote"
    assert written.status == "observed"


def test_close_round_remembers_outcome_in_experiment_memory(tmp_path: Path) -> None:
    """A constraint-violating round (rollback) is remembered as blocked."""
    from yolo_agent.core.detection_error_delta import ResourceSnapshot

    parent = _complete_profile(
        "prof-parent", "baseline", map50_95=0.300, fn_total=40, fp_total=50
    )
    candidate = _complete_profile(
        "prof-cand", "cand", map50_95=0.310, fn_total=36, fp_total=56
    )
    # Give both sides real latency snapshots with a TaskSpec ceiling breach.
    parent = parent.model_copy(
        update={"resources": ResourceSnapshot(latency_ms=10.0)}
    )
    candidate = candidate.model_copy(
        update={"resources": ResourceSnapshot(latency_ms=14.0)}
    )
    constrained = _spec().model_copy(update={"max_latency_ms": 12.0})
    delta = build_detection_error_delta(candidate, parent)
    gate = evaluate_error_profile_evidence(
        baseline_profile=parent, candidate_profile=candidate, delta=delta
    )
    trace = build_error_decision_trace(candidate, gate, delta=delta)
    memory = ExperimentMemory()

    result = close_round(
        trace,
        delta,
        constrained,
        memory,
        selected_action_id="train.neck.p2",
        selected_parameters={"epochs": 30},
        parent_fingerprint="parent-1",
    )

    assert result.next_decision.decision == "rollback"
    # The rolled-back action with identical problem structure is now blocked.
    from yolo_agent.core.error_trace_next import _movement_profile

    blocked = memory.check_repeat(
        profile=_movement_profile(result.trace, delta),
        action_id="train.neck.p2",
        parameters={"epochs": 30},
        parent_fingerprint="parent-1",
    )
    assert blocked is not None
    assert blocked.outcome == "rolled_back"
