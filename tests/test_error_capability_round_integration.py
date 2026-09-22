"""The 18J decision chain feeds the capability promotion record (§8).

The ``error_delta_next_round`` capability may only become executable when a
real run's trace consumed a real delta *and* the TaskSpec verdict engine
acted on it.  These tests wire ``close_round`` (18J) into
``evaluate_error_loop_promotion`` (18I) and pin the joint semantics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_detection_error_profile_builder as fixtures  # noqa: E402

from yolo_agent.core.detection_error_delta import (  # noqa: E402
    ResourceSnapshot,
    build_detection_error_delta,
)
from yolo_agent.core.detection_error_profile import (  # noqa: E402
    ErrorProfileSource,
)
from yolo_agent.core.detection_error_profile_builder import (  # noqa: E402
    build_detection_error_profile,
)
from yolo_agent.core.error_capability_promotion import (  # noqa: E402
    evaluate_error_loop_promotion,
)
from yolo_agent.core.error_decision_trace import (  # noqa: E402
    build_error_decision_trace,
)
from yolo_agent.core.error_profile_evidence_gate import (  # noqa: E402
    evaluate_error_profile_evidence,
)
from yolo_agent.core.error_root_cause import (  # noqa: E402
    derive_root_cause_hypotheses,
)
from yolo_agent.core.error_trace_next import close_round  # noqa: E402
from yolo_agent.core.experiment_memory import ExperimentMemory  # noqa: E402
from yolo_agent.core.task_spec import (  # noqa: E402
    MetricPriority,
    TaskSpec,
)


@pytest.fixture()
def real_round(tmp_path: Path):
    """A real builder-produced round from synthetic GT/predictions."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "gt.json").write_text(json.dumps(fixtures.GT))
    (data / "preds.json").write_text(json.dumps(fixtures.PREDICTIONS))
    # The builder refuses to estimate mAP@[.5:.95] itself — it comes from the
    # run's official metrics.  Provide a deterministic synthetic one.
    (data / "official_base.json").write_text(
        json.dumps({"map50": 0.30, "map50_95": 0.20, "precision": 0.7, "recall": 0.6})
    )
    (data / "official_cand.json").write_text(
        json.dumps({"map50": 0.31, "map50_95": 0.22, "precision": 0.72, "recall": 0.62})
    )
    baseline = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=data / "gt.json", predictions_json=data / "preds.json",
            official_metrics_json=data / "official_base.json",
            run_id="run-18j", candidate_id="baseline", split="val", role="baseline",
        )
    )
    candidate = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=data / "gt.json", predictions_json=data / "preds.json",
            official_metrics_json=data / "official_cand.json",
            run_id="run-18j", candidate_id="cand", split="val", role="candidate",
        )
    )
    return baseline, candidate


def _spec() -> TaskSpec:
    return TaskSpec(
        class_names=["person"],
        primary_metric=MetricPriority(name="map50_95"),
    )


def _trace(baseline, candidate, delta):
    gate = evaluate_error_profile_evidence(
        baseline_profile=baseline, candidate_profile=candidate, delta=delta
    )
    return build_error_decision_trace(
        profile=candidate,
        gate=gate,
        delta=delta,
        hypotheses=derive_root_cause_hypotheses(candidate, delta=delta),
    )


def test_taskspec_verdict_flows_into_promotion(real_round) -> None:
    baseline, candidate = real_round
    delta = build_detection_error_delta(candidate=candidate, parent=baseline)
    trace = _trace(baseline, candidate, delta)

    closed = close_round(trace, delta, _spec(), ExperimentMemory())

    record = evaluate_error_loop_promotion(
        run_id="run-18j",
        baseline_profile=baseline,
        candidate_profile=candidate,
        delta=delta,
        decision_trace=closed.trace,
    )
    assert record.facts_capability == "executable"
    assert record.decision_capability == "executable"
    assert record.next_round_decision_consumed_delta is True


def test_rolled_back_round_still_proves_decision_consumption(real_round) -> None:
    """A constraint breach produces a rollback — the strongest possible
    proof that the verdict engine consumed more than the headline mAP."""
    baseline, candidate = real_round
    baseline = baseline.model_copy(
        update={"resources": ResourceSnapshot(latency_ms=10.0)}
    )
    candidate = candidate.model_copy(
        update={"resources": ResourceSnapshot(latency_ms=30.0)}
    )
    delta = build_detection_error_delta(candidate=candidate, parent=baseline)
    trace = _trace(baseline, candidate, delta)
    constrained = _spec().model_copy(update={"max_latency_ms": 12.0})

    closed = close_round(trace, delta, constrained, ExperimentMemory())
    assert closed.next_decision.decision == "rollback"

    record = evaluate_error_loop_promotion(
        run_id="run-18j",
        baseline_profile=baseline,
        candidate_profile=candidate,
        delta=delta,
        decision_trace=closed.trace,
    )
    assert record.decision_capability == "executable"


def test_open_round_without_observation_cannot_promote_decision(real_round) -> None:
    baseline, candidate = real_round
    delta = build_detection_error_delta(candidate=candidate, parent=baseline)
    trace = _trace(baseline, candidate, delta)

    record = evaluate_error_loop_promotion(
        run_id="run-18j",
        baseline_profile=baseline,
        candidate_profile=candidate,
        delta=delta,
        decision_trace=trace,
    )
    # The unobserved trace has no consumed-delta proof of a *next* decision.
    assert record.decision_capability != "executable"
