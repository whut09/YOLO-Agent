"""Prompt-18I §8: capability promotion tests (evidence-driven, no README edits)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_detection_error_profile_builder as fixtures  # noqa: E402

from yolo_agent.core.detection_error_delta import build_detection_error_delta  # noqa: E402
from yolo_agent.core.detection_error_profile import ErrorProfileSource  # noqa: E402
from yolo_agent.core.detection_error_profile_builder import (  # noqa: E402
    build_detection_error_profile,
)
from yolo_agent.core.error_capability_promotion import (  # noqa: E402
    evaluate_error_loop_promotion,
    manifest_upgrade_allowed,
)
from yolo_agent.core.error_decision_trace import build_error_decision_trace  # noqa: E402
from yolo_agent.core.error_profile_evidence_gate import (  # noqa: E402
    evaluate_error_profile_evidence,
)
from yolo_agent.core.error_root_cause import derive_root_cause_hypotheses  # noqa: E402


@pytest.fixture()
def complete_round(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "gt.json").write_text(__import__("json").dumps(fixtures.GT))
    (data / "preds.json").write_text(__import__("json").dumps(fixtures.PREDICTIONS))
    baseline = build_detection_error_profile(
        ErrorProfileSource(gt_json=data / "gt.json", predictions_json=data / "preds.json",
                           run_id="run-1", candidate_id="baseline", split="val", role="baseline")
    )
    candidate = build_detection_error_profile(
        ErrorProfileSource(gt_json=data / "gt.json", predictions_json=data / "preds.json",
                           run_id="run-1", candidate_id="cand", split="val", role="candidate")
    )
    delta = build_detection_error_delta(candidate=candidate, parent=baseline)
    gate = evaluate_error_profile_evidence(
        baseline_profile=baseline, candidate_profile=candidate, delta=delta
    )
    trace = build_error_decision_trace(
        profile=candidate,
        gate=gate,
        delta=delta,
        hypotheses=derive_root_cause_hypotheses(candidate, delta=delta),
    )
    return baseline, candidate, delta, trace


def test_complete_round_promotes_both_capabilities(complete_round) -> None:
    baseline, candidate, delta, trace = complete_round
    # Since 18J, the decision capability requires the post-run observation:
    # close the round against the TaskSpec before judging promotion.
    from yolo_agent.core.error_trace_next import close_round
    from yolo_agent.core.experiment_memory import ExperimentMemory
    from yolo_agent.core.task_spec import MetricPriority, TaskSpec

    spec = TaskSpec(
        class_names=["person"],
        primary_metric=MetricPriority(name="map50_95"),
    )
    closed = close_round(trace, delta, spec, ExperimentMemory())

    record = evaluate_error_loop_promotion(
        run_id="run-1",
        baseline_profile=baseline,
        candidate_profile=candidate,
        delta=delta,
        decision_trace=closed.trace,
    )
    assert record.facts_capability == "executable"
    assert record.decision_capability == "executable"
    assert manifest_upgrade_allowed(record)
    assert record.reasons == []


def test_missing_baseline_blocks_facts_capability(complete_round) -> None:
    _, candidate, delta, trace = complete_round
    record = evaluate_error_loop_promotion(
        run_id="run-1",
        baseline_profile=None,
        candidate_profile=candidate,
        delta=delta,
        decision_trace=trace,
    )
    assert record.facts_capability == "not_promoted"
    assert record.decision_capability == "not_promoted"
    assert not manifest_upgrade_allowed(record)
    assert any("baseline" in reason for reason in record.reasons)


def test_unmatched_delta_blocks_promotion(complete_round) -> None:
    baseline, candidate, delta, trace = complete_round
    broken_delta = delta.model_copy(update={"matched_evaluation": False})
    record = evaluate_error_loop_promotion(
        run_id="run-1",
        baseline_profile=baseline,
        candidate_profile=candidate,
        delta=broken_delta,
        decision_trace=trace,
    )
    assert record.facts_capability == "not_promoted"
    assert record.decision_capability == "not_promoted"


def test_unconsumed_delta_blocks_decision_capability(complete_round) -> None:
    baseline, candidate, delta, _ = complete_round
    record = evaluate_error_loop_promotion(
        run_id="run-1",
        baseline_profile=baseline,
        candidate_profile=candidate,
        delta=delta,
        decision_trace=None,
    )
    assert record.facts_capability == "executable"
    assert record.decision_capability == "not_promoted"
    assert not manifest_upgrade_allowed(record)
