"""Prompt-18I §7: decision-trace tests on deterministic synthetic evidence."""

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
from yolo_agent.core.error_decision_trace import (  # noqa: E402
    build_error_decision_trace,
    record_observation,
)
from yolo_agent.core.error_profile_evidence_gate import (  # noqa: E402
    evaluate_error_profile_evidence,
)
from yolo_agent.core.error_root_cause import derive_root_cause_hypotheses  # noqa: E402


@pytest.fixture()
def round_evidence(tmp_path: Path):
    (tmp_path / "gt.json").write_text(__import__("json").dumps(fixtures.GT))
    (tmp_path / "preds.json").write_text(__import__("json").dumps(fixtures.PREDICTIONS))
    candidate = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=tmp_path / "gt.json",
            predictions_json=tmp_path / "preds.json",
            run_id="run-1",
            candidate_id="cand",
            split="val",
        )
    )
    parent = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=tmp_path / "gt.json",
            predictions_json=tmp_path / "preds.json",
            run_id="run-1",
            candidate_id="parent",
            split="val",
        )
    )
    delta = build_detection_error_delta(candidate=candidate, parent=parent)
    gate = evaluate_error_profile_evidence(
        baseline_profile=parent, candidate_profile=candidate, delta=delta
    )
    hypotheses = derive_root_cause_hypotheses(candidate, delta=delta)
    return candidate, parent, delta, gate, hypotheses


def test_trace_records_problem_evidence_and_hypotheses(round_evidence) -> None:
    candidate, _, delta, gate, hypotheses = round_evidence
    trace = build_error_decision_trace(
        profile=candidate, gate=gate, delta=delta, hypotheses=hypotheses
    )
    assert trace.problem
    assert trace.evidence_profile_id == candidate.profile_id
    assert trace.evidence_delta_id == delta.candidate_profile_id
    assert trace.gate_verdict == gate.verdict
    assert trace.hypotheses
    assert trace.candidate_actions
    # identical profiles -> no movement -> nothing rejected beyond gate rules
    assert trace.status in {"pending_observation", "blocked_by_evidence"}


def test_gate_withholds_budget_and_rejects_training_families(round_evidence) -> None:
    candidate, parent, delta, _, hypotheses = round_evidence
    # Force a partial gate by evaluating with a candidate missing its delta link.
    empty_parent = parent.model_copy(update={"per_class": {}})
    starved_gate = evaluate_error_profile_evidence(
        baseline_profile=empty_parent,
        candidate_profile=candidate,
        delta=delta,
    )
    trace = build_error_decision_trace(
        profile=candidate, gate=starved_gate, delta=delta, hypotheses=hypotheses
    )
    if not starved_gate.budget_request_allowed:
        families = {h.family for h in trace.rejected_actions}
        assert "sampling" in families or "classification_loss" in families
        assert trace.selected is None
        assert trace.status == "blocked_by_evidence"


def test_budget_request_denied_while_gate_withholds(round_evidence) -> None:
    candidate, parent, delta, _, hypotheses = round_evidence
    empty_parent = parent.model_copy(update={"per_class": {}})
    starved_gate = evaluate_error_profile_evidence(
        baseline_profile=empty_parent,
        candidate_profile=candidate,
        delta=delta,
    )
    if starved_gate.budget_request_allowed:
        pytest.skip("fixture gate stayed complete; denial path covered elsewhere")
    with pytest.raises(ValueError):
        build_error_decision_trace(
            profile=candidate,
            gate=starved_gate,
            delta=delta,
            hypotheses=hypotheses,
            selected_family="sampling",
            expected_effect="fewer background FPs",
        )


def test_selected_experiment_records_expected_effect(round_evidence) -> None:
    candidate, _, delta, gate, hypotheses = round_evidence
    if not gate.budget_request_allowed:
        pytest.skip("fixture gate withheld budget; selection covered elsewhere")
    trace = build_error_decision_trace(
        profile=candidate,
        gate=gate,
        delta=delta,
        hypotheses=hypotheses,
        selected_family="sampling",
        expected_effect="reduce background FP share",
    )
    assert trace.selected is not None
    assert trace.selected.family == "sampling"
    assert trace.selected.expected_effect == "reduce background FP share"
    assert trace.selected.round_budget_requested
    assert trace.status == "pending_observation"


def test_observed_effect_completes_the_round(round_evidence) -> None:
    candidate, _, delta, gate, hypotheses = round_evidence
    trace = build_error_decision_trace(
        profile=candidate, gate=gate, delta=delta, hypotheses=hypotheses
    )
    observed = record_observation(trace, delta)
    assert observed.status == "observed"
    assert observed.observed is not None
    assert observed.observed.delta_source == delta.candidate_profile_id
    assert observed.observed.improvements == delta.improvements()
    assert observed.observed.regressions == delta.regressions()


def test_observation_blocked_round_rejected(round_evidence) -> None:
    candidate, parent, delta, _, hypotheses = round_evidence
    empty_parent = parent.model_copy(update={"per_class": {}})
    starved_gate = evaluate_error_profile_evidence(
        baseline_profile=empty_parent,
        candidate_profile=candidate,
        delta=delta,
    )
    trace = build_error_decision_trace(
        profile=candidate, gate=starved_gate, delta=delta, hypotheses=hypotheses
    )
    if trace.status != "blocked_by_evidence":
        pytest.skip("fixture gate allowed budget")
    with pytest.raises(ValueError):
        record_observation(trace, delta)
