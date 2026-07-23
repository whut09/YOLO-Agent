"""Cross-round ASHA scheduler tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.asha_scheduler import ASHAObservation, ASHAScheduler, ASHAStudyStore
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_queue import ExecutionQueue
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.round_execution_plan import build_asha_assignment_plan
from tests.paired_result_helpers import verified_paired_result


def _node(candidate_id: str) -> ExperimentNode:
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=640,
        batch=48,
    )
    return ExperimentNode(
        node_id=f"node_{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        command=command.display(),
        command_spec=command,
        changed_variables={"mosaic": candidate_id},
    )


def _register(scheduler: ASHAScheduler, candidate_id: str) -> None:
    control = _node("baseline_matched_control")
    control.changed_variables = {}
    scheduler.register_trial(
        trial_id=candidate_id,
        candidate_id=candidate_id,
        source_run_id=f"run-{candidate_id}",
        source_node=_node(candidate_id),
        target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
        baseline_control_node=control,
    )


def _report(scheduler: ASHAScheduler, candidate_id: str, stage: str, delta: float, improved: int = 0) -> None:
    node_id = f"node_{candidate_id}__{stage}"
    paired = verified_paired_result(
        candidate_id=candidate_id,
        node_id=node_id,
        delta=delta,
        target_improved=improved > 0,
    )
    scheduler.report(
        candidate_id,
        ASHAObservation(
            stage_id=stage,  # type: ignore[arg-type]
            node_id=node_id,
            seed=42,
            paired_delta=delta,
            paired_result_verified=True,
            paired_result_hash=paired.result_hash,
            protocol_match_status="matched",
            paired_experiment_result=paired,
            target_error_improved_count=improved,
            diagnosis_gate_passed=(None if stage == "pilot_3" else True),
        ),
    )


def _report_assignment(scheduler: ASHAScheduler, assignment, *, delta: float, improved: int = 0) -> None:
    node_id = f"node_{assignment.candidate_id}__{assignment.stage_id}_seed{assignment.seed_index}"
    paired = verified_paired_result(
        candidate_id=assignment.candidate_id,
        node_id=node_id,
        delta=delta,
        target_improved=improved > 0,
    )
    scheduler.report(
        assignment.trial_id,
        ASHAObservation(
            stage_id=assignment.stage_id,
            node_id=node_id,
            seed_index=assignment.seed_index,
            seed=assignment.seed,
            paired_delta=delta,
            paired_result_verified=True,
            paired_result_hash=paired.result_hash,
            protocol_match_status="matched",
            paired_experiment_result=paired,
            target_error_improved_count=improved,
            diagnosis_gate_passed=(None if assignment.stage_id == "pilot_3" else True),
        ),
    )


def _assert_assignment_is_only_queue_authority(scheduler: ASHAScheduler, assignment) -> None:
    trial = scheduler.study.trial(assignment.trial_id)
    plan = build_asha_assignment_plan(
        run_id=f"run-{assignment.assignment_id}",
        source_node=trial.source_node,
        stage_id=assignment.stage_id,
        epochs=assignment.epochs,
        fraction=assignment.fraction,
        seed=int(assignment.seed),
        seed_index=assignment.seed_index,
        assignment_id=assignment.assignment_id,
        baseline_control_node=trial.baseline_control_node,
    )
    queue = ExecutionQueue.from_round_execution_plan(plan.run_id, plan)
    assert queue.metadata["source_authority"] == "RoundExecutionPlan"
    assert queue.metadata["scheduler_mode"] == "external_asha"
    assert queue.metadata["asha_assignment_id"] == assignment.assignment_id
    candidates = [item for item in queue.items if not item.command.metadata.get("matched_baseline_control")]
    controls = [item for item in queue.items if item.command.metadata.get("matched_baseline_control")]
    assert {item.candidate_id for item in candidates} == {assignment.candidate_id}
    assert len(controls) == 1


def test_asha_requires_a_cohort_before_pilot_10() -> None:
    scheduler = ASHAScheduler.create("coco")
    for candidate_id, delta in [("a", 0.01), ("b", 0.03)]:
        _register(scheduler, candidate_id)
        _report(scheduler, candidate_id, "pilot_3", delta)

    assert scheduler.next_assignment() is None

    _register(scheduler, "c")
    _report(scheduler, "c", "pilot_3", 0.02)
    assignment = scheduler.next_assignment()

    assert assignment is not None
    assert assignment.candidate_id == "b"
    assert assignment.stage_id == "pilot_10"
    assert assignment.epochs == 10


def test_asha_finishes_all_registered_pilot_3_trials_before_ranking() -> None:
    scheduler = ASHAScheduler.create("coco")
    for candidate_id in ("a", "b", "c", "d"):
        _register(scheduler, candidate_id)
    for candidate_id, delta in (("a", 0.01), ("b", 0.03), ("c", 0.02)):
        assignment = scheduler.next_assignment()
        assert assignment is not None and assignment.candidate_id == candidate_id
        scheduler.mark_running(assignment)
        _report_assignment(scheduler, assignment, delta=delta)

    fourth = scheduler.next_assignment()
    assert fourth is not None
    assert fourth.stage_id == "pilot_3"
    assert fourth.candidate_id == "d"
    scheduler.mark_running(fourth)
    _report_assignment(scheduler, fourth, delta=0.05)

    promoted = scheduler.next_assignment()
    assert promoted is not None
    assert promoted.stage_id == "pilot_10"
    assert promoted.candidate_id == "d"


def test_asha_eliminates_non_positive_pilot_without_spending_ten_epochs() -> None:
    scheduler = ASHAScheduler.create("coco")
    _register(scheduler, "bad")

    _report(scheduler, "bad", "pilot_3", -0.001)

    trial = scheduler.study.trial("bad")
    assert trial.status == "eliminated"
    assert trial.eliminated_reason == "pilot_3_non_positive_paired_delta"


def test_pilot_10_requires_target_error_fact_improvement() -> None:
    scheduler = ASHAScheduler.create("coco")
    for candidate_id, delta in [("a", 0.01), ("b", 0.03), ("c", 0.02)]:
        _register(scheduler, candidate_id)
        _report(scheduler, candidate_id, "pilot_3", delta)
    assignment = scheduler.next_assignment()
    assert assignment is not None
    scheduler.mark_running(assignment)

    _report(scheduler, assignment.trial_id, "pilot_10", 0.04, improved=0)

    trial = scheduler.study.trial(assignment.trial_id)
    assert trial.status == "eliminated"
    assert trial.eliminated_reason == "pilot_10_target_error_fact_not_improved"


def test_full_budget_needs_confirmation_and_three_positive_seeds() -> None:
    scheduler = ASHAScheduler.create("coco")
    for candidate_id, delta in [("a", 0.01), ("b", 0.03), ("c", 0.02)]:
        _register(scheduler, candidate_id)
        _report(scheduler, candidate_id, "pilot_3", delta)
    pilot_10 = scheduler.next_assignment()
    assert pilot_10 is not None
    scheduler.mark_running(pilot_10)
    _report(scheduler, pilot_10.trial_id, "pilot_10", 0.04, improved=1)

    assert scheduler.next_assignment(confirm_full_run=False) is None
    seed_1 = scheduler.next_assignment(confirm_full_run=True)
    assert seed_1 is not None
    assert seed_1.stage_id == "candidate_full_seed_1"
    scheduler.mark_running(seed_1)
    _report(scheduler, seed_1.trial_id, "candidate_full_seed_1", 0.025, improved=1)

    for expected_index in (2, 3):
        assignment = scheduler.next_assignment(confirm_full_run=True)
        assert assignment is not None
        assert assignment.stage_id == "candidate_full_confirmation"
        assert assignment.seed_index == expected_index
        scheduler.mark_running(assignment)
        scheduler.report(
            assignment.trial_id,
            ASHAObservation(
                stage_id="candidate_full_confirmation",
                node_id=f"full-seed-{expected_index}",
                seed_index=expected_index,
                seed=assignment.seed,
                paired_delta=0.02,
                paired_result_verified=True,
                paired_experiment_result=verified_paired_result(
                    candidate_id=assignment.candidate_id,
                    node_id=f"full-seed-{expected_index}",
                    delta=0.02,
                ),
                diagnosis_gate_passed=True,
            ),
        )

    confirmed = scheduler.study.trial(seed_1.trial_id)
    assert confirmed.status == "confirmed"
    assert confirmed.confirmation_ci_low is not None and confirmed.confirmation_ci_low > 0


def test_three_positive_full_seeds_are_not_confirmed_when_interval_crosses_zero() -> None:
    scheduler = ASHAScheduler.create("coco")
    _register(scheduler, "unstable")
    scheduler.report(
        "unstable",
        ASHAObservation(
            stage_id="candidate_full_seed_1", node_id="seed-1", seed_index=1,
            seed=42, paired_delta=0.001, diagnosis_gate_passed=True,
            paired_result_verified=True,
            paired_experiment_result=verified_paired_result(
                candidate_id="unstable", node_id="seed-1", delta=0.001,
            ),
        ),
    )
    for seed_index, delta in ((2, 0.001), (3, 0.10)):
        scheduler.report(
            "unstable",
            ASHAObservation(
                stage_id="candidate_full_confirmation", node_id=f"seed-{seed_index}",
                    seed_index=seed_index, seed=41 + seed_index,
                    paired_delta=delta, diagnosis_gate_passed=True,
                    paired_result_verified=True,
                    paired_experiment_result=verified_paired_result(
                        candidate_id="unstable", node_id=f"seed-{seed_index}", delta=delta,
                    ),
                ),
        )
    trial = scheduler.study.trial("unstable")
    assert trial.status == "eliminated"
    assert trial.confirmation_ci_low is not None and trial.confirmation_ci_low < 0
    assert trial.eliminated_reason == "candidate_full_confirmation_confidence_interval_not_positive"


def test_asha_state_round_trips_between_auto_rounds(tmp_path: Path) -> None:
    store = ASHAStudyStore(tmp_path / "asha_state.yaml")
    scheduler = store.load_or_create("coco")
    _register(scheduler, "a")
    _report(scheduler, "a", "pilot_3", 0.01)
    store.save(scheduler)

    restored = store.load_or_create("coco")

    assert restored.study.trial("a").observation("pilot_3") is not None
    assert restored.study.confirmation_seeds == [42, 43, 44]


def test_asha_controls_complete_recoverable_three_seed_state_machine(tmp_path: Path) -> None:
    scheduler = ASHAScheduler.create("coco")
    for candidate_id in ("a", "b", "c"):
        _register(scheduler, candidate_id)

    store = ASHAStudyStore(tmp_path / "asha_state.yaml")
    pilot_deltas = {"a": 0.01, "b": 0.04, "c": 0.02}
    for index, candidate_id in enumerate(("a", "b", "c"), start=1):
        assignment = scheduler.next_assignment()
        assert assignment is not None
        assert assignment.stage_id == "pilot_3"
        assert assignment.candidate_id == candidate_id
        _assert_assignment_is_only_queue_authority(scheduler, assignment)
        scheduler.mark_running(
            assignment,
            run_id=f"pilot-run-{index}",
            node_id=f"pilot-node-{candidate_id}",
        )
        store.save(scheduler)
        scheduler = store.load_or_create("coco")
        recovered = scheduler.next_assignment()
        assert recovered is not None
        assert recovered.assignment_id == assignment.assignment_id
        assert recovered.status == "running"
        _report_assignment(scheduler, recovered, delta=pilot_deltas[candidate_id])
        with pytest.raises(RuntimeError, match="already consumed"):
            scheduler.mark_running(recovered)
        if index < 3:
            pending = scheduler.next_assignment()
            assert pending is not None and pending.stage_id == "pilot_3"

    pilot_10 = scheduler.next_assignment()
    assert pilot_10 is not None
    assert pilot_10.candidate_id == "b"
    assert pilot_10.stage_id == "pilot_10"
    _assert_assignment_is_only_queue_authority(scheduler, pilot_10)
    scheduler.mark_running(pilot_10, run_id="pilot-10-run", node_id="pilot-10-node")
    _report_assignment(scheduler, pilot_10, delta=0.05, improved=1)

    assert scheduler.next_assignment(confirm_full_run=False) is None
    seed_1 = scheduler.next_assignment(confirm_full_run=True)
    assert seed_1 is not None and seed_1.stage_id == "candidate_full_seed_1"
    _assert_assignment_is_only_queue_authority(scheduler, seed_1)
    scheduler.mark_running(seed_1, run_id="full-seed-1", node_id="full-node-1")
    _report_assignment(scheduler, seed_1, delta=0.03, improved=1)

    for seed_index in (2, 3):
        confirmation = scheduler.next_assignment(confirm_full_run=True)
        assert confirmation is not None
        assert confirmation.stage_id == "candidate_full_confirmation"
        assert confirmation.seed_index == seed_index
        _assert_assignment_is_only_queue_authority(scheduler, confirmation)
        scheduler.mark_running(
            confirmation,
            run_id=f"full-seed-{seed_index}",
            node_id=f"full-node-{seed_index}",
        )
        _report_assignment(scheduler, confirmation, delta=0.03, improved=1)

    assert scheduler.study.trial("b").status == "confirmed"
    assignment_ids = [item.assignment_id for item in scheduler.study.assignments]
    assert len(assignment_ids) == 7
    assert len(set(assignment_ids)) == 7
    assert {item.status for item in scheduler.study.assignments} == {"completed"}
    assert scheduler.next_assignment(confirm_full_run=True) is None
