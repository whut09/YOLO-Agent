"""Cross-round ASHA scheduler tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.asha_scheduler import ASHAObservation, ASHAScheduler, ASHAStudyStore
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode


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
    scheduler.register_trial(
        trial_id=candidate_id,
        candidate_id=candidate_id,
        source_run_id=f"run-{candidate_id}",
        source_node=_node(candidate_id),
        target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
    )


def _report(scheduler: ASHAScheduler, candidate_id: str, stage: str, delta: float, improved: int = 0) -> None:
    scheduler.report(
        candidate_id,
        ASHAObservation(
            stage_id=stage,  # type: ignore[arg-type]
            node_id=f"node_{candidate_id}__{stage}",
            seed=42,
            paired_delta=delta,
            target_error_improved_count=improved,
        ),
    )


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
            ),
        )

    assert scheduler.study.trial(seed_1.trial_id).status == "confirmed"


def test_asha_state_round_trips_between_auto_rounds(tmp_path: Path) -> None:
    store = ASHAStudyStore(tmp_path / "asha_state.yaml")
    scheduler = store.load_or_create("coco")
    _register(scheduler, "a")
    _report(scheduler, "a", "pilot_3", 0.01)
    store.save(scheduler)

    restored = store.load_or_create("coco")

    assert restored.study.trial("a").observation("pilot_3") is not None
    assert restored.study.confirmation_seeds == [42, 43, 44]
