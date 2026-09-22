"""Resume planning and graph projection across the six run states (§18K §6)."""

from __future__ import annotations

import pytest

from yolo_agent.agents.confirmation_resume import (
    node_status_for,
    plan_resume,
    project_confirmation_plan,
)
from yolo_agent.agents.seed_confirmation import SeedConfirmationState

PRIMARY = "map50_95"


def _state() -> SeedConfirmationState:
    return SeedConfirmationState.create(
        confirmation_id="conf-resume", pilot_winner_id="trial-abc"
    )


def _complete(run_state: SeedConfirmationState, role: str, seed: int) -> None:
    run_state.mark_running(role, seed)  # type: ignore[arg-type]
    run_state.mark_completed(role, seed, {PRIMARY: 0.3})  # type: ignore[arg-type]


def test_fresh_state_executes_all_six_runs() -> None:
    plan = plan_resume(_state())

    assert len(plan.execute) == 6
    assert plan.skip_completed == []
    assert plan.is_empty is False
    assert plan.ready_for_statistics is False


def test_completed_seed_is_never_re_executed() -> None:
    state = _state()
    _complete(state, "baseline", 42)

    plan = plan_resume(state)

    assert "baseline:seed42" not in plan.execute
    assert "baseline:seed42" in plan.skip_completed
    assert len(plan.execute) == 5


def test_interrupted_running_seed_is_reclaimed() -> None:
    state = _state()
    state.mark_running("candidate", 43)  # type: ignore[arg-type]
    # Simulate a crash: state persisted while the run was still running.

    plan = plan_resume(state)

    assert "candidate:seed43" in plan.interrupted_running
    assert "candidate:seed43" in plan.execute


def test_failed_seed_is_retried_not_dropped() -> None:
    state = _state()
    state.mark_running("baseline", 44)  # type: ignore[arg-type]
    state.mark_failed("baseline", 44, "cuda OOM")  # type: ignore[arg-type]

    plan = plan_resume(state)

    assert "baseline:seed44" in plan.retry_failed
    assert "baseline:seed44" in plan.execute


def test_completed_run_is_terminal() -> None:
    state = _state()
    _complete(state, "baseline", 42)

    with pytest.raises(ValueError, match="terminal"):
        state.mark_running("baseline", 42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="terminal"):
        state.mark_failed("baseline", 42, "nope")  # type: ignore[arg-type]


def test_cannot_complete_without_the_primary_metric() -> None:
    state = _state()
    state.mark_running("baseline", 42)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="primary metric"):
        state.mark_completed("baseline", 42, {"precision": 0.7})  # type: ignore[arg-type]

    # Failed runs may be retried back through running to completion.
    state.mark_failed("baseline", 42, "transient")  # type: ignore[arg-type]
    state.mark_running("baseline", 42)  # type: ignore[arg-type]
    state.mark_completed("baseline", 42, {PRIMARY: 0.31})  # type: ignore[arg-type]
    assert state.run_for("baseline", 42).is_completed  # type: ignore[arg-type]


def test_resume_then_execute_reaches_ready_for_statistics() -> None:
    state = _state()
    _complete(state, "baseline", 42)

    plan = plan_resume(state)
    for key in list(plan.execute):
        role, seed_token = key.split(":")
        seed = int(seed_token.replace("seed", ""))
        state.mark_running(role, seed)  # type: ignore[arg-type]
        state.mark_completed(role, seed, {PRIMARY: 0.32})  # type: ignore[arg-type]

    assert plan_resume(state).ready_for_statistics is True
    assert state.is_matrix_complete is True


def test_graph_projection_exposes_all_four_run_states() -> None:
    state = _state()
    _complete(state, "baseline", 42)
    state.mark_running("baseline", 43)  # type: ignore[arg-type]
    state.mark_running("baseline", 44)  # type: ignore[arg-type]
    state.mark_failed("baseline", 44, "boom")  # type: ignore[arg-type]

    plan = project_confirmation_plan(state, data_version="v1")
    status_by_id = {node.node_id: node.status for node in plan.nodes}

    assert len(plan.nodes) == 6
    assert status_by_id["conf-resume:baseline:seed42"] == "completed"
    assert status_by_id["conf-resume:baseline:seed43"] == "running"
    assert status_by_id["conf-resume:baseline:seed44"] == "failed"
    assert status_by_id["conf-resume:candidate:seed42"] == "planned"
    # Seeds on the nodes mirror the matched confirmation seeds.
    assert {node.seed for node in plan.nodes} == {42, 43, 44}


def test_unknown_run_status_has_no_graph_vocabulary() -> None:
    with pytest.raises(ValueError, match="unknown confirmation run status"):
        node_status_for("exploded")
