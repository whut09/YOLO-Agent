"""Three-seed confirmation models: pairing and matrix invariants (§18K §2/§6)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yolo_agent.agents.seed_confirmation import (
    DEFAULT_CONFIRMATION_SEEDS,
    ConfirmationRun,
    SeedConfirmationState,
)


def test_create_builds_six_pending_runs_without_approval() -> None:
    state = SeedConfirmationState.create(
        confirmation_id="conf-1", pilot_winner_id="trial-abc"
    )

    assert state.approved is False
    assert len(state.runs) == 6
    assert all(run.status == "pending" for run in state.runs)
    assert state.baseline_seeds == list(DEFAULT_CONFIRMATION_SEEDS)
    assert state.candidate_seeds == list(DEFAULT_CONFIRMATION_SEEDS)
    assert state.is_matrix_complete is False


def test_unmatched_seed_sets_are_rejected_at_construction() -> None:
    """baseline 0,1,2 against candidate 3,4,5 is not a paired comparison."""
    with pytest.raises(ValidationError, match="seed pairing must be matched"):
        SeedConfirmationState(
            confirmation_id="conf-bad",
            pilot_winner_id="trial-abc",
            baseline_seeds=[0, 1, 2],
            candidate_seeds=[3, 4, 5],
        )


def test_order_mismatch_is_rejected() -> None:
    """Same seeds in different order still breaks positional pairing."""
    with pytest.raises(ValidationError, match="seed pairing must be matched"):
        SeedConfirmationState(
            confirmation_id="conf-bad-order",
            pilot_winner_id="trial-abc",
            baseline_seeds=[42, 43, 44],
            candidate_seeds=[44, 43, 42],
        )


def test_fewer_than_three_seeds_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 3 matched seeds"):
        SeedConfirmationState(
            confirmation_id="conf-two",
            pilot_winner_id="trial-abc",
            baseline_seeds=[42, 43],
            candidate_seeds=[42, 43],
        )


def test_run_matrix_must_match_the_seed_lists() -> None:
    runs = [
        ConfirmationRun(role=role, seed=seed)
        for role in ("baseline", "candidate")
        for seed in (42, 43, 44)
    ]
    runs.pop()  # drop one candidate seed
    with pytest.raises(ValidationError, match="role x seed matrix"):
        SeedConfirmationState(
            confirmation_id="conf-missing",
            pilot_winner_id="trial-abc",
            baseline_seeds=[42, 43, 44],
            candidate_seeds=[42, 43, 44],
            runs=runs,
        )


def test_run_lookup_and_completion_tracking() -> None:
    state = SeedConfirmationState.create(
        confirmation_id="conf-2", pilot_winner_id="trial-abc"
    )

    run = state.run_for("baseline", 42)
    assert run.status == "pending"
    with pytest.raises(KeyError, match="candidate:seed99"):
        state.run_for("candidate", 99)

    run.status = "running"
    assert run.is_completed is False
    run.status = "completed"
    run.metrics = {"map50_95": 0.41}
    assert run.is_completed is True
    assert state.completed_runs == [run]
    assert state.is_matrix_complete is False


def test_yaml_roundtrip_preserves_pairing_and_status(tmp_path) -> None:
    state = SeedConfirmationState.create(
        confirmation_id="conf-3", pilot_winner_id="trial-abc"
    )
    state.run_for("baseline", 42).status = "completed"
    path = tmp_path / "state.yaml"
    state.to_yaml(path)

    loaded = SeedConfirmationState.from_yaml(path)

    assert loaded.baseline_seeds == loaded.candidate_seeds
    assert loaded.run_for("baseline", 42).status == "completed"
    assert loaded.approved is False
