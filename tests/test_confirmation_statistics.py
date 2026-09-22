"""Paired-seed statistics: mean/std/95% CI across matched seeds (§18K §3)."""

from __future__ import annotations

import pytest

from yolo_agent.agents.confirmation_statistics import (
    compute_confirmation_statistics,
    paired_confidence_interval,
)
from yolo_agent.agents.seed_confirmation import SeedConfirmationState


def _completed_state(
    baseline: dict[str, list[float]],
    candidate: dict[str, list[float]],
    *,
    skip_primary: set[str] | None = None,
) -> SeedConfirmationState:
    """Build a completed six-run state from per-metric per-seed lists."""
    state = SeedConfirmationState.create(
        confirmation_id="conf-stats", pilot_winner_id="trial-abc"
    )
    metric_order = ["map50_95", "precision", "recall", "ap_small", "latency_ms"]
    for role, table in (("baseline", baseline), ("candidate", candidate)):
        for index, seed in enumerate(state.baseline_seeds):
            run = state.run_for(role, seed)  # type: ignore[arg-type]
            run.status = "completed"
            for metric in metric_order:
                if metric in table and metric not in (skip_primary or set()):
                    run.metrics[metric] = table[metric][index]
    return state


def test_consistent_improvement_yields_exact_paired_statistics() -> None:
    state = _completed_state(
        baseline={
            "map50_95": [0.30, 0.30, 0.30],
            "precision": [0.70, 0.70, 0.70],
            "recall": [0.60, 0.60, 0.60],
            "ap_small": [0.20, 0.20, 0.20],
            "latency_ms": [10.0, 10.0, 10.0],
        },
        candidate={
            "map50_95": [0.34, 0.35, 0.33],
            "precision": [0.72, 0.72, 0.72],
            "recall": [0.63, 0.63, 0.63],
            "ap_small": [0.24, 0.24, 0.24],
            "latency_ms": [11.0, 11.0, 11.0],
        },
    )

    stats = compute_confirmation_statistics(state)

    primary = stats.primary
    # Deltas are candidate - baseline, paired by matched seed: +0.04/+0.05/+0.03
    assert primary.paired_deltas == pytest.approx([0.04, 0.05, 0.03])
    assert primary.mean_delta == pytest.approx(0.04)
    assert primary.std_delta == pytest.approx(0.01, abs=1e-9)
    assert primary.ci95_low == pytest.approx(0.015157, abs=1e-4)
    assert primary.ci95_high == pytest.approx(0.064843, abs=1e-4)
    assert primary.baseline_mean == pytest.approx(0.30)
    assert primary.candidate_mean == pytest.approx(0.34)

    # Secondary metrics recorded as required.
    assert set(stats.secondary) == {
        "precision", "recall", "ap_small", "latency_ms",
    }
    assert stats.secondary["latency_ms"].mean_delta == pytest.approx(1.0)
    assert stats.missing_metrics == []


def test_paired_deltas_follow_matched_seed_order() -> None:
    """Pairing is positional over the matched seed lists, never by value."""
    state = _completed_state(
        baseline={"map50_95": [0.30, 0.31, 0.32]},
        candidate={"map50_95": [0.35, 0.32, 0.36]},
    )

    stats = compute_confirmation_statistics(state)

    # seed42: +0.05, seed43: +0.01, seed44: +0.04
    assert stats.primary.paired_deltas == pytest.approx([0.05, 0.01, 0.04])
    assert stats.seeds == state.baseline_seeds


def test_metric_missing_on_one_run_is_recorded_not_averaged() -> None:
    state = _completed_state(
        baseline={
            "map50_95": [0.30, 0.30, 0.30],
            "precision": [0.7, 0.7, 0.7],
            "latency_ms": [10.0, 10.0, 10.0],
        },
        candidate={
            "map50_95": [0.32, 0.32, 0.32],
            "precision": [0.7, 0.7, 0.7],
            "latency_ms": [10.0, 10.0, 10.0],
        },
    )
    # One candidate run failed to record ap_small.
    state.run_for("candidate", 43).metrics.pop("ap_small", None)  # type: ignore[arg-type]

    stats = compute_confirmation_statistics(state)

    assert "ap_small" in stats.missing_metrics
    assert "recall" in stats.missing_metrics
    assert "latency_ms" not in stats.missing_metrics
    assert stats.primary.mean_delta == pytest.approx(0.02)


def test_incomplete_matrix_refuses_statistics() -> None:
    state = SeedConfirmationState.create(
        confirmation_id="conf-partial", pilot_winner_id="trial-abc"
    )
    state.run_for("baseline", 42).status = "completed"  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="completed matrix"):
        compute_confirmation_statistics(state)


def test_missing_primary_metric_refuses_statistics() -> None:
    state = _completed_state(
        baseline={"map50_95": [0.3, 0.3, 0.3]},
        candidate={"map50_95": [0.32, 0.32, 0.32]},
        skip_primary={"map50_95"},
    )

    with pytest.raises(ValueError, match="primary metric 'map50_95'"):
        compute_confirmation_statistics(state)


def test_confidence_interval_needs_at_least_two_deltas() -> None:
    assert paired_confidence_interval([0.1]) is None
    assert paired_confidence_interval([]) is None
    interval = paired_confidence_interval([0.10, 0.12, 0.09])
    assert interval is not None
    low, high = interval
    assert low < 0.11 < high
