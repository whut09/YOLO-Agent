"""Paired-seed statistics for the three-seed confirmation (§18K §3).

For every metric the confirmation runs produced, pair the baseline and
candidate values **by their matched seed** and report:

* per-seed paired deltas,
* ``mean_delta`` and sample ``std_delta``,
* a two-sided 95% Student-t confidence interval,

The primary metric (``map50_95``) is mandatory; precision, recall,
AP_small and latency are recorded when all six runs carry them, and any
metric a side failed to record is listed in ``missing_metrics`` rather
than silently averaged away.
"""

from __future__ import annotations

from math import sqrt
from statistics import stdev

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.agents.seed_confirmation import (
    PRIMARY_METRIC,
    RECORDED_METRICS,
    SeedConfirmationState,
)

CONFIRMATION_STATS_SCHEMA_VERSION = "confirmation_statistics.v1"

#: Two-sided 95% Student-t critical values by degrees of freedom.
_T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def paired_confidence_interval(deltas: list[float]) -> tuple[float, float] | None:
    """Two-sided 95% Student-t CI over paired per-seed deltas."""
    if len(deltas) < 2:
        return None
    mean = sum(deltas) / len(deltas)
    critical = _T_975.get(len(deltas) - 1, 1.96)
    margin = critical * stdev(deltas) / sqrt(len(deltas))
    return mean - margin, mean + margin


class MetricStatistics(BaseModel):
    """One metric's paired-seed movement."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    baseline_values: list[float] = Field(default_factory=list)
    candidate_values: list[float] = Field(default_factory=list)
    #: candidate - baseline, aligned with the matched seed order.
    paired_deltas: list[float] = Field(default_factory=list)
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    mean_delta: float | None = None
    std_delta: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None


class ConfirmationStatistics(BaseModel):
    """All paired statistics for one completed confirmation matrix."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CONFIRMATION_STATS_SCHEMA_VERSION
    confirmation_id: str
    seeds: list[int] = Field(default_factory=list)
    primary: MetricStatistics
    secondary: dict[str, MetricStatistics] = Field(default_factory=dict)
    missing_metrics: list[str] = Field(default_factory=list)


def _stats_for(metric: str, baseline: list[float], candidate: list[float]) -> MetricStatistics:
    deltas = [c - b for b, c in zip(baseline, candidate, strict=True)]
    interval = paired_confidence_interval(deltas)
    mean = sum(deltas) / len(deltas)
    return MetricStatistics(
        metric=metric,
        baseline_values=baseline,
        candidate_values=candidate,
        paired_deltas=[round(d, 9) for d in deltas],
        baseline_mean=round(sum(baseline) / len(baseline), 9),
        candidate_mean=round(sum(candidate) / len(candidate), 9),
        mean_delta=round(mean, 9),
        std_delta=round(stdev(deltas), 9) if len(deltas) >= 2 else None,
        ci95_low=round(interval[0], 9) if interval else None,
        ci95_high=round(interval[1], 9) if interval else None,
    )


def compute_confirmation_statistics(
    state: SeedConfirmationState,
) -> ConfirmationStatistics:
    """Paired statistics across the completed six-run matrix.

    Fail-closed: an incomplete matrix or an unrecorded primary metric is a
    ValueError, never a silent partial average.
    """
    if not state.is_matrix_complete:
        incomplete = [run.key for run in state.runs if not run.is_completed]
        raise ValueError(
            "confirmation statistics require a completed matrix; "
            f"incomplete runs: {', '.join(incomplete)}"
        )

    seeds = state.baseline_seeds

    def _values(metric: str, role: str) -> list[float] | None:
        values: list[float] = []
        for seed in seeds:
            run = state.run_for(role, seed)  # type: ignore[arg-type]
            if metric not in run.metrics:
                return None
            values.append(float(run.metrics[metric]))
        return values

    primary_baseline = _values(PRIMARY_METRIC, "baseline")
    primary_candidate = _values(PRIMARY_METRIC, "candidate")
    if primary_baseline is None or primary_candidate is None:
        raise ValueError(
            f"every confirmation run must record the primary metric '{PRIMARY_METRIC}'"
        )

    secondary: dict[str, MetricStatistics] = {}
    missing: list[str] = []
    for metric in RECORDED_METRICS:
        baseline = _values(metric, "baseline")
        candidate = _values(metric, "candidate")
        if baseline is None or candidate is None:
            missing.append(metric)
            continue
        secondary[metric] = _stats_for(metric, baseline, candidate)

    return ConfirmationStatistics(
        confirmation_id=state.confirmation_id,
        seeds=list(seeds),
        primary=_stats_for(PRIMARY_METRIC, primary_baseline, primary_candidate),
        secondary=secondary,
        missing_metrics=missing,
    )


__all__ = [
    "CONFIRMATION_STATS_SCHEMA_VERSION",
    "ConfirmationStatistics",
    "MetricStatistics",
    "compute_confirmation_statistics",
    "paired_confidence_interval",
]
