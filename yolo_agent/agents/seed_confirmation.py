"""End-to-end three-seed confirmation workflow (Prompt-18K §1/§2/§6).

The promotion pipeline this module models:

``pilot winner`` → ``full-run confirmation approval`` →
``baseline seed 1/2/3`` + ``candidate seed 1/2/3`` →
``paired statistics`` → ``confidence interval`` → ``confirmed / rejected``

Two rules are enforced structurally rather than by convention:

* **Matched seeds only** — the baseline and candidate seed lists must be
  identical, position by position.  ``baseline 0,1,2`` against
  ``candidate 3,4,5`` is not a paired comparison and cannot even be
  constructed.
* **Per-run state** — every one of the six runs carries its own
  ``pending / running / completed / failed`` status so an interrupted
  confirmation can resume without re-running completed seeds.

Approval (§5) lives on the state as ``approved``; execution helpers refuse
to schedule work until the user has granted it explicitly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin

CONFIRMATION_SCHEMA_VERSION = "seed_confirmation.v1"

ConfirmationRole = Literal["baseline", "candidate"]
RunStatus = Literal["pending", "running", "completed", "failed"]

#: The primary optimization objective for the promotion rule.
PRIMARY_METRIC = "map50_95"

#: Secondary metrics §3 requires to be carried through the report when the
#: runs produce them.
RECORDED_METRICS: tuple[str, ...] = (
    "precision",
    "recall",
    "ap_small",
    "latency_ms",
)

#: Default seeds match the ASHA study's confirmation seeds so a workflow
#: spawned from a scheduler uses the same seed set.
DEFAULT_CONFIRMATION_SEEDS: tuple[int, ...] = (42, 43, 44)


class ConfirmationRun(BaseModel):
    """One of the six seeds in the confirmation matrix."""

    model_config = ConfigDict(extra="forbid")

    role: ConfirmationRole
    seed: int
    run_id: str = ""
    status: RunStatus = "pending"
    #: Evaluated metrics; empty until the run completes.
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def key(self) -> str:
        return f"{self.role}:seed{self.seed}"


class SeedConfirmationState(BaseModel, YAMLModelMixin):
    """The full six-run confirmation state for one pilot winner."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = CONFIRMATION_SCHEMA_VERSION
    confirmation_id: str
    pilot_winner_id: str
    baseline_seeds: list[int] = Field(default_factory=lambda: list(DEFAULT_CONFIRMATION_SEEDS))
    candidate_seeds: list[int] = Field(default_factory=lambda: list(DEFAULT_CONFIRMATION_SEEDS))

    #: §5: nothing may execute before the user grants this explicitly.
    approved: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None

    #: §4 promotion inputs.
    target_delta: float = 0.0
    #: Required lower bound of the 95% CI on the primary metric.
    ci_lower_bound: float = 0.0
    #: Hard runtime constraint: candidate latency may not exceed the
    #: baseline by more than this many milliseconds.
    max_latency_regression_ms: float | None = None

    runs: list[ConfirmationRun] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_pairing_and_matrix(self) -> "SeedConfirmationState":
        if self.baseline_seeds != self.candidate_seeds:
            raise ValueError(
                "seed pairing must be matched: baseline seeds "
                f"{self.baseline_seeds} != candidate seeds {self.candidate_seeds}"
            )
        if len(self.baseline_seeds) < 3:
            raise ValueError(
                "a three-seed confirmation requires at least 3 matched seeds"
            )
        expected = {
            (role, seed)
            for role in ("baseline", "candidate")
            for seed in self.baseline_seeds
        }
        actual = {(run.role, run.seed) for run in self.runs}
        if actual and actual != expected:
            raise ValueError(
                "confirmation runs must be exactly the matched "
                f"role x seed matrix; missing={sorted(expected - actual)} "
                f"extra={sorted(actual - expected)}"
            )
        return self

    @classmethod
    def create(
        cls,
        *,
        confirmation_id: str,
        pilot_winner_id: str,
        seeds: list[int] | None = None,
        target_delta: float = 0.0,
        ci_lower_bound: float = 0.0,
        max_latency_regression_ms: float | None = None,
    ) -> "SeedConfirmationState":
        """Build the six-run pending matrix for a pilot winner."""
        seed_list = list(seeds or DEFAULT_CONFIRMATION_SEEDS)
        runs = [
            ConfirmationRun(role=role, seed=seed)
            for role in ("baseline", "candidate")
            for seed in seed_list
        ]
        return cls(
            confirmation_id=confirmation_id,
            pilot_winner_id=pilot_winner_id,
            baseline_seeds=seed_list,
            candidate_seeds=seed_list,
            target_delta=target_delta,
            ci_lower_bound=ci_lower_bound,
            max_latency_regression_ms=max_latency_regression_ms,
            runs=runs,
        )

    def run_for(self, role: ConfirmationRole, seed: int) -> ConfirmationRun:
        for run in self.runs:
            if run.role == role and run.seed == seed:
                return run
        raise KeyError(f"no confirmation run for {role}:seed{seed}")

    # --- legal run-state transitions (§6) --------------------------------
    #
    # pending -> running -> completed | failed, with failed -> running retry.
    # ``completed`` is terminal: a finished seed is never re-run, so a loaded
    # state with status ``running`` only ever comes from an interrupted
    # process and may be re-claimed.

    def mark_running(self, role: ConfirmationRole, seed: int) -> ConfirmationRun:
        run = self.run_for(role, seed)
        if run.is_completed:
            raise ValueError(
                f"completed run {run.key} is terminal and must not re-run"
            )
        run.status = "running"
        run.error = None
        run.started_at = datetime.now(timezone.utc)
        run.finished_at = None
        self.updated_at = datetime.now(timezone.utc)
        return run

    def mark_completed(
        self, role: ConfirmationRole, seed: int, metrics: dict[str, float]
    ) -> ConfirmationRun:
        run = self.run_for(role, seed)
        if run.status != "running":
            raise ValueError(
                f"only a running mark can complete, {run.key} is '{run.status}'"
            )
        if PRIMARY_METRIC not in metrics:
            raise ValueError(
                f"run {run.key} cannot complete without the primary metric "
                f"'{PRIMARY_METRIC}'"
            )
        run.status = "completed"
        run.metrics = dict(metrics)
        run.finished_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        return run

    def mark_failed(
        self, role: ConfirmationRole, seed: int, error: str
    ) -> ConfirmationRun:
        run = self.run_for(role, seed)
        if run.is_completed:
            raise ValueError(
                f"completed run {run.key} is terminal and must not be failed"
            )
        run.status = "failed"
        run.error = error
        run.finished_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        return run

    @property
    def completed_runs(self) -> list[ConfirmationRun]:
        return [run for run in self.runs if run.is_completed]

    @property
    def is_matrix_complete(self) -> bool:
        return all(run.is_completed for run in self.runs)


__all__ = [
    "CONFIRMATION_SCHEMA_VERSION",
    "DEFAULT_CONFIRMATION_SEEDS",
    "PRIMARY_METRIC",
    "RECORDED_METRICS",
    "ConfirmationRole",
    "ConfirmationRun",
    "RunStatus",
    "SeedConfirmationState",
]
