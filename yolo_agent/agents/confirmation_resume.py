"""Crash-safe resume planning for the six-run confirmation (Prompt-18K §6).

A persisted ``SeedConfirmationState`` knows exactly which of its six runs are
``pending``, ``running``, ``completed`` or ``failed``.  Resuming must:

- never re-execute a ``completed`` seed (its evidence is already on disk),
- re-execute ``failed`` seeds as retries,
- treat a still-``running`` entry as an interrupted process and re-claim it,
- launch ``pending`` seeds.

``project_confirmation_plan`` mirrors those four run states into an
``ExperimentPlan`` so the ExperimentGraph sees the same truth:
run ``pending`` -> node ``planned``, ``running`` -> ``running``,
``completed`` -> ``completed``, ``failed`` -> ``failed``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.agents.seed_confirmation import SeedConfirmationState
from yolo_agent.core.experiment_graph import (
    ExperimentNode,
    ExperimentPlan,
    ExperimentStatus,
)

#: Run states that must be (re-)executed to finish the matrix.
RESUMABLE_RUN_STATUSES = ("pending", "running", "failed")


class ResumePlan(BaseModel):
    """Which of the six runs a resume must execute, and which it must skip."""

    model_config = ConfigDict(extra="forbid")

    confirmation_id: str
    #: Run keys to execute now: pending + failed (retry) + interrupted running.
    execute: list[str] = Field(default_factory=list)
    #: Completed seeds — evidence kept, never re-run (§6).
    skip_completed: list[str] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    retry_failed: list[str] = Field(default_factory=list)
    #: Status ``running`` loaded from disk means the process died mid-run.
    interrupted_running: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.execute

    @property
    def ready_for_statistics(self) -> bool:
        return self.is_empty and bool(self.skip_completed)


def plan_resume(state: SeedConfirmationState) -> ResumePlan:
    """Plan the remaining work for a persisted confirmation state."""
    execute: list[str] = []
    skip: list[str] = []
    pending: list[str] = []
    retry: list[str] = []
    interrupted: list[str] = []

    for run in state.runs:
        if run.is_completed:
            skip.append(run.key)
            continue
        if run.status == "pending":
            pending.append(run.key)
        elif run.status == "failed":
            retry.append(run.key)
        elif run.status == "running":
            interrupted.append(run.key)
        else:  # pragma: no cover - RunStatus is a closed Literal
            raise ValueError(f"unknown run status '{run.status}' on {run.key}")
        if run.status in RESUMABLE_RUN_STATUSES:
            execute.append(run.key)

    return ResumePlan(
        confirmation_id=state.confirmation_id,
        execute=execute,
        skip_completed=skip,
        pending=pending,
        retry_failed=retry,
        interrupted_running=interrupted,
    )


def node_status_for(run_status: str) -> ExperimentStatus:
    """Map a confirmation run state onto the ExperimentGraph vocabulary."""
    mapping: dict[str, ExperimentStatus] = {
        "pending": "planned",
        "running": "running",
        "completed": "completed",
        "failed": "failed",
    }
    if run_status not in mapping:
        raise ValueError(f"unknown confirmation run status '{run_status}'")
    return mapping[run_status]


def project_confirmation_plan(
    state: SeedConfirmationState, *, data_version: str = ""
) -> ExperimentPlan:
    """Expose all six runs to the ExperimentGraph with their true statuses."""
    nodes: list[ExperimentNode] = []
    for run in state.runs:
        nodes.append(
            ExperimentNode(
                node_id=f"{state.confirmation_id}:{run.key}",
                candidate_config={
                    "candidate_id": f"{state.pilot_winner_id}:{run.key}",
                    "base_model": "confirmation",
                    "scale": "confirmation",
                    "framework": "ultralytics",
                },
                data_version=data_version,
                seed=run.seed,
                status=node_status_for(run.status),
            )
        )
    return ExperimentPlan(
        plan_id=f"confirmation:{state.confirmation_id}",
        nodes=nodes,
        metadata={
            "confirmation_id": state.confirmation_id,
            "pilot_winner_id": state.pilot_winner_id,
            "kinds": {run.key: run.role for run in state.runs},
        },
    )


__all__ = [
    "RESUMABLE_RUN_STATUSES",
    "ResumePlan",
    "node_status_for",
    "plan_resume",
    "project_confirmation_plan",
]
