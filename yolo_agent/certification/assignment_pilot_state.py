"""Persistent state for assignment shadow-to-active promotion."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.yaml_io import YAMLModelMixin


AssignmentPilotStateName = Literal[
    "shadow_planned",
    "shadow_evidence_complete",
    "active_candidate_eligible",
    "active_pilot",
    "promoted",
    "rejected",
]

_STATE_ORDER: dict[AssignmentPilotStateName, int] = {
    "shadow_planned": 0,
    "shadow_evidence_complete": 1,
    "active_candidate_eligible": 2,
    "active_pilot": 3,
    "promoted": 4,
    "rejected": 4,
}


class AssignmentPilotState(BaseModel):
    """One canonical assignment component's promotion state."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    trial_id: str
    candidate_id: str
    canonical_component_id: str
    shadow_recipe_id: str
    active_recipe_id: str | None = None
    protocol_hash: str
    matched_control_node_id: str | None = None
    matched_control_protocol_hash: str | None = None
    shadow_evidence_path: Path | None = None
    state: AssignmentPilotStateName = "shadow_planned"
    disposition: Literal[
        "queued",
        "blocked_runtime",
        "evidence_recovery",
        "already_tested",
    ] = "queued"
    reason_codes: list[str] = Field(default_factory=list)
    shadow_metrics: dict[str, float] = Field(default_factory=dict)
    active_trial_id: str | None = None
    active_candidate_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def transition(
        self,
        next_state: AssignmentPilotStateName,
        *,
        reason_codes: list[str] | None = None,
        disposition: str | None = None,
        **updates: object,
    ) -> "AssignmentPilotState":
        """Advance one state, allowing idempotent recovery of the same state."""
        current_rank = _STATE_ORDER[self.state]
        next_rank = _STATE_ORDER[next_state]
        if next_rank < current_rank:
            raise ValueError(f"assignment state cannot move backward: {self.state} -> {next_state}")
        if next_rank == current_rank and next_state != self.state:
            raise ValueError(f"assignment terminal state conflict: {self.state} -> {next_state}")
        self.state = next_state
        if reason_codes is not None:
            self.reason_codes = list(dict.fromkeys(reason_codes))
        if disposition is not None:
            self.disposition = disposition  # type: ignore[assignment]
        for key, value in updates.items():
            setattr(self, key, value)
        self.updated_at = datetime.now(timezone.utc)
        return self


class AssignmentPilotStateLedger(BaseModel, YAMLModelMixin):
    """Replayable state ledger stored beside a run's ASHA artifacts."""

    schema_version: str = "assignment_pilot_state.v1"
    run_id: str
    records: list[AssignmentPilotState] = Field(default_factory=list)

    def record(self, trial_id: str) -> AssignmentPilotState | None:
        return next((item for item in self.records if item.trial_id == trial_id), None)

    def upsert(self, item: AssignmentPilotState) -> AssignmentPilotState:
        existing = self.record(item.trial_id)
        if existing is None:
            self.records.append(item)
            return item
        if existing.canonical_component_id != item.canonical_component_id:
            raise ValueError(f"assignment trial {item.trial_id} changed canonical component")
        existing.model_copy(update=item.model_dump(exclude={"created_at"}), deep=True)
        for key, value in item.model_dump(exclude={"created_at"}).items():
            setattr(existing, key, value)
        return existing

    def transition(
        self,
        trial_id: str,
        next_state: AssignmentPilotStateName,
        *,
        reason_codes: list[str] | None = None,
        disposition: str | None = None,
        **updates: object,
    ) -> AssignmentPilotState:
        item = self.record(trial_id)
        if item is None:
            raise KeyError(f"unknown assignment pilot state: {trial_id}")
        item.transition(
            next_state,
            reason_codes=reason_codes,
            disposition=disposition,
            **updates,
        )
        return item

    @classmethod
    def load_or_create(cls, path: Path | str, *, run_id: str) -> "AssignmentPilotStateLedger":
        target = Path(path)
        if not target.is_file():
            return cls(run_id=run_id)
        ledger = cls.from_yaml(target)
        if ledger.run_id != run_id:
            raise ValueError(f"assignment state ledger belongs to {ledger.run_id}, not {run_id}")
        return ledger

    def save(self, path: Path | str) -> Path:
        return self.to_yaml(path, sort_keys=False)


def assignment_state_path(run_dir: Path | str) -> Path:
    """Return the stable artifact path used by the optimizer."""
    return Path(run_dir) / "artifacts" / "assignment_pilot_state.yaml"
