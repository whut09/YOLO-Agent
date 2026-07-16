"""Persistent state and contracts for paper-component reproduction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from yolo_agent.core.yaml_io import YAMLModelMixin


ReproductionStatus = Literal[
    "registered", "license_checked", "adapter_required", "adapter_implemented",
    "unit_tested", "smoke_passed", "debug_passed", "pilot_running",
    "pilot_reproduced", "pilot_rejected", "full_pending_confirmation",
    "full_reproduced", "confirmed_multi_seed",
]


class ReproductionContract(BaseModel):
    """Inputs required and outputs provided by one reproduction state."""

    requires: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)


class ReproductionState(BaseModel, YAMLModelMixin):
    """Resumable state persisted independently from paper metadata."""

    schema_version: str = "reproduction_state.v1"
    paper_id: str
    component_id: str
    status: ReproductionStatus = "registered"
    contracts: dict[str, ReproductionContract] = Field(default_factory=dict)
    satisfied_evidence: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    paper_claims: list[dict[str, Any]] = Field(default_factory=list)
    local_delta: dict[str, Any] = Field(default_factory=dict)
    queued_stage: str | None = None
    queue_id: str | None = None
    attempts: int = 0
    last_error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def has(self, requirement: str) -> bool:
        return requirement in self.satisfied_evidence or requirement in self.evidence

    def refresh_satisfied_evidence(self) -> None:
        self.satisfied_evidence = sorted(set(self.satisfied_evidence) | set(self.evidence))
        self.updated_at = datetime.now(timezone.utc)


__all__ = ["ReproductionContract", "ReproductionState", "ReproductionStatus"]
