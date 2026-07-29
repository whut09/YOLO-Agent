"""Cooldown and duplicate guards for implementation work queues."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from yolo_agent.agents.paper_adapter_planning.schemas import ImplementationHistoryRecord


class ImplementationDiversityAssessment(BaseModel):
    deferred: bool = False
    reason: str | None = None


def assess_implementation_diversity(
    *,
    fingerprint: str,
    component_family: str,
    current_round: int,
    history: Iterable[ImplementationHistoryRecord],
    cooldown_rounds: int,
) -> ImplementationDiversityAssessment:
    records = list(history)
    if any(item.fingerprint == fingerprint for item in records):
        return ImplementationDiversityAssessment(
            deferred=True,
            reason="duplicate_implementation_fingerprint",
        )
    family_rounds = [
        item.round_index for item in records if item.component_family == component_family
    ]
    if family_rounds:
        latest = max(family_rounds)
        if current_round - latest <= cooldown_rounds:
            return ImplementationDiversityAssessment(
                deferred=True,
                reason=(
                    f"implementation_family_cooldown:{component_family}:"
                    f"last_round={latest}:cooldown={cooldown_rounds}"
                ),
            )
    return ImplementationDiversityAssessment()


__all__ = ["ImplementationDiversityAssessment", "assess_implementation_diversity"]
