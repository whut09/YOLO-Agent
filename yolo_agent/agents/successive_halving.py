"""Successive-halving budget ladder for guarded candidates."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.budget_optimizer import BudgetArm


HalvingDecision = Literal["run", "promote", "eliminate"]


class HalvingStage(BaseModel):
    """One stage in the pilot-to-full budget ladder."""

    stage_id: str
    training_profile: str
    epochs: int = Field(ge=1)
    fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    keep_top_k: int | None = Field(default=None, ge=1)
    keep_ratio: float | None = Field(default=None, gt=0.0, le=1.0)


class HalvingCandidate(BaseModel):
    """Candidate signal used by successive halving."""

    candidate_id: str
    node_id: str
    score: float
    risk: str = "medium"
    policy_id: str | None = None

    @classmethod
    def from_budget_arm(cls, arm: BudgetArm, score: float | None = None) -> "HalvingCandidate":
        """Build a halving candidate from a guarded bandit arm."""
        return cls(
            candidate_id=arm.candidate_id,
            node_id=arm.node_id,
            score=arm.utility if score is None else score,
            risk=arm.risk,
            policy_id=arm.policy_id,
        )


class HalvingAssignment(BaseModel):
    """Budget assignment for one candidate at one stage."""

    stage_id: str
    candidate_id: str
    node_id: str
    training_profile: str
    epochs: int
    fraction: float
    decision: HalvingDecision
    reason: str
    rank: int


class SuccessiveHalvingPlan(BaseModel):
    """Deterministic pilot/full budget ladder."""

    stages: list[HalvingStage]
    assignments: list[HalvingAssignment]
    promoted_to_full: list[str] = Field(default_factory=list)
    eliminated: list[str] = Field(default_factory=list)
    guardrail: str = "successive_halving_consumes_only_guarded_candidates"

    def assignments_for_stage(self, stage_id: str) -> list[HalvingAssignment]:
        """Return assignments for one stage."""
        return [assignment for assignment in self.assignments if assignment.stage_id == stage_id]


class SuccessiveHalvingPlanner:
    """Allocate pilot budgets and narrow candidates before full runs."""

    def __init__(self, stages: list[HalvingStage] | None = None) -> None:
        self.stages = stages or default_halving_stages()

    def plan(self, candidates: list[HalvingCandidate] | list[BudgetArm]) -> SuccessiveHalvingPlan:
        """Build a static successive-halving ladder from current candidate scores."""
        normalized = [_normalize_candidate(candidate) for candidate in candidates]
        active = sorted(normalized, key=lambda candidate: candidate.score, reverse=True)
        assignments: list[HalvingAssignment] = []
        eliminated: list[str] = []

        for stage_index, stage in enumerate(self.stages):
            ranked = sorted(active, key=lambda candidate: candidate.score, reverse=True)
            keep_count = _keep_count(stage, len(ranked))
            keep_ids = {candidate.candidate_id for candidate in ranked[:keep_count]}
            for rank, candidate in enumerate(ranked, start=1):
                kept = candidate.candidate_id in keep_ids
                is_last = stage_index == len(self.stages) - 1
                decision: HalvingDecision = "run" if kept else "eliminate"
                reason = "run_current_halving_stage" if kept else "eliminated_by_successive_halving"
                if kept and is_last:
                    decision = "promote"
                    reason = "promoted_to_full_budget"
                assignments.append(
                    HalvingAssignment(
                        stage_id=stage.stage_id,
                        candidate_id=candidate.candidate_id,
                        node_id=candidate.node_id,
                        training_profile=stage.training_profile,
                        epochs=stage.epochs,
                        fraction=stage.fraction,
                        decision=decision,
                        reason=reason,
                        rank=rank,
                    )
                )
                if not kept:
                    eliminated.append(candidate.candidate_id)
            active = [candidate for candidate in ranked if candidate.candidate_id in keep_ids]

        promoted = [
            assignment.candidate_id
            for assignment in assignments
            if assignment.decision == "promote"
        ]
        return SuccessiveHalvingPlan(
            stages=self.stages,
            assignments=assignments,
            promoted_to_full=list(dict.fromkeys(promoted)),
            eliminated=list(dict.fromkeys(eliminated)),
        )


def default_halving_stages() -> list[HalvingStage]:
    """Return the default COCO candidate budget ladder."""
    return [
        HalvingStage(
            stage_id="pilot_3",
            training_profile="pilot",
            epochs=3,
            fraction=0.1,
            keep_ratio=0.5,
        ),
        HalvingStage(
            stage_id="pilot_10",
            training_profile="pilot",
            epochs=10,
            fraction=0.1,
            keep_top_k=2,
        ),
        HalvingStage(
            stage_id="candidate_full",
            training_profile="candidate_full",
            epochs=100,
            fraction=1.0,
            keep_top_k=1,
        ),
    ]


def _normalize_candidate(candidate: HalvingCandidate | BudgetArm) -> HalvingCandidate:
    if isinstance(candidate, HalvingCandidate):
        return candidate
    return HalvingCandidate.from_budget_arm(candidate)


def _keep_count(stage: HalvingStage, count: int) -> int:
    if count <= 0:
        return 0
    if stage.keep_top_k is not None:
        return min(count, stage.keep_top_k)
    if stage.keep_ratio is not None:
        return max(1, min(count, math.ceil(count * stage.keep_ratio)))
    return count
