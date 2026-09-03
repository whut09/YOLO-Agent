"""Matched-control availability gate for materialized paper candidates."""

from __future__ import annotations

from pydantic import BaseModel, Field

from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.matched_baseline import (
    MatchedControlPlan,
    assess_matched_control_plan,
)


class MatchedControlAssessment(BaseModel):
    available: bool
    matched_control_plan_ready: bool = False
    matched_control_result_ready: bool = False
    plan: MatchedControlPlan | None = None
    protocol_hash: str | None = None
    reasons: list[str] = Field(default_factory=list)


def assess_matched_control(
    candidate: ExperimentNode,
    control: ExperimentNode | None,
    *,
    required_protocol_hash: str,
) -> MatchedControlAssessment:
    assessment = assess_matched_control_plan(
        candidate,
        control,
        required_protocol_hash=required_protocol_hash,
    )
    return MatchedControlAssessment(
        available=assessment.matched_control_plan_ready,
        matched_control_plan_ready=assessment.matched_control_plan_ready,
        plan=assessment.plan,
        protocol_hash=(assessment.plan.protocol_hash if assessment.plan else None),
        reasons=assessment.blockers,
    )


__all__ = ["MatchedControlAssessment", "assess_matched_control"]
