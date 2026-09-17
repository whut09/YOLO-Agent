"""Round decision engine over full error-delta profiles.

Deterministic core of the bounded autonomous loop: each round consumes the
complete :class:`ErrorDeltaProfile` (never a lone mAP scalar), scores the
candidate on the TaskSpec-weighted Pareto surface, and emits exactly one
next action — promote, reject, refine, rollback, collect evidence/data,
request annotation, or one of the three stops.  Stopping never invents a
new action: when no evidence-supported action remains the loop reports
``stop_exhausted`` instead of randomly mutating parameters.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.error_delta_profile import (
    ErrorDeltaProfile,
)
from yolo_agent.core.task_spec import TaskSpec


LoopDecisionType = Literal[
    "promote",
    "reject",
    "refine",
    "rollback",
    "collect_evidence",
    "collect_data",
    "request_annotation",
    "stop_exhausted",
    "stop_goal_met",
    "stop_budget",
]

LoopStopReason = Literal[
    "no_evidence_supported_action",
    "objective_confirmed",
    "budget_exhausted",
]


class ParetoWeights(BaseModel):
    """TaskSpec-derived objective weights and hard constraints."""

    accuracy: float = 1.0
    precision: float = 0.5
    recall: float = 0.5
    ap_small: float = 0.5
    latency: float = 0.5
    params: float = 0.25
    flops: float = 0.25
    memory: float = 0.25
    hard_constraints: list[str] = Field(default_factory=list)


def pareto_weights_from_task_spec(task_spec: TaskSpec | None) -> ParetoWeights:
    """Derive Pareto weights and hard constraints from the TaskSpec."""

    if task_spec is None:
        return ParetoWeights()
    primary = task_spec.primary_metric
    weights = ParetoWeights()
    boosts = {
        "map50": "accuracy",
        "map50_95": "accuracy",
        "ap_small": "ap_small",
        "ap_medium": "accuracy",
        "ap_large": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "latency_ms": "latency",
    }
    name = boosts.get(primary.name)
    if name is not None:
        setattr(weights, name, max(getattr(weights, name), primary.weight))
    for secondary in task_spec.secondary_metrics:
        sec_name = boosts.get(secondary.name)
        if sec_name is not None:
            setattr(weights, sec_name, max(getattr(weights, sec_name), secondary.weight))
    if task_spec.max_latency_ms is not None:
        weights.hard_constraints.append("max_latency_ms")
    if task_spec.max_model_size_mb is not None:
        weights.hard_constraints.append("max_model_size_mb")
    return weights


_PARETO_METRIC_SOURCES: dict[str, tuple[str, ...]] = {
    "accuracy": ("map50_95", "mAP50-95", "map50"),
    "precision": ("precision",),
    "recall": ("recall",),
    "ap_small": ("ap_small",),
    "latency": ("latency_ms",),
    "params": ("params_m",),
    "flops": ("flops_g",),
    "memory": ("memory_mb",),
}


class ParetoScore(BaseModel):
    """Weighted scalarization of the full delta profile for ranking only."""

    weighted_improvement: float
    per_axis: dict[str, float] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    dominated: bool = False


def pareto_score(
    profile: ErrorDeltaProfile,
    weights: ParetoWeights | None = None,
) -> ParetoScore:
    """Score the full profile on all Pareto axes — accuracy, precision,
    recall, AP_small, latency, params, FLOPs, memory.

    Axes are *relative* improvements (delta / baseline magnitude) so
    heterogeneous units cannot swamp each other: an 18% latency regression
    and a +6-point AP_small gain become comparable fractions, never
    ``-1.8 ms`` vs ``+0.06 mAP``.
    """

    weights = weights or ParetoWeights()
    per_axis: dict[str, float] = {}
    total = 0.0
    for axis, metric_names in _PARETO_METRIC_SOURCES.items():
        delta = next((profile.delta_for(n) for n in metric_names if profile.delta_for(n) is not None), None)
        if delta is None:
            continue
        if delta.baseline_value not in (0.0,):
            axis_value = delta.improvement / abs(delta.baseline_value)
        else:
            axis_value = delta.improvement
        per_axis[axis] = axis_value
        weight = getattr(weights, axis)
        total += weight * axis_value
    violations = [
        f"{item.constraint}:{item.metric_name}={item.observed:.4g}>{item.limit:.4g}"
        for item in profile.hard_constraint_violations
    ]
    return ParetoScore(
        weighted_improvement=total,
        per_axis=per_axis,
        violations=violations,
        dominated=bool(violations),
    )


class NextRoundDecision(BaseModel):
    """The deterministic decision for one finished round."""

    decision: LoopDecisionType
    round_index: int
    pareto: ParetoScore
    reasons: list[str] = Field(default_factory=list)
    primary_problem_tag: str | None = None
    missing_evidence: list[str] = Field(default_factory=list)
    refine_action_id: str | None = None
    rollback_candidate_id: str | None = None
    annotation_targets: list[str] = Field(default_factory=list)
    stop_reason: LoopStopReason | None = None

    @property
    def continues(self) -> bool:
        return self.decision in {"promote", "refine", "collect_evidence", "collect_data", "request_annotation"}


class DecisionEngineConfig(BaseModel):
    """Thresholds for the deterministic decision rules."""

    promote_min_improvement: float = Field(default=0.0, ge=0.0)
    evidence_complete_required_for_promote: bool = True
    min_supported_actions_for_continue: int = Field(default=1, ge=0)


def decide_next_round(
    profile: ErrorDeltaProfile,
    *,
    round_index: int,
    supported_action_ids: list[str] | None = None,
    task_spec: TaskSpec | None = None,
    budget_remaining: bool = True,
    goal_met: bool = False,
    config: DecisionEngineConfig | None = None,
) -> NextRoundDecision:
    """Map one round's full delta profile to the next loop action.

    Decision order (deterministic): hard-constraint violation → rollback or
    reject; evidence incomplete → collect_evidence; goal met → stop; no
    supported actions → stop_exhausted; budget gone → stop_budget;
    improvement → promote; regression on primary → refine.
    """

    config = config or DecisionEngineConfig()
    weights = pareto_weights_from_task_spec(task_spec)
    score = pareto_score(profile, weights)
    supported = list(supported_action_ids or [])
    reasons: list[str] = []

    if profile.hard_constraint_violations:
        reasons.extend(f"hard_constraint_violated:{item}" for item in score.violations)
        rollback = profile.candidate_id or None
        return NextRoundDecision(
            decision="rollback",
            round_index=round_index,
            pareto=score,
            reasons=reasons,
            rollback_candidate_id=rollback,
        )

    if profile.evidence_incomplete and config.evidence_complete_required_for_promote:
        missing = sorted({item.metric_name for item in profile.evidence_incomplete})
        reasons.append("evidence_incomplete:" + ",".join(missing))
        return NextRoundDecision(
            decision="collect_evidence",
            round_index=round_index,
            pareto=score,
            reasons=reasons,
            missing_evidence=missing,
        )

    if goal_met:
        reasons.append("objective_confirmed_by_delta_profile")
        return NextRoundDecision(
            decision="stop_goal_met",
            round_index=round_index,
            pareto=score,
            reasons=reasons,
            stop_reason="objective_confirmed",
        )

    primary = profile.primary_map_delta()
    progressed = score.weighted_improvement > 0 or (primary is not None and primary.improvement > 0)
    if not supported and not progressed:
        reasons.append("no_evidence_supported_action_remaining")
        return NextRoundDecision(
            decision="stop_exhausted",
            round_index=round_index,
            pareto=score,
            reasons=reasons,
            stop_reason="no_evidence_supported_action",
        )

    if not budget_remaining:
        reasons.append("compute_budget_exhausted")
        return NextRoundDecision(
            decision="stop_budget",
            round_index=round_index,
            pareto=score,
            reasons=reasons,
            stop_reason="budget_exhausted",
        )

    if score.weighted_improvement > config.promote_min_improvement and not score.dominated:
        tag = _primary_problem_tag(profile)
        reasons.append(
            f"pareto_improvement={score.weighted_improvement:.4f} axes={sorted(score.per_axis)}"
        )
        return NextRoundDecision(
            decision="promote",
            round_index=round_index,
            pareto=score,
            reasons=reasons,
            primary_problem_tag=tag,
        )

    if primary is not None and primary.improvement < 0:
        reasons.append(f"primary_metric_regressed:{primary.metric_name}={primary.improvement:.4f}")
        refine = supported[0] if supported else None
        return NextRoundDecision(
            decision="refine",
            round_index=round_index,
            pareto=score,
            reasons=reasons,
            primary_problem_tag=_primary_problem_tag(profile),
            refine_action_id=refine,
        )

    reasons.append("no_meaningful_delta;continuing_search_with_supported_actions")
    return NextRoundDecision(
        decision="refine",
        round_index=round_index,
        pareto=score,
        reasons=reasons,
        primary_problem_tag=_primary_problem_tag(profile),
        refine_action_id=supported[0] if supported else None,
    )

def _primary_problem_tag(profile: ErrorDeltaProfile) -> str | None:
    """Pick the dominant remaining problem tag from the worst regression."""

    regressions = profile.regressions()
    if not regressions:
        return None
    worst = min(regressions, key=lambda item: item.improvement)
    if worst.scope == "scale" and "small" in (worst.subject or worst.metric_name):
        return "small_object_fn"
    if worst.metric_name == "precision":
        return "background_fp"
    if worst.metric_name == "recall":
        return "small_object_fn"
    return None


__all__ = [
    "DecisionEngineConfig",
    "LoopDecisionType",
    "LoopStopReason",
    "NextRoundDecision",
    "ParetoScore",
    "ParetoWeights",
    "decide_next_round",
    "pareto_score",
    "pareto_weights_from_task_spec",
]
