"""Four-way promotion rule over a completed three-seed confirmation (Prompt-18K §4).

A candidate is CONFIRMED only when *all three* conditions hold:

1. primary metric mean delta clears the target bar,
2. the CI policy is satisfied (paired 95% CI lower bound above zero),
3. no hard constraint regression (latency/size caps from the rule/TaskSpec).

Everything else falls into exactly one bucket:

- REJECTED        — significant primary regression, a material negative point
                    estimate, or a verifiable hard-constraint violation,
- POSSIBLE        — the delta bar is met but the CI policy is not (underpowered
                    or high-variance evidence: worth more seeds, not promotion),
- INCONCLUSIVE    — evidence cannot resolve the call (effect below target, CI
                    straddling zero, or constraint evidence not recorded).

Fail-closed: an incomplete matrix or a missing primary/constraint metric yields
INCONCLUSIVE rather than a verdict built from partial evidence.  The rule never
reads headline mAP alone — CI and hard constraints are first-class inputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.agents.confirmation_statistics import (
    ConfirmationStatistics,
    compute_confirmation_statistics,
)
from yolo_agent.agents.seed_confirmation import SeedConfirmationState
from yolo_agent.core.task_spec import TaskSpec

PromotionVerdict = Literal["CONFIRMED", "POSSIBLE", "REJECTED", "INCONCLUSIVE"]


class PromotionRule(BaseModel):
    """Declarative bar a candidate must clear to be promoted."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: str = "map50_95"
    primary_goal: Literal["maximize", "minimize"] = "maximize"
    #: Required primary delta (effective, goal-oriented units).
    target_delta: float = Field(default=0.02)
    #: CI policy: the paired 95% CI lower bound must stay above zero.
    ci_policy: Literal["ci_low_above_zero"] = "ci_low_above_zero"
    #: Allowed relative regression on the recorded latency metric (0.05 = +5%).
    max_latency_regression: float | None = Field(default=0.05, ge=0.0)
    #: Allowed relative regression on the recorded model size metric.
    max_model_size_regression: float | None = Field(default=0.10, ge=0.0)

    @classmethod
    def from_objective(
        cls, objective: object, **overrides: object
    ) -> PromotionRule:
        """Build the rule from an ``OptimizationObjective`` (duck-typed)."""
        target = getattr(objective, "target_absolute_delta", None)
        values: dict[str, object] = {
            "primary_metric": str(getattr(objective, "primary_metric", "map50_95")),
            "target_delta": 0.02 if target is None else float(target),
            "max_latency_regression": getattr(
                objective, "max_latency_regression", 0.05
            ),
            "max_model_size_regression": getattr(
                objective, "max_model_size_regression", 0.10
            ),
        }
        values.update(overrides)
        return cls.model_validate(values)  # type: ignore[arg-type]


class PromotionDecision(BaseModel):
    """The four-way verdict with full justification."""

    model_config = ConfigDict(extra="forbid")

    verdict: PromotionVerdict
    primary_metric: str
    target_delta: float
    mean_delta: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None
    #: Why this verdict (ordered, human-readable).
    reasons: list[str] = Field(default_factory=list)
    #: Verified hard-constraint violations that forced a rejection.
    hard_constraint_violations: list[str] = Field(default_factory=list)
    #: Real gains worth acknowledging even when not confirmed.
    acknowledged_benefits: list[str] = Field(default_factory=list)
    statistics: ConfirmationStatistics | None = None

    @property
    def is_confirmed(self) -> bool:
        return self.verdict == "CONFIRMED"


def _role_mean(
    state: SeedConfirmationState, role: str, metric: str
) -> float | None:
    """Mean of one metric across one role's completed runs.

    Fail-closed: ``None`` unless *every* run of the role recorded the metric —
    a partial mean would silently mask a missing seed.
    """
    values: list[float] = []
    for run in state.runs:
        if run.role != role:
            continue
        if metric not in run.metrics:
            return None
        values.append(run.metrics[metric])
    if not values:
        return None
    return sum(values) / len(values)


def _relative_regression(baseline: float, candidate: float) -> float | None:
    if baseline <= 0.0:
        return None
    return (candidate - baseline) / baseline


def _check_hard_constraints(
    state: SeedConfirmationState,
    rule: PromotionRule,
    task_spec: TaskSpec | None,
) -> tuple[list[str], list[str]]:
    """Return ``(violations, missing_evidence)`` for the declared caps."""
    violations: list[str] = []
    missing: list[str] = []

    latency_cap = task_spec.max_latency_ms if task_spec is not None else None
    size_cap = task_spec.max_model_size_mb if task_spec is not None else None

    # Latency: relative regression between roles and/or the absolute cap.
    if rule.max_latency_regression is not None or latency_cap is not None:
        base = _role_mean(state, "baseline", "latency_ms")
        cand = _role_mean(state, "candidate", "latency_ms")
        if base is None or cand is None:
            missing.append("latency_ms")
        else:
            if rule.max_latency_regression is not None:
                rel = _relative_regression(base, cand)
                if rel is not None and rel > rule.max_latency_regression:
                    violations.append(
                        "latency_regression:"
                        f"{rel:.4f}>{rule.max_latency_regression:.4f}"
                    )
            if latency_cap is not None and cand > latency_cap:
                violations.append(f"latency_cap:{cand:.3f}>{latency_cap:.3f}")

    # Model size: relative regression and/or the absolute deployment cap.
    if rule.max_model_size_regression is not None or size_cap is not None:
        base = _role_mean(state, "baseline", "model_size_mb")
        cand = _role_mean(state, "candidate", "model_size_mb")
        if base is None or cand is None:
            missing.append("model_size_mb")
        else:
            if rule.max_model_size_regression is not None:
                rel = _relative_regression(base, cand)
                if rel is not None and rel > rule.max_model_size_regression:
                    violations.append(
                        "model_size_regression:"
                        f"{rel:.4f}>{rule.max_model_size_regression:.4f}"
                    )
            if size_cap is not None and cand > size_cap:
                violations.append(f"model_size_cap:{cand:.3f}>{size_cap:.3f}")

    return violations, missing


def _effective(value: float | None, rule: PromotionRule) -> float | None:
    """Flip the sign for minimize-style primaries."""
    if value is None:
        return None
    return value if rule.primary_goal == "maximize" else -value


def apply_promotion_rule(
    state: SeedConfirmationState,
    rule: PromotionRule,
    *,
    task_spec: TaskSpec | None = None,
) -> PromotionDecision:
    """Classify a completed confirmation matrix into the four-way verdict."""
    base = PromotionDecision(
        verdict="INCONCLUSIVE",
        primary_metric=rule.primary_metric,
        target_delta=rule.target_delta,
    )

    try:
        stats = compute_confirmation_statistics(
            state, primary_metric=rule.primary_metric
        )
    except ValueError as exc:
        base.reasons.append(f"evidence_incomplete:{exc}")
        return base

    primary = stats.primary
    base.statistics = stats
    mean = _effective(primary.mean_delta, rule)
    ci_low = _effective(primary.ci95_low, rule)
    ci_high = _effective(primary.ci95_high, rule)
    base.mean_delta = mean
    base.ci95_low = ci_low
    base.ci95_high = ci_high

    if mean is None or ci_low is None or ci_high is None:
        base.reasons.append(
            f"primary_metric_unresolved:{rule.primary_metric}"
        )
        return base

    violations, missing_constraints = _check_hard_constraints(
        state, rule, task_spec
    )
    base.hard_constraint_violations = violations

    # Acknowledge real gains regardless of the verdict (never mAP-only).
    for name, section in sorted(stats.secondary.items()):
        if section.mean_delta is None:
            continue
        if section.mean_delta > 0:
            base.acknowledged_benefits.append(
                f"secondary_improved:{name}:+{section.mean_delta:.4f}"
            )

    # 1. Verified hard-constraint violation rejects outright.
    if violations:
        base.verdict = "REJECTED"
        base.reasons.append(
            "hard_constraint_violation:" + ",".join(violations)
        )
        return base

    # 2. Significant primary regression (CI entirely below zero).
    if ci_high <= 0.0:
        base.verdict = "REJECTED"
        base.reasons.append(
            f"significant_primary_regression:ci95_high={ci_high:.4f}<=0"
        )
        return base

    # 3. Materially worse point estimate.
    if mean <= -rule.target_delta:
        base.verdict = "REJECTED"
        base.reasons.append(
            f"material_primary_regression:{mean:.4f}<=-{rule.target_delta:.4f}"
        )
        return base

    # 4. Constraint evidence absent: cannot claim "no regression" (fail-closed).
    if missing_constraints:
        base.verdict = "INCONCLUSIVE"
        base.reasons.append(
            "constraint_evidence_missing:" + ",".join(missing_constraints)
        )
        return base

    # 5. Delta bar met AND CI policy satisfied -> CONFIRMED.
    if mean >= rule.target_delta and ci_low > 0.0:
        base.verdict = "CONFIRMED"
        base.reasons.append(
            f"delta_bar_met:{mean:.4f}>={rule.target_delta:.4f}"
        )
        base.reasons.append(f"ci_policy_satisfied:ci95_low={ci_low:.4f}>0")
        base.reasons.append("no_hard_constraint_regression")
        return base

    # 6. Delta bar met but CI policy unmet -> POSSIBLE (needs more seeds).
    if mean >= rule.target_delta:
        base.verdict = "POSSIBLE"
        base.reasons.append(
            f"delta_bar_met:{mean:.4f}>={rule.target_delta:.4f}"
        )
        base.reasons.append(f"ci_policy_unmet:ci95_low={ci_low:.4f}<=0")
        return base

    # 7. Effect below the target bar / CI unresolved -> INCONCLUSIVE.
    base.verdict = "INCONCLUSIVE"
    base.reasons.append(
        f"effect_below_target:{mean:.4f}<{rule.target_delta:.4f}"
    )
    base.reasons.append(f"ci_straddles_zero:{ci_low:.4f}..{ci_high:.4f}")
    return base


__all__ = [
    "PromotionDecision",
    "PromotionRule",
    "PromotionVerdict",
    "apply_promotion_rule",
]
