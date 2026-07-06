"""Bandit-style budget allocation over guard-approved candidates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from yolo_agent.components.compatibility import RiskLevel

if TYPE_CHECKING:
    from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluation


OptimizerKind = Literal["ucb_bandit", "utility_rank"]


class BudgetOptimizerConfig(BaseModel):
    """Configuration for finite, guarded candidate budget allocation."""

    optimizer_kind: OptimizerKind = "ucb_bandit"
    max_candidates: int = Field(default=6, ge=1)
    exploration_bonus: float = Field(default=0.25, ge=0.0)
    risk_penalty: dict[RiskLevel, float] = Field(
        default_factory=lambda: {"low": 0.0, "medium": 0.15, "high": 0.45}
    )
    require_guard_accepted: bool = True


class BudgetArm(BaseModel):
    """One guarded candidate arm available to the budget optimizer."""

    policy_id: str
    candidate_id: str
    node_id: str
    utility: float = 0.0
    prior_score: float = 0.0
    risk: RiskLevel = "medium"
    pulls: int = 0
    guard_decision: str = "accepted"
    target_actions: list[str] = Field(default_factory=list)
    target_error_facts: list[dict[str, object]] = Field(default_factory=list)


class BudgetArmSelection(BaseModel):
    """Bandit/utility decision for one candidate arm."""

    arm: BudgetArm
    bandit_score: float
    selected: bool
    reason: str
    rank: int | None = None


class BudgetOptimizationReport(BaseModel):
    """Budget allocation report for safe candidates only."""

    optimizer_kind: OptimizerKind
    input_count: int
    guarded_count: int
    selected_count: int
    rejected_by_guard: list[str] = Field(default_factory=list)
    selected: list[BudgetArmSelection] = Field(default_factory=list)
    deferred: list[BudgetArmSelection] = Field(default_factory=list)
    guardrail: str = "bandit_or_bo_only_allocates_budget_after guard/evidence compatibility approval"

    @property
    def selected_arms(self) -> list[BudgetArm]:
        """Return selected candidate arms."""
        return [selection.arm for selection in self.selected]


class BudgetOptimizer:
    """Rank guard-approved candidates using a finite-arm bandit score."""

    def __init__(self, config: BudgetOptimizerConfig | None = None) -> None:
        self.config = config or BudgetOptimizerConfig()

    def optimize(self, evaluations: list["LoopPolicyEvaluation"]) -> BudgetOptimizationReport:
        """Allocate this-round budget only among accepted evaluator candidates."""
        arms: list[BudgetArm] = []
        rejected_by_guard: list[str] = []
        for evaluation in evaluations:
            if evaluation.decision != "accepted" or evaluation.candidate_config is None or evaluation.experiment_node is None:
                rejected_by_guard.append(evaluation.policy_id)
                continue
            arms.append(_arm_from_evaluation(evaluation))

        scored = sorted(
            [
                BudgetArmSelection(
                    arm=arm,
                    bandit_score=_bandit_score(arm, len(arms), self.config),
                    selected=False,
                    reason="eligible_guarded_candidate",
                )
                for arm in arms
            ],
            key=lambda selection: selection.bandit_score,
            reverse=True,
        )
        selected: list[BudgetArmSelection] = []
        deferred: list[BudgetArmSelection] = []
        for rank, selection in enumerate(scored, start=1):
            if len(selected) < self.config.max_candidates:
                selected.append(
                    selection.model_copy(
                        update={
                            "selected": True,
                            "rank": rank,
                            "reason": "selected_by_guarded_bandit_budget",
                        }
                    )
                )
            else:
                deferred.append(
                    selection.model_copy(
                        update={
                            "selected": False,
                            "rank": rank,
                            "reason": "deferred_by_bandit_budget_limit",
                        }
                    )
                )
        return BudgetOptimizationReport(
            optimizer_kind=self.config.optimizer_kind,
            input_count=len(evaluations),
            guarded_count=len(arms),
            selected_count=len(selected),
            rejected_by_guard=rejected_by_guard,
            selected=selected,
            deferred=deferred,
        )


def _arm_from_evaluation(evaluation: Any) -> BudgetArm:
    candidate = evaluation.candidate_config
    node = evaluation.experiment_node
    if candidate is None or node is None:
        raise ValueError("BudgetArm can only be built from accepted evaluations.")
    utility = evaluation.utility_score.utility if evaluation.utility_score is not None else evaluation.priority
    pulls = _pull_count(node.command_spec.metadata if node.command_spec is not None else {})
    return BudgetArm(
        policy_id=evaluation.policy_id,
        candidate_id=candidate.candidate_id,
        node_id=node.node_id,
        utility=utility,
        prior_score=evaluation.priority,
        risk=candidate.risk,
        pulls=pulls,
        guard_decision=evaluation.decision,
        target_actions=_target_actions(candidate.train_overrides),
        target_error_facts=_target_error_facts(candidate.train_overrides),
    )


def _bandit_score(arm: BudgetArm, arm_count: int, config: BudgetOptimizerConfig) -> float:
    risk_penalty = config.risk_penalty.get(arm.risk, 0.0)
    base = arm.utility if config.optimizer_kind == "ucb_bandit" else arm.prior_score
    if config.optimizer_kind == "utility_rank":
        return round(base - risk_penalty, 6)
    exploration = config.exploration_bonus * math.sqrt(math.log(max(2, arm_count + 1)) / (arm.pulls + 1))
    return round(base + exploration - risk_penalty, 6)


def _pull_count(metadata: dict[str, object]) -> int:
    value = metadata.get("bandit_pulls") or metadata.get("training_budget_completed_pulls")
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _target_actions(overrides: dict[str, object]) -> list[str]:
    for key in ("target_actions", "target_error_actions", "action_candidates"):
        value = overrides.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _target_error_facts(overrides: dict[str, object]) -> list[dict[str, object]]:
    value = overrides.get("target_error_facts")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
