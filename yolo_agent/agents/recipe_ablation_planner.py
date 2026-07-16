"""Internal ablation matrix generation for coupled component recipes."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.ablation_planner import AblationNode, AblationPlan
from yolo_agent.agents.budget_optimizer import BudgetOptimizationReport, BudgetOptimizer, BudgetOptimizerConfig
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluation
from yolo_agent.agents.successive_halving import HalvingCandidate, SuccessiveHalvingPlan, SuccessiveHalvingPlanner
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.recipes.schemas import CoupledRecipe


MatrixRole = Literal["baseline", "single", "pair", "full"]
ContributionConfidence = Literal["possible", "confirmed"]


class RecipeAblationNode(BaseModel):
    node_id: str
    candidate_config: CandidateConfig
    component_ids: list[str]
    role: MatrixRole
    parent_id: str
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    priority: float = 0.0


class AblationObservation(BaseModel):
    node_id: str
    seed: int
    deltas: dict[str, float] = Field(default_factory=dict)


class ContributionAssessment(BaseModel):
    node_id: str
    component_ids: list[str]
    seed_count: int
    confidence: ContributionConfidence
    mean_deltas: dict[str, float] = Field(default_factory=dict)
    reason: str


class RecipeAblationPlan(BaseModel):
    recipe_id: str
    baseline_id: str
    nodes: list[RecipeAblationNode]
    omitted_combinations: list[list[str]] = Field(default_factory=list)
    budget_max_nodes: int
    single_variable_plan: AblationPlan
    budget_report: BudgetOptimizationReport | None = None
    successive_halving: SuccessiveHalvingPlan | None = None


class RecipeAblationPlanner:
    """Generate baseline, atomic, pairwise, and full recipe experiments."""

    def __init__(self, *, budget_optimizer: BudgetOptimizer | None = None, halving_planner: SuccessiveHalvingPlanner | None = None) -> None:
        self.budget_optimizer = budget_optimizer
        self.halving_planner = halving_planner or SuccessiveHalvingPlanner()

    def plan(self, recipe: CoupledRecipe, baseline: CandidateConfig, *, max_nodes: int = 8) -> RecipeAblationPlan:
        components = list(dict.fromkeys(recipe.component_ids))
        if len(components) < 2:
            raise ValueError("Coupled recipe ablation requires at least two components")
        mandatory_count = 2 + len(components)  # baseline + singles + full
        if max_nodes < mandatory_count:
            raise ValueError(f"Ablation budget requires at least {mandatory_count} nodes to keep baseline, singles, and full recipe")

        baseline_node = RecipeAblationNode(node_id=f"ablate_{baseline.candidate_id}", candidate_config=baseline, component_ids=[], role="baseline", parent_id=baseline.candidate_id, priority=100.0)
        singles = [self._node(recipe, baseline, [component], "single", priority=90.0) for component in components]
        full = self._node(recipe, baseline, components, "full", priority=95.0)
        optional_sets = [list(group) for size in range(2, len(components)) for group in combinations(components, size)]
        remaining = max_nodes - mandatory_count
        selected_optional, omitted, budget_report = self._select_optional(recipe, baseline, optional_sets, remaining)
        pair_nodes = [self._node(recipe, baseline, group, "pair", priority=60.0 - len(group)) for group in selected_optional]
        nodes = [baseline_node, *singles, *pair_nodes, full]

        single_plan = AblationPlan(
            baseline_id=baseline.candidate_id,
            nodes=[AblationNode(node_id=item.node_id, candidate_config=item.candidate_config, parent_id=baseline.candidate_id, changed_variables={"recipe_component": item.component_ids[0]}) for item in singles],
        )
        halving_candidates = [HalvingCandidate(candidate_id=item.candidate_config.candidate_id, node_id=item.node_id, score=item.priority, risk=item.candidate_config.risk, policy_id=recipe.recipe_id) for item in nodes if item.role != "baseline"]
        return RecipeAblationPlan(
            recipe_id=recipe.recipe_id,
            baseline_id=baseline.candidate_id,
            nodes=nodes,
            omitted_combinations=omitted,
            budget_max_nodes=max_nodes,
            single_variable_plan=single_plan,
            budget_report=budget_report,
            successive_halving=self.halving_planner.plan(halving_candidates),
        )

    def assess_contributions(self, plan: RecipeAblationPlan, observations: Iterable[AblationObservation], *, confirmed_seed_count: int = 3) -> list[ContributionAssessment]:
        by_node: dict[str, list[AblationObservation]] = {}
        for item in observations:
            by_node.setdefault(item.node_id, []).append(item)
        assessments: list[ContributionAssessment] = []
        nodes = {item.node_id: item for item in plan.nodes}
        for node_id, records in by_node.items():
            node = nodes.get(node_id)
            if node is None or node.role == "baseline":
                continue
            seeds = {item.seed for item in records}
            metrics = sorted({name for item in records for name in item.deltas})
            means = {name: sum(item.deltas.get(name, 0.0) for item in records) / len(records) for name in metrics}
            consistent = all(_consistent_direction([item.deltas[name] for item in records if name in item.deltas]) for name in metrics)
            confirmed = len(seeds) >= confirmed_seed_count and consistent
            if confirmed:
                reason = f"repeated_seeds:{len(seeds)};consistent_direction"
            elif len(seeds) < confirmed_seed_count:
                reason = f"insufficient_repeated_seeds:{len(seeds)}/{confirmed_seed_count}"
            else:
                reason = "repeated_seeds_but_inconsistent_direction"
            assessments.append(ContributionAssessment(node_id=node_id, component_ids=node.component_ids, seed_count=len(seeds), confidence="confirmed" if confirmed else "possible", mean_deltas=means, reason=reason))
        return sorted(assessments, key=lambda item: item.node_id)

    def _select_optional(self, recipe: CoupledRecipe, baseline: CandidateConfig, groups: list[list[str]], limit: int) -> tuple[list[list[str]], list[list[str]], BudgetOptimizationReport | None]:
        if not groups or limit <= 0:
            return [], groups, None
        evaluations = []
        for group in groups:
            node = self._node(recipe, baseline, group, "pair", priority=50.0 - len(group))
            evaluations.append(LoopPolicyEvaluation(policy_id=node.candidate_config.candidate_id, decision="accepted", priority=node.priority, candidate_config=node.candidate_config, experiment_node=ExperimentNode(node_id=node.node_id, candidate_config=node.candidate_config, data_version="ablation", changed_variables=node.changed_variables)))
        optimizer = self.budget_optimizer or BudgetOptimizer(BudgetOptimizerConfig(max_candidates=max(1, limit), optimizer_kind="utility_rank"))
        report = optimizer.optimize(evaluations)
        actual_selected = report.selected[:limit]
        overflow = [item.model_copy(update={"selected": False, "reason": "deferred_by_recipe_ablation_budget_limit"}) for item in report.selected[limit:]]
        report = report.model_copy(update={"selected": actual_selected, "deferred": [*overflow, *report.deferred], "selected_count": len(actual_selected)})
        selected_ids = {item.arm.candidate_id for item in actual_selected}
        selected = [group for group in groups if self._candidate_id(recipe, group) in selected_ids]
        omitted = [group for group in groups if group not in selected]
        return selected, omitted, report

    def _node(self, recipe: CoupledRecipe, baseline: CandidateConfig, components: list[str], role: MatrixRole, *, priority: float) -> RecipeAblationNode:
        candidate = baseline.model_copy(update={"candidate_id": self._candidate_id(recipe, components), "components": list(components), "train_overrides": {**baseline.train_overrides, **recipe.train_overrides, "profile": "pilot", "imgsz": 640}, "expected_effect": [f"Internal {recipe.recipe_id} ablation: {', '.join(components)}"]})
        return RecipeAblationNode(node_id=f"ablate_{candidate.candidate_id}", candidate_config=candidate, component_ids=list(components), role=role, parent_id=baseline.candidate_id, changed_variables={"recipe_components": list(components)}, priority=priority)

    @staticmethod
    def _candidate_id(recipe: CoupledRecipe, components: list[str]) -> str:
        suffix = "_plus_".join(item.replace(".", "_").replace("-", "_") for item in components)
        return f"{recipe.recipe_id}__{suffix}"


def _consistent_direction(values: list[float]) -> bool:
    nonzero = [value for value in values if value != 0]
    return bool(nonzero) and (all(value > 0 for value in nonzero) or all(value < 0 for value in nonzero))


__all__ = ["AblationObservation", "ContributionAssessment", "RecipeAblationNode", "RecipeAblationPlan", "RecipeAblationPlanner"]
