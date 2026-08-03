"""Internal ablation matrix generation for coupled component recipes."""

from __future__ import annotations

from itertools import combinations
import json
from statistics import stdev
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.ablation_planner import AblationNode, AblationPlan
from yolo_agent.agents.budget_optimizer import BudgetOptimizationReport, BudgetOptimizer, BudgetOptimizerConfig
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluation
from yolo_agent.agents.successive_halving import HalvingCandidate, SuccessiveHalvingPlan, SuccessiveHalvingPlanner
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.round_execution_plan import (
    RoundAblationNode,
    RoundExecutionPlan,
    build_round_execution_plan,
)
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
    guard_metrics: list[str] = Field(default_factory=list)
    attribution_excluded_metrics: list[str] = Field(default_factory=list)
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
    confirmation_metric: str | None = None
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    reason: str


class MinimumAblationCohortStatus(BaseModel):
    required_node_ids: list[str]
    completed_node_ids: list[str]
    failed_node_ids: list[str]
    missing_node_ids: list[str]
    ready_for_asha_elimination: bool
    contribution_ready: bool
    reason: str


class RecipeAblationPlan(BaseModel):
    recipe_id: str
    baseline_id: str
    nodes: list[RecipeAblationNode]
    omitted_combinations: list[list[str]] = Field(default_factory=list)
    budget_max_nodes: int
    target_metrics: list[str] = Field(default_factory=list)
    single_variable_plan: AblationPlan
    budget_report: BudgetOptimizationReport | None = None
    successive_halving: SuccessiveHalvingPlan | None = None
    minimum_internal_ablation_node_ids: list[str] = Field(default_factory=list)

    def minimum_cohort_status(
        self,
        *,
        completed_node_ids: Iterable[str],
        failed_node_ids: Iterable[str] = (),
    ) -> MinimumAblationCohortStatus:
        """Require baseline/A/B/A+B to reach a terminal state before elimination."""
        required = list(self.minimum_internal_ablation_node_ids)
        completed = sorted(set(completed_node_ids) & set(required))
        failed = sorted(set(failed_node_ids) & set(required))
        terminal = set(completed) | set(failed)
        missing = [item for item in required if item not in terminal]
        ready = not missing
        contribution_ready = ready and not failed
        return MinimumAblationCohortStatus(
            required_node_ids=required,
            completed_node_ids=completed,
            failed_node_ids=failed,
            missing_node_ids=missing,
            ready_for_asha_elimination=ready,
            contribution_ready=contribution_ready,
            reason=(
                "minimum_internal_ablation_complete"
                if contribution_ready
                else "minimum_internal_ablation_terminal_with_failures"
                if ready
                else "minimum_internal_ablation_incomplete"
            ),
        )


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

        declared = _declared_ablation_entries(recipe)
        baseline_policy = _entry_for_components(declared, [])
        baseline_node = RecipeAblationNode(
            node_id=f"ablate_{baseline.candidate_id}",
            candidate_config=baseline,
            component_ids=[],
            role="baseline",
            parent_id=baseline.candidate_id,
            priority=100.0,
            guard_metrics=_string_list(baseline_policy.get("guard_metrics")),
            attribution_excluded_metrics=_string_list(
                baseline_policy.get("attribution_excluded_metrics")
            ),
        )
        singles = [
            self._node(
                recipe,
                baseline,
                [component],
                "single",
                priority=99.0 - index,
                policy=_entry_for_components(declared, [component]),
            )
            for index, component in enumerate(components)
        ]
        full = self._node(
            recipe,
            baseline,
            components,
            "full",
            priority=90.0,
            policy=_entry_for_components(declared, components),
        )
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
        halving = _protect_minimum_pilot_cohort(
            self.halving_planner.plan(halving_candidates),
            {item.candidate_config.candidate_id for item in nodes if item.role != "baseline"},
        )
        return RecipeAblationPlan(
            recipe_id=recipe.recipe_id,
            baseline_id=baseline.candidate_id,
            nodes=nodes,
            omitted_combinations=omitted,
            budget_max_nodes=max_nodes,
            target_metrics=list(recipe.target_metrics),
            single_variable_plan=single_plan,
            budget_report=budget_report,
            successive_halving=halving,
            minimum_internal_ablation_node_ids=[item.node_id for item in nodes],
        )

    def assess_contributions(
        self,
        plan: RecipeAblationPlan,
        observations: Iterable[AblationObservation],
        *,
        confirmed_seed_count: int = 3,
    ) -> list[ContributionAssessment]:
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
            excluded = set(node.attribution_excluded_metrics)
            metrics = sorted(
                {name for item in records for name in item.deltas if name not in excluded}
            )
            means = {
                name: sum(item.deltas[name] for item in records if name in item.deltas)
                / sum(1 for item in records if name in item.deltas)
                for name in metrics
            }
            confirmation_metric = _confirmation_metric(plan.target_metrics, metrics)
            interval = _cross_seed_interval(records, confirmation_metric)
            confirmed = (
                len(seeds) >= confirmed_seed_count
                and interval is not None
                and interval[0] > 0.0
            )
            if confirmed:
                reason = f"repeated_seeds:{len(seeds)};positive_confidence_interval"
            elif len(seeds) < confirmed_seed_count:
                reason = f"insufficient_repeated_seeds:{len(seeds)}/{confirmed_seed_count}"
            else:
                reason = "paired_seed_confidence_interval_not_strictly_positive"
            assessments.append(
                ContributionAssessment(
                    node_id=node_id,
                    component_ids=node.component_ids,
                    seed_count=len(seeds),
                    confidence="confirmed" if confirmed else "possible",
                    mean_deltas=means,
                    confirmation_metric=confirmation_metric,
                    confidence_interval_low=interval[0] if interval is not None else None,
                    confidence_interval_high=interval[1] if interval is not None else None,
                    reason=reason,
                )
            )
        return sorted(assessments, key=lambda item: item.node_id)

    def materialize_round_execution_plan(
        self,
        *,
        run_id: str,
        recipe: CoupledRecipe,
        ablation_plan: RecipeAblationPlan,
        baseline_control_node: ExperimentNode,
        prepared_nodes: dict[str, ExperimentNode],
        objective_hash: str | None = None,
        primary_metric: str = "ap_small",
    ) -> RoundExecutionPlan:
        """Create the authoritative paired pilot queue for a guarded coupled ablation."""
        expected = [item for item in ablation_plan.nodes if item.role != "baseline"]
        expected_ids = {item.candidate_config.candidate_id for item in expected}
        missing = sorted(expected_ids - set(prepared_nodes))
        extra = sorted(set(prepared_nodes) - expected_ids)
        if missing or extra:
            raise ValueError(
                "prepared coupled ablation nodes do not match the mandatory matrix: "
                f"missing={missing}; extra={extra}"
            )
        ordered_nodes: list[ExperimentNode] = []
        coupled_node_ids: set[str] = set()
        ranks: dict[str, int] = {}
        for rank, ablation in enumerate(expected, start=1):
            candidate_id = ablation.candidate_config.candidate_id
            node = prepared_nodes[candidate_id]
            if node.candidate_config.components != ablation.component_ids:
                raise ValueError(f"prepared node component mismatch: {candidate_id}")
            node = _annotate_execution_node(node, recipe, ablation)
            ordered_nodes.append(node)
            ranks[candidate_id] = rank
            if ablation.role in {"pair", "full"}:
                coupled_node_ids.add(node.node_id)
        round_plan = build_round_execution_plan(
            run_id=run_id,
            nodes=ordered_nodes,
            ranks=ranks,
            objective_hash=objective_hash,
            primary_metric=primary_metric,
            baseline_control_node=_annotate_baseline_node(
                baseline_control_node,
                recipe,
                ablation_plan.nodes[0],
            ),
            coupled_node_ids=coupled_node_ids,
        )
        round_plan.selected_recipes = [
            {
                "recipe_id": recipe.recipe_id,
                "version": recipe.version,
                "kind": recipe.kind,
                "component_ids": recipe.component_ids,
                "coupling_reason": recipe.coupling_reason,
            }
        ]
        round_plan.require_complete_post_eval = True
        requirements = [
            "predictions.json",
            "coco_eval.json",
            "coco_error_report.json",
            "coco_evidence_contract.json",
            "verified_paired_delta",
        ]
        round_plan.evidence_requirements = {
            item.execution_node_id: list(requirements)
            for item in round_plan.assignments
            if item.status == "active"
        }
        ablation_by_candidate = {
            item.candidate_config.candidate_id: item for item in ablation_plan.nodes
        }
        round_plan.ablation_nodes = [
            RoundAblationNode(
                node_id=item.node_id,
                candidate_id=item.candidate_config.candidate_id,
                parent_id=item.parent_id,
                changed_variables=item.changed_variables,
                component_ids=item.component_ids,
                role=item.role,
                guard_metrics=item.guard_metrics,
                attribution_excluded_metrics=item.attribution_excluded_metrics,
                valid=True,
                reason=(
                    "matched_baseline_control"
                    if item.role == "baseline"
                    else "justified_coupled_recipe"
                    if item.role in {"pair", "full"}
                    else "atomic_recipe_ablation"
                ),
            )
            for item in ablation_plan.nodes
        ]
        for execution in round_plan.execution_nodes:
            candidate = execution.candidate_config.candidate_id
            source = ablation_by_candidate.get(candidate)
            if source is not None:
                _set_execution_evidence_metadata(execution, source)
            else:
                _set_execution_evidence_metadata(execution, ablation_plan.nodes[0])
        return RoundExecutionPlan.model_validate(round_plan.model_dump(mode="json"))

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

    def _node(
        self,
        recipe: CoupledRecipe,
        baseline: CandidateConfig,
        components: list[str],
        role: MatrixRole,
        *,
        priority: float,
        policy: dict[str, Any] | None = None,
    ) -> RecipeAblationNode:
        policy = policy or {}
        candidate = baseline.model_copy(update={"candidate_id": self._candidate_id(recipe, components), "components": list(components), "train_overrides": {**baseline.train_overrides, **recipe.train_overrides, "profile": "pilot", "imgsz": 640}, "expected_effect": [f"Internal {recipe.recipe_id} ablation: {', '.join(components)}"]})
        changed = policy.get("changed_variables")
        if not isinstance(changed, dict):
            changed = {"recipe_components": list(components)}
        return RecipeAblationNode(
            node_id=f"ablate_{candidate.candidate_id}",
            candidate_config=candidate,
            component_ids=list(components),
            role=role,
            parent_id=baseline.candidate_id,
            changed_variables=changed,
            guard_metrics=_string_list(policy.get("guard_metrics")),
            attribution_excluded_metrics=_string_list(
                policy.get("attribution_excluded_metrics")
            ),
            priority=priority,
        )

    @staticmethod
    def _candidate_id(recipe: CoupledRecipe, components: list[str]) -> str:
        suffix = "_plus_".join(item.replace(".", "_").replace("-", "_") for item in components)
        return f"{recipe.recipe_id}__{suffix}"


def _declared_ablation_entries(recipe: CoupledRecipe) -> list[dict[str, Any]]:
    entries = [item for item in recipe.internal_ablation_plan if isinstance(item, dict)]
    if not any(isinstance(item.get("components"), list) for item in entries):
        return []
    groups = [tuple(_string_list(item.get("components"))) for item in entries]
    required = {(), *((component,) for component in recipe.component_ids), tuple(recipe.component_ids)}
    if not required.issubset(set(groups)):
        raise ValueError("declared internal ablation must contain baseline, every single, and full recipe")
    return entries


def _protect_minimum_pilot_cohort(
    plan: SuccessiveHalvingPlan,
    mandatory_candidate_ids: set[str],
) -> SuccessiveHalvingPlan:
    """Make pilot_3 evidence-driven by preventing pre-run coupled-arm elimination."""
    assignments = [
        item.model_copy(
            update={
                "decision": "run",
                "reason": "minimum_coupled_ablation_pilot_required",
            }
        )
        if item.stage_id == "pilot_3"
        and item.candidate_id in mandatory_candidate_ids
        else item
        for item in plan.assignments
    ]
    return plan.model_copy(
        update={
            "assignments": assignments,
            "eliminated": [
                item for item in plan.eliminated if item not in mandatory_candidate_ids
            ],
            "guardrail": (
                "minimum_coupled_ablation_precedes_evidence_driven_elimination"
            ),
        }
    )


def _entry_for_components(
    entries: list[dict[str, Any]], components: list[str]
) -> dict[str, Any]:
    return next(
        (
            item
            for item in entries
            if _string_list(item.get("components")) == components
        ),
        {},
    )


def _confirmation_metric(target_metrics: list[str], observed: list[str]) -> str | None:
    preferred = ["ap_small", "target_class_recall", "recall", "map50_95"]
    return next(
        (name for name in [*target_metrics, *preferred] if name in observed),
        observed[0] if observed else None,
    )


def _cross_seed_interval(
    records: list[AblationObservation], metric_name: str | None
) -> tuple[float, float] | None:
    if metric_name is None:
        return None
    by_seed: dict[int, list[float]] = {}
    for item in records:
        if metric_name in item.deltas:
            by_seed.setdefault(item.seed, []).append(item.deltas[metric_name])
    values = [sum(items) / len(items) for items in by_seed.values()]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    standard_error = stdev(values) / (len(values) ** 0.5)
    critical = _student_t_critical(len(values) - 1)
    return mean - critical * standard_error, mean + critical * standard_error


def _student_t_critical(degrees_of_freedom: int) -> float:
    values = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    for upper in sorted(values):
        if degrees_of_freedom <= upper:
            return values[upper]
    return 1.96


def _annotate_execution_node(
    node: ExperimentNode,
    recipe: CoupledRecipe,
    ablation: RecipeAblationNode,
) -> ExperimentNode:
    spec = node.command_spec
    if spec is None:
        raise ValueError(f"prepared ablation node requires CommandSpec: {node.node_id}")
    spec = spec.model_copy(
        update={
            "metadata": {
                **spec.metadata,
                "guarded_coupled_ablation_member": True,
                "coupled_recipe_id": recipe.recipe_id,
                "coupling_reason": recipe.coupling_reason,
                "internal_ablation_plan": json.dumps(
                    recipe.internal_ablation_plan,
                    sort_keys=True,
                ),
                "ablation_role": ablation.role,
            }
        }
    )
    return node.model_copy(
        update={
            "command_spec": spec,
            "command": spec.display(),
            "parent_id": ablation.parent_id,
            "changed_variables": ablation.changed_variables,
        }
    )


def _annotate_baseline_node(
    node: ExperimentNode,
    recipe: CoupledRecipe,
    ablation: RecipeAblationNode,
) -> ExperimentNode:
    spec = node.command_spec
    if spec is None:
        raise ValueError("coupled ablation baseline requires CommandSpec")
    spec = spec.model_copy(
        update={
            "metadata": {
                **spec.metadata,
                "guarded_coupled_ablation_member": True,
                "coupled_recipe_id": recipe.recipe_id,
                "ablation_role": "baseline",
            }
        }
    )
    return node.model_copy(update={"command_spec": spec, "command": spec.display()})


def _set_execution_evidence_metadata(
    node: ExperimentNode,
    ablation: RecipeAblationNode,
) -> None:
    if node.command_spec is None:
        return
    node.command_spec = node.command_spec.model_copy(
        update={
            "metadata": {
                **node.command_spec.metadata,
                "post_eval_required": True,
                "coco_error_facts_required": True,
                "matched_control_required": ablation.role != "baseline",
                "ablation_role": ablation.role,
                "guard_metrics": ",".join(ablation.guard_metrics),
                "attribution_excluded_metrics": ",".join(
                    ablation.attribution_excluded_metrics
                ),
            }
        }
    )
    node.command = node.command_spec.display()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


__all__ = [
    "AblationObservation",
    "ContributionAssessment",
    "MinimumAblationCohortStatus",
    "RecipeAblationNode",
    "RecipeAblationPlan",
    "RecipeAblationPlanner",
]
