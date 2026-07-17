"""Evidence-driven planning over paper components and recipe bundles."""

from __future__ import annotations

from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.budget_optimizer import BudgetOptimizer, BudgetOptimizerConfig
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluation
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.agents.utility_scorer import UtilityCost, UtilityScore, UtilityScorer
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.components.yolo26_compatibility import YOLO26CompatibilityChecker
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import ExperimentNode, MetricEvidence
from yolo_agent.core.policy_memory import (
    ActionFingerprint,
    PolicyMemoryRecord,
    PolicyMemoryStore,
    stable_negative_action_reasons,
)
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.core.schemas import DeploymentConstraints
from yolo_agent.core.task_spec import MetricPriority, TaskSpec
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.recipes.schemas import RecipeSpec
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.tools.dataset_stats import DatasetReport


RecipeDecision = Literal["selected", "deferred", "rejected", "implementation_proposal", "needs_evidence"]


class PlannedRecipe(BaseModel):
    recipe_id: str
    version: str
    decision: RecipeDecision
    reasons: list[str] = Field(default_factory=list)
    expected_metric_targets: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cost: UtilityCost = Field(default_factory=UtilityCost)
    stop_conditions: list[str] = Field(default_factory=list)
    required_adapters: list[str] = Field(default_factory=list)
    related_papers: list[str] = Field(default_factory=list)
    utility: float = 0.0


class PaperRecipePlan(BaseModel):
    selected_recipes: list[PlannedRecipe] = Field(default_factory=list)
    deferred_recipes: list[PlannedRecipe] = Field(default_factory=list)
    rejected_recipes: list[PlannedRecipe] = Field(default_factory=list)
    evidence_actions: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    expected_metric_targets: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    cost: UtilityCost = Field(default_factory=UtilityCost)
    stop_conditions: list[str] = Field(default_factory=list)
    training_profile: str = "pilot"
    fixed_imgsz: int = 640


class PaperRecipePlanner:
    """Select evidence-grounded recipes without materializing full candidates."""

    def __init__(self, utility_scorer: UtilityScorer | None = None, budget_optimizer: BudgetOptimizer | None = None) -> None:
        self.utility_scorer = utility_scorer or UtilityScorer()
        self.budget_optimizer = budget_optimizer or BudgetOptimizer(BudgetOptimizerConfig(max_candidates=3))

    def plan(
        self,
        *,
        error_facts: Iterable[ErrorFact],
        dataset_report: DatasetReport | None,
        node_metrics: Iterable[MetricEvidence],
        policy_memory: PolicyMemoryStore | Iterable[PolicyMemoryRecord],
        paper_registry: PaperRegistry,
        component_registry: ComponentRegistry,
        recipe_registry: RecipeRegistry,
        deployment: DeploymentConstraints | None = None,
        tried_actions: Iterable[str] = (),
        rejected_component_families: Iterable[str] = (),
        training_budget: dict[str, Any] | None = None,
        optimization_objective: OptimizationObjective | None = None,
    ) -> PaperRecipePlan:
        facts = list(error_facts)
        metrics = list(node_metrics)
        memory = policy_memory.read() if isinstance(policy_memory, PolicyMemoryStore) else list(policy_memory)
        tried = set(tried_actions)
        rejected_families = set(rejected_component_families)
        budget = training_budget or {}
        if str(budget.get("profile", "pilot")) == "candidate_full":
            budget = {**budget, "profile": "pilot"}

        evidence_actions = _evidence_actions(facts, metrics, dataset_report)
        if not facts:
            return PaperRecipePlan(
                evidence_actions=evidence_actions or ["mine_coco_error_facts"],
                reasons=["No current error facts; training proposals are forbidden until diagnosis evidence exists."],
                training_profile="pilot",
            )

        categories = _categories_for_facts(facts)
        papers = _papers_for_categories(paper_registry, categories)
        paper_ids = {paper.paper_id for paper in papers}
        task = _task_spec(deployment)
        registry_items = list(component_registry.cards)
        if registry_items and isinstance(registry_items[0], ComponentContract):
            contracts = {item.component_id: item for item in registry_items}
        else:
            contracts = {item.component_id: item for item in component_registry.get_contracts()}
        decisions: list[tuple[RecipeSpec, PlannedRecipe, UtilityScore | None]] = []
        guarded: list[LoopPolicyEvaluation] = []

        for recipe in recipe_registry.list():
            reasons: list[str] = []
            related_papers = sorted(paper_ids & set(recipe.coupling_source_papers))
            if not _recipe_matches(recipe, facts, categories, papers):
                continue
            if recipe.train_overrides.get("imgsz", 640) != 640 or recipe.fixed_variables.get("imgsz") != 640:
                decisions.append((recipe, _planned(recipe, "rejected", ["fixed_imgsz_must_equal_640"]), None))
                continue
            family_hits = _recipe_families(recipe) & rejected_families
            if family_hits:
                decisions.append((recipe, _planned(recipe, "rejected", [f"rejected_component_family:{item}" for item in sorted(family_hits)]), None))
                continue
            negative_memory = _negative_memory(recipe, memory)
            if negative_memory:
                decisions.append((recipe, _planned(recipe, "rejected", negative_memory), None))
                continue
            missing_contracts = sorted(set(recipe.component_ids) - set(contracts))
            metadata_components = [item for item in recipe.component_ids if item in contracts and not contracts[item].can_execute]
            if missing_contracts or metadata_components:
                required = [contracts[item].adapter_class or f"{item}.adapter" for item in metadata_components if item in contracts]
                reasons.extend([f"unknown_component:{item}" for item in missing_contracts])
                reasons.extend([f"implementation_required:{item}" for item in metadata_components])
                decisions.append((recipe, _planned(recipe, "implementation_proposal", reasons, required_adapters=required, related_papers=related_papers), None))
                continue
            yolo26 = YOLO26CompatibilityChecker().check(
                components=[contracts[item] for item in recipe.component_ids],
                train_overrides=recipe.train_overrides,
                changed_variables=None,
                single_variable=not bool(recipe.coupled_variables),
                export_format=(deployment.preferred_export if deployment else "none"),
            )
            if yolo26.incompatible:
                decisions.append((recipe, _planned(recipe, "rejected", yolo26.blocked_by, required_adapters=yolo26.required_adapters, related_papers=related_papers), None))
                continue
            missing_recipe_evidence = _missing_recipe_evidence(recipe, metrics, dataset_report)
            if missing_recipe_evidence:
                evidence_actions.extend(missing_recipe_evidence)
                decisions.append((recipe, _planned(recipe, "needs_evidence", ["Missing evidence before training."], related_papers=related_papers), None))
                continue

            proposal = _proposal(
                recipe,
                facts,
                tried,
                budget,
                memory_prior=_memory_prior(recipe, memory),
            )
            utility = self.utility_scorer.score(
                proposal=proposal,
                task_spec=task,
                changed_variables={recipe.primary_changed_variable: recipe.recipe_id},
                error_facts=facts,
                optimization_objective=optimization_objective,
                policy_memory=memory,
                action_fingerprint=ActionFingerprint(
                    action=recipe.recipe_id,
                    recipe_id=recipe.recipe_id,
                    recipe_version=recipe.version,
                    component_versions={
                        component_id: contracts[component_id].schema_version
                        for component_id in recipe.component_ids
                        if component_id in contracts
                    },
                    changed_variable=recipe.primary_changed_variable,
                    before_value=recipe.fixed_variables.get(recipe.primary_changed_variable),
                    after_value=recipe.train_overrides.get(recipe.primary_changed_variable, recipe.recipe_id),
                    model_family="yolo26",
                    dataset_signature=str(budget.get("dataset_signature") or "unknown"),
                    protocol_hash=str(budget.get("protocol_hash") or "unknown"),
                    fidelity=str(budget.get("fidelity") or "pilot_3"),
                    seed=budget.get("seed", 1),
                ),
                observed_pilot_delta=_numeric_or_none(budget.get("observed_pilot_delta")),
            )
            planned = _planned(
                recipe,
                "deferred" if utility.decision == "reject" else "selected",
                utility.reasons,
                utility=utility,
                related_papers=related_papers,
            )
            decisions.append((recipe, planned, utility))
            if planned.decision == "selected":
                guarded.append(_guarded_evaluation(recipe, proposal, utility))

        allocation = self.budget_optimizer.optimize(guarded)
        selected_ids = {item.arm.policy_id for item in allocation.selected}
        selected: list[PlannedRecipe] = []
        deferred: list[PlannedRecipe] = []
        rejected: list[PlannedRecipe] = []
        for recipe, decision, _ in decisions:
            if decision.decision == "selected" and recipe.recipe_id not in selected_ids:
                decision = decision.model_copy(update={"decision": "deferred", "reasons": [*decision.reasons, "deferred_by_budget_optimizer"]})
            if decision.decision == "selected":
                selected.append(decision)
            elif decision.decision in {"deferred", "needs_evidence", "implementation_proposal"}:
                deferred.append(decision)
            else:
                rejected.append(decision)

        return PaperRecipePlan(
            selected_recipes=selected,
            deferred_recipes=deferred,
            rejected_recipes=rejected,
            evidence_actions=sorted(set(evidence_actions)),
            reasons=[f"Matched error categories: {', '.join(sorted(categories))}."],
            expected_metric_targets=_aggregate_targets(selected),
            confidence=max((item.confidence for item in selected), default=0.0),
            cost=_aggregate_cost(selected),
            stop_conditions=sorted({condition for item in selected for condition in item.stop_conditions}),
            training_profile="pilot",
        )


def _categories_for_facts(facts: list[ErrorFact]) -> set[str]:
    mapping = {
        "area_metric": {"detection_head", "feature_pyramid", "augmentation", "sampling", "bbox_regression_loss", "assigner"},
        "false_negative_heavy_class": {"assigner", "sampling", "augmentation", "classification_loss"},
        "localization_heavy_class": {"bbox_regression_loss", "quality_estimation", "assigner"},
        "background_false_positive_class": {"sampling", "classification_loss", "threshold", "label_quality"},
        "class_confusion_pair": {"classification_loss", "sampling", "label_quality"},
    }
    categories: set[str] = set()
    for fact in facts:
        categories.update(mapping.get(fact.fact_type, {"sampling", "augmentation"}))
    return categories


def _papers_for_categories(registry: PaperRegistry, categories: set[str]) -> list[Any]:
    papers: dict[str, Any] = {}
    for category in categories:
        for paper in registry.list(component_category=category):
            papers[paper.paper_id] = paper
    return list(papers.values())


def _recipe_matches(recipe: RecipeSpec, facts: list[ErrorFact], categories: set[str], papers: list[Any]) -> bool:
    fact_text = " ".join(f"{item.fact_type} {item.subject} {item.area or ''} {' '.join(item.action_candidates)}" for item in facts).lower()
    recipe_text = f"{recipe.recipe_id} {recipe.primary_changed_variable} {recipe.target_error_facts} {recipe.component_ids}".lower()
    if any(token in recipe_text for token in fact_text.split() if len(token) > 3):
        return True
    paper_components = {component for paper in papers for component in paper.component_ids}
    return bool(set(recipe.component_ids) & paper_components or _recipe_families(recipe) & categories)


def _recipe_families(recipe: RecipeSpec) -> set[str]:
    values = {recipe.primary_changed_variable, *recipe.coupled_variables}
    for component in recipe.component_ids:
        values.add(component.split(".", 1)[0])
    aliases = {"head": "detection_head", "loss": "bbox_regression_loss", "bbox_loss": "bbox_regression_loss"}
    return {aliases.get(item, item) for item in values}


def _negative_memory(recipe: RecipeSpec, records: list[PolicyMemoryRecord]) -> list[str]:
    actions = {recipe.recipe_id, recipe.primary_changed_variable, *recipe.component_ids, *recipe.data_actions, *recipe.inference_actions}
    return stable_negative_action_reasons(records, actions)


def _evidence_actions(facts: list[ErrorFact], metrics: list[MetricEvidence], report: DatasetReport | None) -> list[str]:
    actions: list[str] = []
    if not facts:
        actions.append("mine_coco_error_facts")
    metric_names = {item.metric_name for item in metrics if item.verified}
    if "map50_95" not in metric_names:
        actions.append("import_verified_map50_95")
    if report is None:
        actions.append("profile_dataset")
    return actions


def _missing_recipe_evidence(recipe: RecipeSpec, metrics: list[MetricEvidence], report: DatasetReport | None) -> list[str]:
    available = {item.metric_name for item in metrics if item.verified}
    actions: list[str] = []
    for requirement in recipe.compatibility_requirements:
        if requirement in {"dataset_report", "dataset_profile"} and report is None:
            actions.append("profile_dataset")
        if requirement.startswith("metric:") and requirement.split(":", 1)[1] not in available:
            actions.append(f"import_{requirement.split(':', 1)[1]}")
    return actions


def _proposal(
    recipe: RecipeSpec,
    facts: list[ErrorFact],
    tried: set[str],
    budget: dict[str, Any],
    *,
    memory_prior: dict[str, Any] | None = None,
) -> CandidatePolicy:
    expected = {key: value for key, value in recipe.expected_effects.items() if isinstance(value, (int, float))}
    prior = memory_prior or {}
    prior_effect = prior.get("mean_effect_delta")
    if isinstance(prior_effect, (int, float)):
        expected["policy_memory_prior"] = float(prior_effect)
    expected_improvement: dict[str, Any] = {"expected_gain": expected}
    if isinstance(prior.get("confidence"), (int, float)):
        expected_improvement["confidence"] = float(prior["confidence"])
    priority = 0.8 if recipe.recipe_id not in tried else 0.3
    priority *= float(prior.get("priority_multiplier", 1.0))
    return CandidatePolicy(
        policy_id=recipe.recipe_id,
        source="rule_engine",
        action_domain="model" if recipe.component_ids else "data",
        base_model=str(recipe.fixed_variables.get("model", "yolo26n.pt")),
        scale="n",
        framework="ultralytics",
        components=list(recipe.component_ids),
        train_overrides={**recipe.train_overrides, "profile": "pilot", "gpu_hours": budget.get("gpu_hours", 1.0)},
        target_error_facts=[item.model_dump(mode="json") for item in facts],
        expected_improvement=expected_improvement,
        priority_hint=round(priority, 6),
        expected_effect=[str(item) for item in recipe.expected_effects],
        risk=recipe.implementation_risk if recipe.implementation_risk != "unknown" else "medium",
        rationale=(
            "Evidence-driven paper recipe proposal; pilot only. "
            f"Policy-memory posterior: {prior.get('interpretation', 'no_local_prior')}."
        ),
    )


def _numeric_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _memory_prior(recipe: RecipeSpec, records: list[PolicyMemoryRecord]) -> dict[str, Any]:
    """Return a soft local prior; isolated failures reduce confidence but never hard-ban."""
    actions = {recipe.recipe_id, recipe.primary_changed_variable, *recipe.component_ids, *recipe.data_actions, *recipe.inference_actions}
    matched = [record for record in records if record.action in actions and record.effect_delta is not None]
    if not matched:
        return {}
    weights = [max(record.seed_count, 1) for record in matched]
    total = sum(weights)
    mean = sum(float(record.effect_delta) * weight for record, weight in zip(matched, weights)) / total
    if mean < 0:
        confidence = 0.2 if total == 1 else min(0.55, 0.25 + total * 0.08)
        multiplier = 0.75 if total == 1 else 0.6
        interpretation = "negative_prior_lowers_confidence"
    elif mean > 0:
        confidence = min(0.9, 0.35 + total * 0.12)
        multiplier = min(1.2, 1.0 + total * 0.03)
        interpretation = "positive_local_prior"
    else:
        confidence = 0.25
        multiplier = 0.8
        interpretation = "neutral_local_prior"
    return {
        "mean_effect_delta": round(mean, 6),
        "seed_count": total,
        "confidence": round(confidence, 6),
        "priority_multiplier": round(multiplier, 6),
        "interpretation": interpretation,
    }


def _guarded_evaluation(recipe: RecipeSpec, proposal: CandidatePolicy, utility: UtilityScore) -> LoopPolicyEvaluation:
    candidate = CandidateConfig(candidate_id=recipe.recipe_id, base_model=proposal.base_model, scale="n", framework="ultralytics", components=recipe.component_ids, train_overrides={**recipe.train_overrides, "target_error_facts": proposal.target_error_facts}, risk=proposal.risk)
    node = ExperimentNode(node_id=f"planner_{recipe.recipe_id}", candidate_config=candidate, data_version="planning", changed_variables={recipe.primary_changed_variable: recipe.recipe_id}, command_spec=CommandSpec(command_type="custom", argv=["planner-only"], metadata={"bandit_pulls": 0}))
    return LoopPolicyEvaluation(policy_id=recipe.recipe_id, decision="accepted", priority=utility.utility, utility_score=utility, candidate_config=candidate, experiment_node=node)


def _planned(recipe: RecipeSpec, decision: RecipeDecision, reasons: list[str], *, utility: UtilityScore | None = None, required_adapters: list[str] | None = None, related_papers: list[str] | None = None) -> PlannedRecipe:
    return PlannedRecipe(recipe_id=recipe.recipe_id, version=recipe.version, decision=decision, reasons=reasons, expected_metric_targets=recipe.expected_effects, confidence=utility.confidence if utility else 0.0, cost=utility.cost if utility else UtilityCost(), stop_conditions=recipe.stop_conditions, required_adapters=sorted(set(required_adapters or [])), related_papers=related_papers or [], utility=utility.utility if utility else 0.0)


def _task_spec(deployment: DeploymentConstraints | None) -> TaskSpec:
    return TaskSpec(task_type="detect", scene="generic", class_names=["object"], primary_metric=MetricPriority(name="map50_95"), max_latency_ms=deployment.max_latency_ms if deployment else None, max_model_size_mb=deployment.max_model_size_mb if deployment else None)


def _aggregate_targets(items: list[PlannedRecipe]) -> dict[str, Any]:
    return {key: value for item in items for key, value in item.expected_metric_targets.items()}


def _aggregate_cost(items: list[PlannedRecipe]) -> UtilityCost:
    return UtilityCost(gpu_hours=sum(item.cost.gpu_hours for item in items), training_cost=sum(item.cost.training_cost for item in items), latency_risk=sum(item.cost.latency_risk for item in items), model_size_risk=sum(item.cost.model_size_risk for item in items), implementation_risk=sum(item.cost.implementation_risk for item in items), evidence_gap_penalty=sum(item.cost.evidence_gap_penalty for item in items))


__all__ = ["PaperRecipePlan", "PaperRecipePlanner", "PlannedRecipe"]
