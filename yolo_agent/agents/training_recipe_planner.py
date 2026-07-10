"""Evidence-driven planner for executable Ultralytics training recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.core.experiment_graph import Evidence
from yolo_agent.core.run_context import RunContext
from yolo_agent.resources import ResourcePaths


RecipeDecision = Literal["selected", "exhausted", "rejected_by_evidence", "not_relevant"]


class TrainingRecipeVariant(BaseModel):
    action_id: str
    overrides: dict[str, str | int | float | bool]


class TrainingRecipe(BaseModel):
    family: str
    action_domain: Literal["training", "augmentation"]
    trigger_actions: list[str] = Field(default_factory=list)
    target_fact_types: list[str] = Field(default_factory=list)
    metric_name: str = "map50_95"
    expected_gain: float = Field(default=0.3, ge=0.0)
    priority: float = Field(default=3.2, ge=0.0)
    minimum_effect_delta: float = 0.0002
    stop_after_non_positive: int = Field(default=2, ge=1)
    effect: str
    variants: list[TrainingRecipeVariant]


class TrainingRecipeCatalog(BaseModel):
    max_recipes_per_round: int = Field(default=3, ge=1)
    recipes: list[TrainingRecipe]

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "TrainingRecipeCatalog":
        config_path = Path(path) if path is not None else ResourcePaths.TRAINING_RECIPES
        with config_path.open("r", encoding="utf-8-sig") as file:
            raw = yaml.safe_load(file) or {}
        return cls.model_validate(raw)


class RecipeFamilyDecision(BaseModel):
    family: str
    decision: RecipeDecision
    tried_actions: list[str] = Field(default_factory=list)
    observed_deltas: dict[str, float] = Field(default_factory=dict)
    selected_action: str | None = None
    reason: str


class TrainingRecipePlan(BaseModel):
    baseline_metric: float | None = None
    policies: list[CandidatePolicy] = Field(default_factory=list)
    family_decisions: list[RecipeFamilyDecision] = Field(default_factory=list)


class TrainingRecipePlanner:
    """Choose new single-variable recipes using diagnosis relevance and prior deltas."""

    def __init__(self, catalog: TrainingRecipeCatalog | None = None) -> None:
        self.catalog = catalog or TrainingRecipeCatalog.from_yaml()

    def plan(
        self,
        *,
        context: RunContext,
        evidence: Evidence,
        focus_items: list[dict[str, Any]],
        allowed_actions: set[str],
        tried_actions: set[str],
        existing_policy_ids: set[str] | None = None,
    ) -> TrainingRecipePlan:
        existing = existing_policy_ids or set()
        candidate_metrics = _candidate_metrics(evidence, "map50_95")
        baseline = _baseline_metric(candidate_metrics)
        policies: list[CandidatePolicy] = []
        decisions: list[RecipeFamilyDecision] = []
        for recipe in sorted(self.catalog.recipes, key=lambda item: item.priority, reverse=True):
            targets = _targets_for_recipe(focus_items, recipe, allowed_actions)
            if not targets:
                decisions.append(RecipeFamilyDecision(family=recipe.family, decision="not_relevant", reason="No matching error facts or actions."))
                continue
            variants = {variant.action_id: variant for variant in recipe.variants}
            tried = [action for action in variants if action in tried_actions]
            deltas = _observed_deltas(candidate_metrics, variants, baseline)
            if len(deltas) >= recipe.stop_after_non_positive and max(deltas.values()) <= recipe.minimum_effect_delta:
                decisions.append(
                    RecipeFamilyDecision(
                        family=recipe.family,
                        decision="rejected_by_evidence",
                        tried_actions=tried,
                        observed_deltas=deltas,
                        reason="Historical pilot variants did not improve map50_95.",
                    )
                )
                continue
            variant = next((item for item in recipe.variants if item.action_id not in tried_actions), None)
            if variant is None:
                decisions.append(
                    RecipeFamilyDecision(
                        family=recipe.family,
                        decision="exhausted",
                        tried_actions=tried,
                        observed_deltas=deltas,
                        reason="All configured variants were already tested.",
                    )
                )
                continue
            policy_id = f"next_{recipe.action_domain}_{variant.action_id}"
            if policy_id in existing:
                continue
            overrides = dict(variant.overrides)
            overrides[f"{recipe.action_domain}_action"] = variant.action_id
            matched_diagnosis_actions = sorted(set(recipe.trigger_actions).intersection(allowed_actions))
            overrides["target_actions"] = [variant.action_id, *matched_diagnosis_actions]
            policies.append(
                CandidatePolicy(
                    policy_id=policy_id,
                    source="rule_engine",
                    action_domain=recipe.action_domain,
                    action_id=variant.action_id,
                    execution_action="run_training",
                    base_model=str(context.metadata.get("training_model") or "yolo26n.pt"),
                    scale=_model_scale(str(context.metadata.get("training_model") or "yolo26n.pt")),
                    framework="ultralytics",
                    train_overrides=overrides,
                    target_error_facts=targets,
                    expected_improvement={
                        "metric_name": recipe.metric_name,
                        "expected_gain": {recipe.metric_name: recipe.expected_gain},
                        "minimum_expected_delta": recipe.minimum_effect_delta,
                        "summary": recipe.effect,
                    },
                    priority_hint=recipe.priority,
                    expected_effect=[recipe.effect],
                    risk="low",
                    rationale=f"Evidence-driven recipe family {recipe.family}; next untried variant {variant.action_id}.",
                )
            )
            decisions.append(
                RecipeFamilyDecision(
                    family=recipe.family,
                    decision="selected",
                    tried_actions=tried,
                    observed_deltas=deltas,
                    selected_action=variant.action_id,
                    reason="Selected next untried variant after evidence review.",
                )
            )
            if len(policies) >= self.catalog.max_recipes_per_round:
                break
        return TrainingRecipePlan(baseline_metric=baseline, policies=policies, family_decisions=decisions)


def _candidate_metrics(evidence: Evidence, metric_name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for record in sorted(evidence.metric_records, key=lambda item: item.created_at):
        if record.metric_name != metric_name or not record.verified or not isinstance(record.value, (int, float)):
            continue
        values[record.candidate_id] = float(record.value)
    return values


def _baseline_metric(values: dict[str, float]) -> float | None:
    preferred = [value for candidate, value in values.items() if "coco_pilot" in candidate or "reduce_mosaic_strength" in candidate]
    if preferred:
        return max(preferred)
    return max(values.values()) if values else None


def _observed_deltas(
    values: dict[str, float],
    variants: dict[str, TrainingRecipeVariant],
    baseline: float | None,
) -> dict[str, float]:
    if baseline is None:
        return {}
    deltas: dict[str, float] = {}
    for action in variants:
        matches = [value for candidate, value in values.items() if candidate.endswith(action)]
        if matches:
            deltas[action] = round(max(matches) - baseline, 6)
    return deltas


def _targets_for_recipe(
    focus_items: list[dict[str, Any]],
    recipe: TrainingRecipe,
    allowed_actions: set[str],
) -> list[dict[str, Any]]:
    if recipe.trigger_actions and not set(recipe.trigger_actions).intersection(allowed_actions):
        return []
    targets = [
        item
        for item in focus_items
        if not recipe.target_fact_types or str(item.get("fact_type")) in recipe.target_fact_types
    ]
    return targets or focus_items[:1]


def _model_scale(model: str) -> str:
    stem = Path(model).stem.lower()
    return next((scale for scale in ("n", "s", "m", "l", "x") if stem.endswith(scale)), "n")


__all__ = [
    "RecipeFamilyDecision",
    "TrainingRecipe",
    "TrainingRecipeCatalog",
    "TrainingRecipePlan",
    "TrainingRecipePlanner",
    "TrainingRecipeVariant",
]
