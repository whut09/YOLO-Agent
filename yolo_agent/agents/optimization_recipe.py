"""Optimization recipes that jointly select loss, assigner, head, and guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.resources import ResourcePaths
from yolo_agent.tools.dataset_stats import DatasetReport
from yolo_agent.utils import dedupe_list


class OptimizationRecipeConditions(BaseModel):
    """Conditions that decide whether a recipe applies."""

    scenes: list[str] = Field(default_factory=list)
    error_types: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    health_problems: list[str] = Field(default_factory=list)
    min_error_count: int | None = Field(default=None, ge=0)
    min_severity: str | None = None


class RecipeComponents(BaseModel):
    """Component choices proposed by a recipe."""

    bbox_loss: list[str] = Field(default_factory=list)
    head: list[str] = Field(default_factory=list)
    assigner: list[str] = Field(default_factory=list)
    neck: list[str] = Field(default_factory=list)
    augmentation: list[str] = Field(default_factory=list)

    def all_ids(self) -> list[str]:
        """Return all component ids in stable order."""
        return dedupe_list(
            [
                *self.bbox_loss,
                *self.head,
                *self.assigner,
                *self.neck,
                *self.augmentation,
            ]
        )


class OptimizationRecipeRule(BaseModel):
    """Configurable optimization recipe."""

    conditions: OptimizationRecipeConditions = Field(default_factory=OptimizationRecipeConditions)
    components: RecipeComponents = Field(default_factory=RecipeComponents)
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    postprocess: list[str] = Field(default_factory=list)
    data_checks: list[str] = Field(default_factory=list)
    expected_effect: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence_required: list[str] = Field(default_factory=list)


class OptimizationRecipeRecommendation(BaseModel):
    """One matched recipe with rationale."""

    recipe_id: str
    components: RecipeComponents = Field(default_factory=RecipeComponents)
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    postprocess: list[str] = Field(default_factory=list)
    data_checks: list[str] = Field(default_factory=list)
    expected_effect: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence_required: list[str] = Field(default_factory=list)
    rationale: str


class OptimizationRecipePlan(BaseModel):
    """Merged recipe plan for a task and observed evidence."""

    task_scene: str
    recommendations: list[OptimizationRecipeRecommendation] = Field(default_factory=list)
    component_candidates: RecipeComponents = Field(default_factory=RecipeComponents)
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    postprocess: list[str] = Field(default_factory=list)
    data_checks: list[str] = Field(default_factory=list)
    expected_effect: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence_required: list[str] = Field(default_factory=list)


class OptimizationRecipeEngine:
    """Select joint optimization recipes from task, error, and dataset evidence."""

    def __init__(self, recipes: dict[str, OptimizationRecipeRule]) -> None:
        self.recipes = recipes

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "OptimizationRecipeEngine":
        """Load recipe rules from YAML."""
        recipe_path = Path(path) if path is not None else default_optimization_recipe_path()
        with recipe_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Optimization recipe YAML must contain a mapping: {recipe_path}")
        raw_recipes = data.get("recipes", {})
        if not isinstance(raw_recipes, dict):
            raise ValueError("Optimization recipe YAML requires a 'recipes' mapping.")
        return cls(
            {
                str(recipe_id): OptimizationRecipeRule.model_validate(recipe)
                for recipe_id, recipe in raw_recipes.items()
                if isinstance(recipe, dict)
            }
        )

    def recommend(
        self,
        task_spec: TaskSpec,
        error_profile: list[DetectionErrorObservation] | None = None,
        dataset_report: DatasetReport | None = None,
    ) -> OptimizationRecipePlan:
        """Return matched and merged recipes for current evidence."""
        observations = error_profile or []
        recommendations: list[OptimizationRecipeRecommendation] = []
        for recipe_id, recipe in self.recipes.items():
            if not _matches(recipe, task_spec, observations, dataset_report):
                continue
            recommendations.append(
                OptimizationRecipeRecommendation(
                    recipe_id=recipe_id,
                    components=recipe.components,
                    train_overrides=recipe.train_overrides,
                    postprocess=recipe.postprocess,
                    data_checks=recipe.data_checks,
                    expected_effect=recipe.expected_effect,
                    risks=recipe.risks,
                    evidence_required=recipe.evidence_required,
                    rationale=_rationale(recipe_id, recipe, task_spec, observations, dataset_report),
                )
            )
        return _merge_plan(task_spec.scene, recommendations)


def default_optimization_recipe_path() -> Path:
    """Return bundled optimization recipe config."""
    return ResourcePaths.OPTIMIZATION_RECIPES


def _matches(
    recipe: OptimizationRecipeRule,
    task_spec: TaskSpec,
    observations: list[DetectionErrorObservation],
    dataset_report: DatasetReport | None,
) -> bool:
    conditions = recipe.conditions
    if conditions.scenes and task_spec.scene not in set(conditions.scenes):
        return False

    metric_names = {task_spec.primary_metric.name}
    metric_names.update(metric.name for metric in task_spec.secondary_metrics)
    if conditions.metrics and not metric_names.intersection(conditions.metrics):
        return False

    if conditions.error_types:
        matched = [
            observation
            for observation in observations
            if observation.error_type in set(conditions.error_types)
        ]
        if not matched:
            return False
        if conditions.min_error_count is not None and sum(observation.count for observation in matched) < conditions.min_error_count:
            return False
        if conditions.min_severity is not None and not _has_min_severity(matched, conditions.min_severity):
            return False

    if conditions.health_problems:
        problems = set(dataset_report.dataset_health.problems) if dataset_report is not None else set()
        if not problems.intersection(conditions.health_problems):
            return False

    return True


def _merge_plan(
    scene: str,
    recommendations: list[OptimizationRecipeRecommendation],
) -> OptimizationRecipePlan:
    components = RecipeComponents()
    train_overrides: dict[str, Any] = {}
    postprocess: list[str] = []
    data_checks: list[str] = []
    expected_effect: list[str] = []
    risks: list[str] = []
    evidence_required: list[str] = []

    for recommendation in recommendations:
        components.bbox_loss.extend(recommendation.components.bbox_loss)
        components.head.extend(recommendation.components.head)
        components.assigner.extend(recommendation.components.assigner)
        components.neck.extend(recommendation.components.neck)
        components.augmentation.extend(recommendation.components.augmentation)
        train_overrides.update(recommendation.train_overrides)
        postprocess.extend(recommendation.postprocess)
        data_checks.extend(recommendation.data_checks)
        expected_effect.extend(recommendation.expected_effect)
        risks.extend(recommendation.risks)
        evidence_required.extend(recommendation.evidence_required)

    components.bbox_loss = dedupe_list(components.bbox_loss)
    components.head = dedupe_list(components.head)
    components.assigner = dedupe_list(components.assigner)
    components.neck = dedupe_list(components.neck)
    components.augmentation = dedupe_list(components.augmentation)
    return OptimizationRecipePlan(
        task_scene=scene,
        recommendations=recommendations,
        component_candidates=components,
        train_overrides=train_overrides,
        postprocess=dedupe_list(postprocess),
        data_checks=dedupe_list(data_checks),
        expected_effect=dedupe_list(expected_effect),
        risks=dedupe_list(risks),
        evidence_required=dedupe_list(evidence_required),
    )


def _rationale(
    recipe_id: str,
    recipe: OptimizationRecipeRule,
    task_spec: TaskSpec,
    observations: list[DetectionErrorObservation],
    dataset_report: DatasetReport | None,
) -> str:
    matched_errors = [
        observation.error_type
        for observation in observations
        if observation.error_type in set(recipe.conditions.error_types)
    ]
    parts = [f"recipe={recipe_id}", f"scene={task_spec.scene}"]
    if matched_errors:
        parts.append("errors=" + ",".join(dedupe_list(matched_errors)))
    if dataset_report is not None and recipe.conditions.health_problems:
        problems = set(dataset_report.dataset_health.problems).intersection(recipe.conditions.health_problems)
        if problems:
            parts.append("health_problems=" + ",".join(sorted(problems)))
    return "; ".join(parts)


def _has_min_severity(observations: list[DetectionErrorObservation], severity: str) -> bool:
    order = {"low": 0, "medium": 1, "high": 2}
    threshold = order.get(severity, 0)
    return any(order[observation.severity] >= threshold for observation in observations)
