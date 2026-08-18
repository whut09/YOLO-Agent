"""Planner and critic guards for paper-specific recipes."""

from __future__ import annotations

from typing import Any, Iterable

from yolo_agent.recipes.paper_recipe_bindings import bindings_by_paper_id
from yolo_agent.recipes.schemas import RecipeSpec
from yolo_agent.research.paper_mechanism_resolver import GENERIC_MECHANISM_IDS


SMALL_OBJECT_MARKERS = (
    "small_object",
    "head.p2_small_object",
    "sampling.small_object",
    "ap_small",
)


def recipe_is_inference_only(recipe: RecipeSpec) -> bool:
    if recipe.component_ids:
        return all(str(item).startswith("inference.") for item in recipe.component_ids)
    return bool(recipe.inference_actions)


def recipe_is_small_object_only(recipe: RecipeSpec) -> bool:
    haystack = " ".join(
        [
            recipe.recipe_id,
            " ".join(recipe.component_ids),
            " ".join(recipe.target_metrics),
        ]
    ).lower()
    return any(marker in haystack for marker in SMALL_OBJECT_MARKERS)


def recipe_uses_only_generic_mechanisms(recipe: RecipeSpec) -> bool:
    components = set(recipe.component_ids)
    return bool(components) and components <= GENERIC_MECHANISM_IDS


def generic_collapse_reasons(recipe: RecipeSpec, related_papers: Iterable[str] = ()) -> list[str]:
    """Reject one generic recipe standing in for many papers."""
    papers = [item for item in related_papers if item]
    if recipe.recipe_id == "yolo26n_distillation" and len(papers) > 1:
        return ["generic_distillation_recipe_cannot_cover_multiple_papers"]
    if "domain_adaptation.general" in recipe.component_ids and len(papers) > 1:
        return ["generic_domain_adaptation_recipe_cannot_cover_multiple_papers"]
    if recipe_uses_only_generic_mechanisms(recipe) and len(papers) > 1:
        return ["generic_mechanism_cannot_cover_multiple_papers"]
    return []


def empty_error_fact_reasons(recipe: RecipeSpec, facts: Iterable[Any]) -> list[str]:
    if list(facts) and getattr(recipe, "target_error_facts", None):
        return []
    if not recipe.target_error_facts:
        return ["target_error_facts_missing", "evidence_recovery"]
    return []


def overall_map_small_object_reasons(recipe: RecipeSpec, *, overall_map_goal: bool) -> list[str]:
    if overall_map_goal and recipe_is_small_object_only(recipe):
        return ["small_object_method_out_of_scope_for_overall_map"]
    return []


def inference_train_reasons(recipe: RecipeSpec) -> list[str]:
    if recipe_is_inference_only(recipe):
        return ["inference_only_not_training_candidate"]
    return []


def is_overall_map_objective(objective: Any | None) -> bool:
    if objective is None:
        return False
    metric = str(
        getattr(objective, "primary_metric", None)
        or getattr(objective, "target_metric", None)
        or ""
    )
    text = " ".join(
        [
            str(getattr(objective, "goal_description", "") or ""),
            str(getattr(objective, "goal_expression", "") or ""),
            str(getattr(objective, "goal", "") or ""),
        ]
    ).lower()
    return metric in {"map50_95", "map", "mAP"} and ("overall" in text or "整体" in text)


def certified_binding_for_papers(paper_ids: Iterable[str]) -> list[str]:
    """Return recipe IDs already bound to certified papers."""
    index = bindings_by_paper_id()
    return sorted({index[paper_id].recipe_id for paper_id in paper_ids if paper_id in index})
