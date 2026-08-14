from __future__ import annotations

from yolo_agent.agents.candidate_generator import (
    CandidateConfig,
    CandidateEvaluationContract,
)
from yolo_agent.agents.paper_recipe_planner import _evaluation_contract
from yolo_agent.recipes.registry import RecipeRegistry


QUALITY_RECIPE_IDS = {
    "yolo26_correlation_auxiliary_loss",
    "yolo26_pseudo_iou_quality_auxiliary_loss",
}


def test_quality_recipes_preserve_primary_localization_and_resource_guards() -> None:
    registry = RecipeRegistry.from_paths(
        ["configs/recipes/yolo26_quality_alignment.yaml"], strict=False
    )
    recipes = [recipe for recipe in registry.list() if recipe.recipe_id in QUALITY_RECIPE_IDS]

    assert {recipe.recipe_id for recipe in recipes} == QUALITY_RECIPE_IDS
    for recipe in recipes:
        contract = _evaluation_contract(recipe)
        assert contract.primary_metric == "map50_95"
        assert "map50_95" in contract.evaluation_metrics
        assert any(
            metric in contract.evaluation_metrics
            for metric in {"ap75", "confidence_iou_correlation"}
        )
        assert contract.latency_metric == "latency_ms"
        assert contract.model_size_metric == "model_size_mb"
        assert "latency_guard" in " ".join(contract.promotion_requirements)
        assert "model_size_guard" in " ".join(contract.promotion_requirements)


def test_candidate_evaluation_contract_is_backward_compatible_and_deduplicated() -> None:
    legacy = CandidateConfig(
        candidate_id="legacy",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
    )
    assert legacy.evaluation_contract.evaluation_metrics == [
        "map50_95",
        "latency_ms",
        "model_size_mb",
    ]

    contract = CandidateEvaluationContract(
        primary_metric="map50_95",
        evaluation_metrics=["map50_95", "ap75", "ap75"],
    )
    assert contract.evaluation_metrics == [
        "map50_95",
        "ap75",
        "latency_ms",
        "model_size_mb",
    ]
