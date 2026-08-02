from __future__ import annotations

from yolo_agent.components.contracts import load_contracts
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.recipes.schemas import AtomicRecipe


def test_data_pipeline_recipes_are_independent_and_conservative() -> None:
    contracts = load_contracts(
        "configs/components/data_pipeline/paper_data_adapters.yaml"
    )
    registry = RecipeRegistry.from_path(
        "configs/recipes/yolo26_data_pipeline.yaml",
        component_contracts=contracts,
    )
    recipes = registry.list()

    assert len(recipes) == 9
    assert all(isinstance(item, AtomicRecipe) for item in recipes)
    assert all(item.maturity == "adapter_implemented" for item in recipes)
    assert all(not item.is_executable for item in recipes)
    assert len({item.primary_changed_variable for item in recipes}) == 9
    for recipe in recipes:
        assert recipe.primary_changed_variable.startswith("data.")
        assert recipe.primary_changed_variable != "data.sampling_policy"
        assert recipe.train_overrides["imgsz"] == 640
        assert recipe.fixed_variables["imgsz"] == 640
        assert recipe.fixed_variables["val_split"] == "unchanged"
        assert recipe.fixed_variables["test_split"] == "unchanged"
        assert len(recipe.component_ids) == 1
        assert recipe.evidence_prior[0]["evidence_level"] == "paper_claim"
