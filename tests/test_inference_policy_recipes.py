from pathlib import Path

from yolo_agent.recipes.registry import RecipeRegistry


def test_inference_policy_recipes_never_change_training() -> None:
    registry = RecipeRegistry.from_path(
        Path("configs/recipes/isolated_inference_policies.yaml")
    )
    recipes = registry.list()

    assert len(recipes) == 5
    for recipe in recipes:
        assert recipe.kind == "atomic"
        assert recipe.train_overrides == {"imgsz": 640}
        assert recipe.fixed_variables["training_recipe"] == "unchanged"
        assert recipe.fixed_variables["checkpoint"] == "unchanged"
        assert len(recipe.component_ids) == 1
        assert recipe.component_ids[0].startswith("inference.")
        assert recipe.inference_actions
        assert not recipe.data_actions
        assert recipe.maturity == "adapter_implemented"
        assert recipe.is_executable is False
