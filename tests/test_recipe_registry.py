from pathlib import Path

import pytest
import yaml

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.recipes.schemas import AtomicRecipe, RecipeValidationError


def _recipe(recipe_id="r", version="v1.0.0", **updates):
    data = {"recipe_id": recipe_id, "version": version, "primary_changed_variable": "data_sampling", "fixed_variables": {"imgsz": 640}, "train_overrides": {"imgsz": 640}}
    data.update(updates)
    return AtomicRecipe.model_validate(data)


def test_registry_is_versioned_and_returns_latest() -> None:
    registry = RecipeRegistry([_recipe(version="v1.0.0"), _recipe(version="v1.2.0")])
    assert registry.get("r").version == "v1.2.0"
    assert registry.get("r", "v1.0.0").version == "v1.0.0"
    assert len(registry.list()) == 2


def test_registry_queries_by_metric_error_and_component() -> None:
    recipe = _recipe(target_metrics=["ap_small"], target_error_facts=[{"fact_type": "small_object"}], component_ids=["head.p2"])
    registry = RecipeRegistry([recipe])
    assert registry.query(target_metric="ap_small") == [recipe]
    assert registry.query(target_error_fact="small_object") == [recipe]
    assert registry.query(component_id="head.p2") == [recipe]


def test_registry_rejects_executable_recipe_with_metadata_component() -> None:
    component = ComponentContract(component_id="paper.only", display_name="Paper", category="head")
    recipe = _recipe(component_ids=[component.component_id], maturity="smoke_passed")
    with pytest.raises(RecipeValidationError):
        RecipeRegistry([recipe], [component])


def test_registry_loads_bundled_config() -> None:
    registry = RecipeRegistry.from_path(Path("configs/recipe_bundles.yaml"))
    assert registry.get("small_object_sampling") is not None
    assert registry.get("yolo26_small_object_pair") is not None
    assert registry.query(target_metric="ap_small")


def test_registry_loads_sahi_maturity_from_recipe_record() -> None:
    registry = RecipeRegistry.from_path(Path("configs/recipes/sahi_inference.yaml"))

    assert registry.get("sahi_slicing_inference").maturity == "adapter_implemented"


def test_registry_merges_sources_with_first_source_precedence(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.yaml"
    local = tmp_path / "local.yaml"
    frozen_recipe = _recipe(primary_changed_variable="frozen")
    local_recipe = _recipe(primary_changed_variable="local")
    frozen.write_text(
        yaml.safe_dump({"recipes": [frozen_recipe.model_dump()]}, sort_keys=False),
        encoding="utf-8",
    )
    local.write_text(
        yaml.safe_dump({"recipes": [local_recipe.model_dump()]}, sort_keys=False),
        encoding="utf-8",
    )

    registry = RecipeRegistry.from_paths([frozen, local])

    assert registry.get("r", "v1.0.0").primary_changed_variable == "frozen"
