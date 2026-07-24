import pytest

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.recipes.schemas import AtomicRecipe, CoupledRecipe, recipe_from_mapping


def _base(**updates):
    data = {"recipe_id": "test", "version": "v1.0.0", "primary_changed_variable": "data_sampling", "fixed_variables": {"imgsz": 640}, "train_overrides": {"imgsz": 640}}
    data.update(updates)
    return data


def test_atomic_recipe_requires_fixed_imgsz() -> None:
    recipe = AtomicRecipe.model_validate(_base())
    assert recipe.kind == "atomic" and recipe.fixed_variables["imgsz"] == 640
    assert not recipe.is_executable


def test_recipe_rejects_imgsz_change() -> None:
    with pytest.raises(ValueError):
        AtomicRecipe.model_validate(_base(train_overrides={"imgsz": 1280}))


def test_atomic_rejects_coupling_fields() -> None:
    with pytest.raises(ValueError):
        AtomicRecipe.model_validate(_base(coupled_variables=["head", "loss"]))


def test_coupled_requires_reason_paper_and_ablation_plan() -> None:
    with pytest.raises(ValueError):
        CoupledRecipe.model_validate(_base(kind="coupled", primary_changed_variable="head", coupled_variables=["head", "loss"]))
    recipe = CoupledRecipe.model_validate(_base(kind="coupled", primary_changed_variable="head", coupled_variables=["head", "loss"], coupling_reason="Jointly required.", coupling_source_papers=["paper:1"], internal_ablation_plan=[{"variables": ["head"]}]))
    assert recipe.kind == "coupled"


def test_paper_claim_is_only_evidence_prior() -> None:
    recipe = AtomicRecipe.model_validate(_base(evidence_prior=[{"evidence_level": "paper_claim", "value": 2.0}]))
    assert recipe.evidence_prior[0]["evidence_level"] == "paper_claim"


def test_executable_recipe_rejects_metadata_component() -> None:
    recipe = AtomicRecipe.model_validate(_base(component_ids=["paper.only"], maturity="smoke_passed"))
    contract = ComponentContract(component_id="paper.only", display_name="Paper", category="head")
    with pytest.raises(ValueError):
        recipe.validate_components({contract.component_id: contract})


def test_executable_recipe_accepts_smoke_component() -> None:
    recipe = AtomicRecipe.model_validate(_base(component_ids=["implemented"], maturity="smoke_passed"))
    contract = ComponentContract(component_id="implemented", display_name="Implemented", category="head", implementation_path="x", adapter_class="A", maturity="smoke_passed")
    recipe.validate_components({contract.component_id: contract})


def test_recipe_factory_selects_kind() -> None:
    assert isinstance(recipe_from_mapping(_base()), AtomicRecipe)
    assert isinstance(recipe_from_mapping(_base(kind="coupled", primary_changed_variable="head", coupled_variables=["head", "loss"], coupling_reason="reason", coupling_source_papers=["paper"], internal_ablation_plan=[{"variables": ["head"]}])), CoupledRecipe)
