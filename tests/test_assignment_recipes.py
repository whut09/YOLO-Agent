"""Assignment component contract and recipe tests."""

from __future__ import annotations

from tests.assignment_fixtures import assignment_recipes
from yolo_agent.components.contracts import load_contracts
from yolo_agent.recipes.schemas import AtomicRecipe


def test_assignment_contracts_and_recipes_are_shadow_first_and_atomic() -> None:
    contracts = load_contracts("configs/components/assigner/yolo26_assignment.yaml")
    recipes = assignment_recipes()

    assert len(contracts) == len(recipes) == 3
    assert all(item.can_execute and item.maturity == "smoke_passed" for item in contracts)
    assert all(isinstance(item, AtomicRecipe) for item in recipes)
    assert all(item.maturity == "smoke_passed" for item in recipes)
    assert all(item.train_overrides[item.primary_changed_variable] == "shadow" for item in recipes)
    assert all(item.fixed_variables["assignment_path"] == "one_to_many" for item in recipes)
    assert all(item.fixed_variables["one_to_one_path"] == "native" for item in recipes)
    assert all(item.fixed_variables["bbox_regression"] == "native_dfl_free" for item in recipes)
    for recipe in recipes:
        assert all(item["evidence_level"] == "paper_prior" for item in recipe.evidence_prior)
        assert all(item["reported_delta"] == {} for item in recipe.evidence_prior)
