"""Contracts, AtomicRecipe boundaries, and catalog alias tests for neck plugins."""

from __future__ import annotations

from tests.neck_fixtures import neck_contracts, neck_recipes
from yolo_agent.research.component_aliases import ComponentAliasResolver


def test_neck_contracts_are_executable_only_with_explicit_runtime_adapters() -> None:
    contracts = neck_contracts()

    assert set(contracts) == {
        "neck.multi_scale_fusion",
        "neck.gold_gather_distribute",
        "neck.rtmdet_large_kernel",
    }
    for contract in contracts.values():
        assert contract.maturity == "smoke_passed"
        assert contract.can_execute
        assert contract.adapter_class
        assert contract.implementation_path
        assert contract.tensor_input_contract["strides"] == [8, 16, 32]
        assert contract.tensor_output_contract["channels"] == "unchanged"


def test_each_neck_is_an_independent_atomic_recipe_with_four_hard_guards() -> None:
    recipes = neck_recipes()

    assert len(recipes) == 3
    for recipe in recipes:
        assert recipe.kind == "atomic"
        assert len(recipe.component_ids) == 1
        assert recipe.primary_changed_variable == "model_graph.neck_plugin"
        assert recipe.fixed_variables["imgsz"] == 640
        assert recipe.train_overrides["imgsz"] == 640
        guard_text = " ".join(
            [
                *recipe.stop_conditions,
                *recipe.promotion_requirements,
                *recipe.training_cost,
                *recipe.inference_cost,
            ]
        )
        for guard in ("latency", "vram", "parameter", "model_size"):
            assert guard in guard_text


def test_awesome_neck_aliases_resolve_to_runtime_components() -> None:
    resolver = ComponentAliasResolver.from_yaml()
    expected = {
        "multi_scale_fusion": "neck.multi_scale_fusion",
        "gather_distribute_neck": "neck.gold_gather_distribute",
        "large_kernel_depthwise_conv": "neck.rtmdet_large_kernel",
    }

    for paper_component, component_id in expected.items():
        mapping = resolver.resolve(paper_component).mappings[0]
        assert mapping.canonical_component_id == component_id
        assert mapping.adapter_verified is True
        assert mapping.maturity == "smoke_passed"
        assert mapping.executable is True


def test_paper_priors_do_not_claim_exact_detector_reproduction() -> None:
    by_id = {recipe.recipe_id: recipe for recipe in neck_recipes()}
    gold = by_id["yolo26_gold_gather_distribute_neck"].evidence_prior[0]
    rtmdet = by_id["yolo26_rtmdet_large_kernel_neck"].evidence_prior[0]

    assert gold["evidence_level"] == "paper_prior"
    assert gold["local_evidence"] is False
    assert gold["adaptation"] == "isolated_gather_distribute_only"
    assert rtmdet["evidence_level"] == "paper_prior"
    assert rtmdet["local_evidence"] is False
    assert rtmdet["adaptation"] == "isolated_large_kernel_depthwise_neck_only"
