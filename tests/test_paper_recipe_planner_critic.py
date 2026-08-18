"""Planner and critic fixtures for paper-specific recipes. No GPU training."""

from __future__ import annotations

from yolo_agent.agents.recipe_critic import RecipeCritic
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.recipes.paper_recipe_bindings import load_certified_paper_recipe_specs
from yolo_agent.recipes.paper_recipe_guards import (
    generic_collapse_reasons,
    inference_train_reasons,
    is_overall_map_objective,
    overall_map_small_object_reasons,
)
from yolo_agent.recipes.schemas import AtomicRecipe
from yolo_agent.research.paper_protocol_ids import CERTIFIED_PAPER_MECHANISMS


def _fact(fact_type: str = "representation_gap") -> ErrorFact:
    return ErrorFact(
        run_id="run-1",
        candidate_id="c1",
        node_id="n1",
        fact_type=fact_type,  # type: ignore[arg-type]
        subject="all",
    )


def _atomic(**overrides: object) -> AtomicRecipe:
    payload = {
        "recipe_id": "probe",
        "version": "v1.0.0",
        "target_error_facts": [{"fact_type": "representation_gap"}],
        "target_metrics": ["map50_95"],
        "component_ids": ["distillation.yolo26_teacher_student"],
        "primary_changed_variable": "distillation",
        "paper_ids": ["cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection", "cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection"],
    }
    payload.update(overrides)
    return AtomicRecipe.model_validate(payload)


def test_generic_distillation_recipe_cannot_cover_many_papers() -> None:
    recipe = _atomic(recipe_id="yolo26n_distillation")
    reasons = generic_collapse_reasons(recipe, recipe.paper_ids)
    assert "generic_distillation_recipe_cannot_cover_multiple_papers" in reasons
    report = RecipeCritic().critique(
        recipe,
        error_facts=[_fact()],
        component_contracts=[],
        compatibility={},
    )
    assert report.accepted is False
    assert "generic_distillation_recipe_cannot_cover_multiple_papers" in report.blocked_by


def test_empty_target_facts_are_evidence_recovery_not_queued() -> None:
    recipe = _atomic(target_error_facts=[])
    report = RecipeCritic().critique(
        recipe,
        error_facts=[],
        component_contracts=[],
        compatibility={},
    )
    assert "target_error_facts_missing" in report.blocked_by
    assert report.accepted is False


def test_inference_only_recipe_is_not_a_train_candidate() -> None:
    recipe = _atomic(
        recipe_id="sahi_slicing_inference",
        component_ids=["inference.sahi_slicing"],
        inference_actions=["sahi"],
        paper_ids=["arxiv:2202.06934"],
    )
    assert inference_train_reasons(recipe) == ["inference_only_not_training_candidate"]
    report = RecipeCritic().critique(
        recipe,
        error_facts=[_fact()],
        component_contracts=[],
        compatibility={},
    )
    assert "inference_only_not_training_candidate" in report.blocked_by


def test_overall_map_goal_rejects_small_object_only_recipe() -> None:
    recipe = _atomic(
        recipe_id="yolo26_small_object",
        component_ids=["sampling.small_object"],
        target_metrics=["ap_small"],
        paper_ids=[],
    )
    reasons = overall_map_small_object_reasons(recipe, overall_map_goal=True)
    assert reasons == ["small_object_method_out_of_scope_for_overall_map"]


def test_every_certified_paper_is_traceable_to_a_spec() -> None:
    specs = {paper_id: spec for spec in load_certified_paper_recipe_specs() for paper_id in spec.paper_ids}
    for paper_id in CERTIFIED_PAPER_MECHANISMS:
        spec = specs[paper_id]
        assert spec.recipe_id
        assert spec.paper_specific_mechanism_id
        assert spec.protocol_hash
        assert spec.execution_fingerprint
        assert paper_id in spec.paper_ids


class _Objective:
    primary_metric = "map50_95"
    goal_description = "Improve overall mAP"


def test_overall_map_objective_helper() -> None:
    assert is_overall_map_objective(_Objective()) is True
