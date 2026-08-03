import pytest

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.recipe_ablation_planner import RecipeAblationPlanner
from yolo_agent.recipes.coupled_library import (
    CouplingEvidence,
    EvidenceBoundCoupledRecipeLibrary,
)


def _evidence(components: list[str]) -> CouplingEvidence:
    return CouplingEvidence(
        evidence_kind="local_diagnosis",
        source_id="diagnosis-1",
        component_ids=components,
        reason="The observed causes require complementary mechanism A and mechanism B.",
        source_locations=["runs/one/diagnosis.yaml#finding"],
        error_fact_ids=["fact:one", "fact:two"],
        confidence=0.8,
        verified=True,
    )


@pytest.mark.parametrize(
    ("components", "template_id", "track"),
    [
        (["head.p2_small_object", "sampling.small_object"], "p2_small_object_sampling", "training"),
        (["neck.weighted_feature_pyramid", "loss.quality.correlation"], "feature_fusion_quality_loss", "training"),
        (["distillation.feature", "sampling.class_balanced"], "distillation_class_balanced_sampling", "training"),
        (["assigner.dynamic_topk", "loss.quality.pseudo_iou"], "assignment_quality_alignment", "training"),
        (["inference.sahi_slicing", "inference.confidence_calibration"], "slicing_confidence_calibration", "inference"),
    ],
)
def test_allowlisted_pairs_materialize_exact_four_arm_recipe(
    components: list[str], template_id: str, track: str
) -> None:
    result = EvidenceBoundCoupledRecipeLibrary().materialize(
        component_ids=components,
        evidence=_evidence(components),
    )

    assert result.decision == "materialized"
    assert result.template_id == template_id
    assert result.execution_track == track
    assert result.recipe is not None
    assert result.recipe.coupling_reason == _evidence(components).reason
    assert [item["name"] for item in result.recipe.internal_ablation_plan] == [
        "baseline",
        "A",
        "B",
        "A+B",
    ]
    assert [item["components"] for item in result.recipe.internal_ablation_plan] == [
        [],
        [result.component_ids[0]],
        [result.component_ids[1]],
        result.component_ids,
    ]
    assert result.recipe.train_overrides == {"imgsz": 640}


def test_inference_pair_never_changes_training_recipe() -> None:
    components = ["inference.sahi_slicing", "inference.confidence_calibration"]
    result = EvidenceBoundCoupledRecipeLibrary().materialize(
        component_ids=components,
        evidence=_evidence(components),
    )

    assert result.recipe is not None
    assert result.recipe.data_actions == []
    assert result.recipe.inference_actions == components
    assert result.recipe.fixed_variables["training_recipe"] == "unchanged"
    assert result.recipe.fixed_variables["checkpoint"] == "unchanged"


def test_unlisted_and_multi_component_bundles_are_rejected() -> None:
    library = EvidenceBoundCoupledRecipeLibrary()
    unsupported = ["head.p2_small_object", "loss.quality.correlation"]
    rejected = library.materialize(
        component_ids=unsupported,
        evidence=_evidence(unsupported),
    )
    bundled = [
        "head.p2_small_object",
        "sampling.small_object",
        "loss.quality.correlation",
    ]
    bundle_evidence = _evidence(bundled[:2]).model_copy(
        update={"component_ids": bundled}
    )
    bundle_result = library.materialize(
        component_ids=bundled,
        evidence=bundle_evidence,
    )

    assert rejected.decision == "rejected"
    assert "allowlisted_complementary_mechanism_pair_required" in rejected.blocked_by
    assert bundle_result.decision == "rejected"
    assert "exactly_two_components_required" in bundle_result.blocked_by


def test_evidence_must_bind_the_requested_pair() -> None:
    requested = ["head.p2_small_object", "sampling.small_object"]
    evidence = _evidence(
        ["neck.weighted_feature_pyramid", "loss.quality.correlation"]
    )

    result = EvidenceBoundCoupledRecipeLibrary().materialize(
        component_ids=requested,
        evidence=evidence,
    )

    assert result.decision == "rejected"
    assert "coupling_evidence_component_mismatch" in result.blocked_by


@pytest.mark.parametrize(
    "components",
    [
        ["head.p2_small_object", "sampling.small_object"],
        ["neck.weighted_feature_pyramid", "loss.quality.correlation"],
        ["distillation.feature", "sampling.class_balanced"],
        ["assigner.dynamic_topk", "loss.quality.pseudo_iou"],
    ],
)
def test_training_templates_feed_protected_four_arm_ablation(
    components: list[str],
) -> None:
    result = EvidenceBoundCoupledRecipeLibrary().materialize(
        component_ids=components,
        evidence=_evidence(components),
    )
    assert result.recipe is not None

    plan = RecipeAblationPlanner().plan(
        result.recipe,
        CandidateConfig(
            candidate_id="baseline",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            train_overrides={"imgsz": 640},
        ),
        max_nodes=4,
    )

    assert [item.role for item in plan.nodes] == [
        "baseline",
        "single",
        "single",
        "full",
    ]
    assert [item.component_ids for item in plan.nodes] == [
        [],
        [result.component_ids[0]],
        [result.component_ids[1]],
        result.component_ids,
    ]
    assert plan.successive_halving is not None
    assert all(
        item.decision == "run"
        for item in plan.successive_halving.assignments_for_stage("pilot_3")
    )
