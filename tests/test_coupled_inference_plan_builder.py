import pytest

from yolo_agent.agents.coupled_inference_plan import CoupledInferencePlanBuilder
from yolo_agent.recipes.coupled_library import (
    CouplingEvidence,
    EvidenceBoundCoupledRecipeLibrary,
)


def _inference_recipe():
    components = ["inference.sahi_slicing", "inference.confidence_calibration"]
    result = EvidenceBoundCoupledRecipeLibrary().materialize(
        component_ids=components,
        evidence=CouplingEvidence(
            evidence_kind="local_diagnosis",
            source_id="diagnosis-inference",
            component_ids=components,
            reason="Slicing increases small-object exposure and calibration controls scores.",
            source_locations=["runs/one/diagnosis.yaml#inference"],
            error_fact_ids=["small_object_fn", "confidence_miscalibration"],
            verified=True,
        ),
    )
    assert result.recipe is not None
    return result.recipe


def test_builds_isolated_inference_baseline_a_b_and_combined() -> None:
    plan = CoupledInferencePlanBuilder().build(_inference_recipe())

    assert [item.role for item in plan.arms] == ["baseline", "A", "B", "A+B"]
    assert [item.component_ids for item in plan.arms] == [
        [],
        ["inference.sahi_slicing"],
        ["inference.confidence_calibration"],
        ["inference.sahi_slicing", "inference.confidence_calibration"],
    ]
    assert plan.arms[0].metric_namespace == "standard_640"
    assert len({item.metric_namespace for item in plan.arms}) == 4
    assert all(
        item.matched_control_arm_id == plan.arms[0].arm_id
        for item in plan.arms[1:]
    )


def test_inference_builder_rejects_training_attribution() -> None:
    recipe = _inference_recipe().model_copy(
        update={"training_cost": {"training_changed": True}}
    )

    with pytest.raises(ValueError, match="training_changed=false"):
        CoupledInferencePlanBuilder().build(recipe)
