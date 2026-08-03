import pytest

from yolo_agent.agents.coupled_inference_plan import (
    CoupledInferenceAblationPlan,
    CoupledInferenceArm,
)


def _arms() -> list[CoupledInferenceArm]:
    baseline = CoupledInferenceArm(
        arm_id="recipe:baseline",
        role="baseline",
        component_ids=[],
        metric_namespace="standard_640",
        inference_policy_changed=False,
    )
    return [
        baseline,
        CoupledInferenceArm(
            arm_id="recipe:A",
            role="A",
            component_ids=["inference.sahi_slicing"],
            changed_variables={"inference.slicing_policy": "sahi"},
            metric_namespace="recipe_A",
            matched_control_arm_id=baseline.arm_id,
            inference_policy_changed=True,
        ),
        CoupledInferenceArm(
            arm_id="recipe:B",
            role="B",
            component_ids=["inference.confidence_calibration"],
            changed_variables={"inference.confidence_calibration": "temperature"},
            metric_namespace="recipe_B",
            matched_control_arm_id=baseline.arm_id,
            inference_policy_changed=True,
        ),
        CoupledInferenceArm(
            arm_id="recipe:A+B",
            role="A+B",
            component_ids=[
                "inference.sahi_slicing",
                "inference.confidence_calibration",
            ],
            changed_variables={
                "inference.slicing_policy": "sahi",
                "inference.confidence_calibration": "temperature",
            },
            metric_namespace="recipe_A_plus_B",
            matched_control_arm_id=baseline.arm_id,
            inference_policy_changed=True,
        ),
    ]


def test_inference_plan_requires_exact_matched_four_arm_cohort() -> None:
    arms = _arms()
    plan = CoupledInferenceAblationPlan(
        recipe_id="recipe",
        coupling_reason="Complementary inference policies.",
        arms=arms,
        required_metrics=["sliced_calibrated_ap_small"],
        minimum_internal_ablation_arm_ids=[item.arm_id for item in arms],
    )

    assert plan.asha_training_budget_allowed is False
    assert all(item.training_attribution_allowed is False for item in plan.arms)

    with pytest.raises(ValueError, match="standard baseline"):
        CoupledInferenceAblationPlan(
            recipe_id="recipe",
            coupling_reason="Complementary inference policies.",
            arms=[
                arms[0],
                arms[1].model_copy(update={"matched_control_arm_id": "other"}),
                *arms[2:],
            ],
            required_metrics=["sliced_calibrated_ap_small"],
            minimum_internal_ablation_arm_ids=[item.arm_id for item in arms],
        )
