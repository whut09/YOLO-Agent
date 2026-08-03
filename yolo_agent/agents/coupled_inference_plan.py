"""Isolated four-arm plans for evidence-bound coupled inference policies."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


InferenceAblationRole = Literal["baseline", "A", "B", "A+B"]


class CoupledInferenceArm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm_id: str
    role: InferenceAblationRole
    component_ids: list[str]
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    metric_namespace: str
    matched_control_arm_id: str | None = None
    standard_imgsz: Literal[640] = 640
    inference_policy_changed: bool
    training_attribution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _baseline_and_candidate_identity(self) -> "CoupledInferenceArm":
        if self.role == "baseline":
            if self.component_ids or self.changed_variables:
                raise ValueError("inference baseline cannot change policy components")
            if self.matched_control_arm_id is not None:
                raise ValueError("inference baseline cannot have a matched control")
            if self.inference_policy_changed:
                raise ValueError("standard inference baseline cannot mark policy changed")
        else:
            if not self.component_ids or not self.changed_variables:
                raise ValueError("inference candidate arm requires policy changes")
            if not self.matched_control_arm_id:
                raise ValueError("inference candidate arm requires matched control")
            if not self.inference_policy_changed:
                raise ValueError("inference candidate must mark policy changed")
        return self


class CoupledInferenceAblationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "coupled_inference_ablation.v1"
    recipe_id: str
    coupling_reason: str
    arms: list[CoupledInferenceArm] = Field(min_length=4, max_length=4)
    required_metrics: list[str] = Field(min_length=1)
    minimum_internal_ablation_arm_ids: list[str] = Field(min_length=4, max_length=4)
    training_recipe: Literal["unchanged"] = "unchanged"
    checkpoint: Literal["unchanged"] = "unchanged"
    asha_training_budget_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _exact_four_arm_contract(self) -> "CoupledInferenceAblationPlan":
        roles = [item.role for item in self.arms]
        if roles != ["baseline", "A", "B", "A+B"]:
            raise ValueError("coupled inference plan requires baseline/A/B/A+B order")
        baseline_id = self.arms[0].arm_id
        if any(
            item.matched_control_arm_id != baseline_id for item in self.arms[1:]
        ):
            raise ValueError("every inference candidate must use the standard baseline")
        arm_ids = [item.arm_id for item in self.arms]
        if self.minimum_internal_ablation_arm_ids != arm_ids:
            raise ValueError("minimum inference cohort must include all four arms")
        return self


__all__ = [
    "CoupledInferenceAblationPlan",
    "CoupledInferenceArm",
    "InferenceAblationRole",
]
