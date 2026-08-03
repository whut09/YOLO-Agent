"""Isolated four-arm plans for evidence-bound coupled inference policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.recipes.schemas import CoupledRecipe


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

    def to_yaml(self, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CoupledInferenceAblationPlan":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        return cls.model_validate(payload)


class CoupledInferencePlanBuilder:
    """Project an inference-only coupled recipe into an isolated four-arm plan."""

    def build(self, recipe: CoupledRecipe) -> CoupledInferenceAblationPlan:
        if not recipe.inference_actions or any(
            not item.startswith("inference.") for item in recipe.component_ids
        ):
            raise ValueError("coupled inference plan requires inference-only components")
        if recipe.training_cost.get("training_changed") is not False:
            raise ValueError("coupled inference plan requires training_changed=false")
        if recipe.fixed_variables.get("training_recipe") != "unchanged":
            raise ValueError("coupled inference plan requires unchanged training recipe")
        if recipe.fixed_variables.get("checkpoint") != "unchanged":
            raise ValueError("coupled inference plan requires unchanged checkpoint")

        entries = {
            str(item.get("name")): item
            for item in recipe.internal_ablation_plan
            if isinstance(item, dict)
        }
        if set(entries) != {"baseline", "A", "B", "A+B"}:
            raise ValueError("coupled inference recipe requires baseline/A/B/A+B")
        baseline_id = f"{recipe.recipe_id}:baseline"
        arms = [
            self._arm(
                recipe,
                role=role,
                entry=entries[role],
                baseline_id=baseline_id,
            )
            for role in ("baseline", "A", "B", "A+B")
        ]
        return CoupledInferenceAblationPlan(
            recipe_id=recipe.recipe_id,
            coupling_reason=recipe.coupling_reason or "",
            arms=arms,
            required_metrics=list(recipe.target_metrics),
            minimum_internal_ablation_arm_ids=[item.arm_id for item in arms],
        )

    @staticmethod
    def _arm(
        recipe: CoupledRecipe,
        *,
        role: InferenceAblationRole,
        entry: dict[str, Any],
        baseline_id: str,
    ) -> CoupledInferenceArm:
        components = entry.get("components", [])
        changed = entry.get("changed_variables", {})
        if not isinstance(components, list) or not isinstance(changed, dict):
            raise ValueError(f"invalid coupled inference arm: {role}")
        return CoupledInferenceArm(
            arm_id=f"{recipe.recipe_id}:{role}",
            role=role,
            component_ids=[str(item) for item in components],
            changed_variables=dict(changed),
            metric_namespace=(
                "standard_640"
                if role == "baseline"
                else _metric_namespace(recipe.recipe_id, role)
            ),
            matched_control_arm_id=None if role == "baseline" else baseline_id,
            inference_policy_changed=role != "baseline",
        )


def _metric_namespace(recipe_id: str, role: InferenceAblationRole) -> str:
    stable_recipe = recipe_id.replace(".", "_").replace("-", "_")
    stable_role = role.replace("+", "_plus_")
    return f"{stable_recipe}_{stable_role}"


__all__ = [
    "CoupledInferenceAblationPlan",
    "CoupledInferenceArm",
    "CoupledInferencePlanBuilder",
    "InferenceAblationRole",
]
