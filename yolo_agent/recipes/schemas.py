"""Versioned recipe bundle schemas for evidence-driven optimization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import maturity_rank
from yolo_agent.core.yaml_io import YAMLModelMixin

RecipeMaturity = Literal["metadata_only", "reference_code_available", "adapter_implemented", "unit_tested", "smoke_passed", "pilot_reproduced", "full_reproduced", "production_eligible"]


class RecipeValidationError(ValueError):
    """Raised when a recipe violates its atomic/coupled contract."""


class RecipeSpec(BaseModel, YAMLModelMixin):
    """Common fields shared by atomic and coupled recipes."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "recipe.v1"
    recipe_id: str
    version: str
    target_error_facts: list[dict[str, Any]] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)
    component_ids: list[str] = Field(default_factory=list)
    train_overrides: dict[str, Any] = Field(default_factory=lambda: {"imgsz": 640})
    data_actions: list[str] = Field(default_factory=list)
    inference_actions: list[str] = Field(default_factory=list)
    fixed_variables: dict[str, Any] = Field(default_factory=lambda: {"imgsz": 640})
    primary_changed_variable: str
    coupled_variables: list[str] = Field(default_factory=list)
    coupling_reason: str | None = None
    coupling_source_papers: list[str] = Field(default_factory=list)
    internal_ablation_plan: list[dict[str, Any]] = Field(default_factory=list)
    compatibility_requirements: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    expected_effects: dict[str, Any] = Field(default_factory=dict)
    evidence_prior: list[dict[str, Any]] = Field(default_factory=list)
    implementation_risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    training_cost: dict[str, Any] = Field(default_factory=dict)
    inference_cost: dict[str, Any] = Field(default_factory=dict)
    stop_conditions: list[str] = Field(default_factory=list)
    promotion_requirements: list[str] = Field(default_factory=list)
    maturity: RecipeMaturity = "metadata_only"

    @field_validator("recipe_id", "version", "primary_changed_variable")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("recipe_id, version, and primary_changed_variable must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def _fixed_imgsz(self) -> "RecipeSpec":
        if self.fixed_variables.get("imgsz") != 640 or self.train_overrides.get("imgsz", 640) != 640:
            raise RecipeValidationError("Recipe input size is fixed at imgsz=640")
        if "imgsz" not in self.fixed_variables:
            raise RecipeValidationError("Recipe must declare fixed_variables.imgsz=640")
        return self

    @property
    def is_executable(self) -> bool:
        return maturity_rank(self.maturity) >= maturity_rank("smoke_passed")

    def validate_components(self, contracts: dict[str, ComponentContract]) -> None:
        missing = sorted(set(self.component_ids) - set(contracts))
        if missing:
            raise RecipeValidationError(f"Recipe references unknown components: {', '.join(missing)}")
        blocked = [component_id for component_id in self.component_ids if not contracts[component_id].can_execute]
        if self.is_executable and blocked:
            raise RecipeValidationError("Executable recipe contains non-executable components: " + ", ".join(sorted(blocked)))


class AtomicRecipe(RecipeSpec):
    """Recipe with exactly one primary changed variable."""

    kind: Literal["atomic"] = "atomic"

    @model_validator(mode="after")
    def _atomic_contract(self) -> "AtomicRecipe":
        if self.coupled_variables or self.coupling_reason or self.coupling_source_papers or self.internal_ablation_plan:
            raise RecipeValidationError("AtomicRecipe cannot declare coupled-recipe fields")
        if self.primary_changed_variable in {"imgsz", "image_size"}:
            raise RecipeValidationError("imgsz is fixed at 640 and cannot be changed")
        return self


class CoupledRecipe(RecipeSpec):
    """Recipe whose variables must change together for a documented reason."""

    kind: Literal["coupled"] = "coupled"

    @model_validator(mode="after")
    def _coupled_contract(self) -> "CoupledRecipe":
        if len(self.coupled_variables) < 2:
            raise RecipeValidationError("CoupledRecipe requires at least two coupled_variables")
        if not self.coupling_reason or not self.coupling_source_papers or not self.internal_ablation_plan:
            raise RecipeValidationError("CoupledRecipe requires coupling_reason, source paper, and internal_ablation_plan")
        if self.primary_changed_variable not in self.coupled_variables:
            raise RecipeValidationError("primary_changed_variable must be one of coupled_variables")
        return self


def recipe_from_mapping(data: dict[str, Any]) -> RecipeSpec:
    return (CoupledRecipe if data.get("kind", "atomic") == "coupled" else AtomicRecipe).model_validate(data)


__all__ = ["AtomicRecipe", "CoupledRecipe", "RecipeMaturity", "RecipeSpec", "RecipeValidationError", "recipe_from_mapping"]
