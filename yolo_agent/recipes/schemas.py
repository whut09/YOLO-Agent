"""Versioned recipe bundle schemas for evidence-driven optimization."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import maturity_rank
from yolo_agent.core.yaml_io import YAMLModelMixin

RecipeMaturity = Literal[
    "metadata_only",
    "recipe_idea_only",
    "adapter_implemented",
    "runtime_integrated",
    "unit_tested",
    "smoke_passed",
    "gpu_certified",
    "pilot_reproduced",
    "full_reproduced",
    "confirmed_multi_seed",
]


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
    paper_ids: list[str] = Field(default_factory=list)
    method_profile_ids: list[str] = Field(default_factory=list)
    paper_specific_mechanism_id: str | None = None
    paper_specific_configuration: dict[str, Any] = Field(default_factory=dict)
    required_evidence: list[str] = Field(default_factory=list)
    mechanism_ids: list[str] = Field(default_factory=list)
    combination_fingerprint: str | None = None

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

    def expected_asha_trial_ids(self, base_run_id: str) -> dict[str, str]:
        """Return stable per-arm ASHA identities before a run is registered."""
        result: dict[str, str] = {}
        base = f"paper_recipe_{self.recipe_id}_{self.version.replace('.', '_')}"
        for item in self.internal_ablation_plan:
            if not isinstance(item, dict):
                continue
            arm_id = str(item.get("arm_id") or item.get("name") or "")
            if arm_id == "baseline" or not item.get("components"):
                continue
            legacy_name = {"arm_A": "A", "arm_B": "B", "arm_A_plus_B": "A+B"}.get(
                arm_id, arm_id
            )
            suffix = "".join(
                char.lower() if char.isalnum() else "_" for char in legacy_name
            ).strip("_")
            result[arm_id] = f"{base_run_id}:{base}__{suffix}"
        return result

    @model_validator(mode="after")
    def _coupled_contract(self) -> "CoupledRecipe":
        if len(self.coupled_variables) < 2:
            raise RecipeValidationError("CoupledRecipe requires at least two coupled_variables")
        if not self.coupling_reason or not self.coupling_source_papers or not self.internal_ablation_plan:
            raise RecipeValidationError("CoupledRecipe requires coupling_reason, source paper, and internal_ablation_plan")
        if self.primary_changed_variable not in self.coupled_variables:
            raise RecipeValidationError("primary_changed_variable must be one of coupled_variables")
        entries = [item for item in self.internal_ablation_plan if isinstance(item, dict)]
        legacy_names = {"A": "arm_A", "B": "arm_B", "A+B": "arm_A_plus_B"}
        legacy_component_names = {
            "small_object_sampling": "arm_A",
            "p2_head": "arm_B",
            "p2_head_plus_small_object_sampling": "arm_A_plus_B",
        }
        normalized: list[dict[str, Any]] = []
        for item in entries:
            item = dict(item)
            label = str(item.get("arm_id") or item.get("name") or item.get("variant") or "")
            item["arm_id"] = "baseline" if label == "baseline" else legacy_names.get(
                label, legacy_component_names.get(label, label)
            )
            if "components" not in item and item["arm_id"] in {
                "baseline",
                "arm_A",
                "arm_B",
                "arm_A_plus_B",
            }:
                item["components"] = {
                    "baseline": [],
                    "arm_A": self.component_ids[:1],
                    "arm_B": self.component_ids[1:2],
                    "arm_A_plus_B": list(self.component_ids),
                }[item["arm_id"]]
            if item["arm_id"] != "baseline":
                item["matched_control_arm_id"] = item.get("matched_control_arm_id", "baseline")
            else:
                item.pop("matched_control_arm_id", None)
            normalized.append(item)
        self.internal_ablation_plan = normalized
        arm_ids = [str(item.get("arm_id") or "") for item in normalized]
        required_arm_ids = {"baseline", "arm_A", "arm_B", "arm_A_plus_B"}
        legacy_generated_matrix = len(normalized) == 1 and arm_ids[0] in {
            "single_and_full",
            "baseline_singles_full",
        }
        strict_coupled_matrix = bool(
            self.combination_fingerprint
            or self.required_evidence
            or required_arm_ids.issubset(set(arm_ids))
        )
        if (
            strict_coupled_matrix
            and set(arm_ids) != required_arm_ids
            and not legacy_generated_matrix
        ):
            raise RecipeValidationError(
                "CoupledRecipe requires baseline, arm_A, arm_B, and arm_A_plus_B"
            )
        if strict_coupled_matrix and not legacy_generated_matrix:
            baseline = next(item for item in normalized if item.get("arm_id") == "baseline")
            if baseline.get("matched_control_arm_id") is not None:
                raise RecipeValidationError("baseline arm cannot reference a matched control")
            for arm_id in ("arm_A", "arm_B", "arm_A_plus_B"):
                arm = next(item for item in normalized if item.get("arm_id") == arm_id)
                if arm.get("matched_control_arm_id") != "baseline":
                    raise RecipeValidationError(f"{arm_id} requires matched_control_arm_id=baseline")
        if not self.required_evidence and strict_coupled_matrix:
            self.required_evidence = ["verified_coupling_evidence"]
        if not self.combination_fingerprint:
            payload = {
                "recipe_id": self.recipe_id,
                "version": self.version,
                "components": self.component_ids,
                "variables": self.coupled_variables,
                "papers": self.coupling_source_papers,
                "configuration": self.paper_specific_configuration,
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.combination_fingerprint = hashlib.sha256(encoded).hexdigest()
        return self


def recipe_from_mapping(data: dict[str, Any]) -> RecipeSpec:
    return (CoupledRecipe if data.get("kind", "atomic") == "coupled" else AtomicRecipe).model_validate(data)


__all__ = ["AtomicRecipe", "CoupledRecipe", "RecipeMaturity", "RecipeSpec", "RecipeValidationError", "recipe_from_mapping"]
