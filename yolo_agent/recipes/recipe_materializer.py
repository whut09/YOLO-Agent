"""Materialize paper priors through local maturity and compatibility gates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import maturity_rank
from yolo_agent.recipes.paper_priors import RecipePrior
from yolo_agent.recipes.schemas import AtomicRecipe, CoupledRecipe, RecipeSpec


MaterializationStatus = Literal[
    "rejected",
    "implementation_proposal",
    "implementation_request",
    "dry_run_recipe",
    "pilot_recipe",
    "prioritized_pilot_recipe",
    "full_candidate_recommendation",
]


class RecipeImplementationAction(BaseModel):
    """A non-training action required before a paper prior can mature."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal["implementation_proposal", "implementation_request"]
    component_ids: list[str]
    required_adapters: list[str] = Field(default_factory=list)
    reason: str


class RecipeMaterialization(BaseModel):
    """Guarded result; recipe presence never implies a generated command."""

    model_config = ConfigDict(extra="forbid")
    prior_id: str
    status: MaterializationStatus
    recipe: AtomicRecipe | CoupledRecipe | None = None
    implementation_action: RecipeImplementationAction | None = None
    reasons: list[str] = Field(default_factory=list)
    recommendation_priority: Literal["none", "normal", "high"] = "none"
    allowed_stage: Literal["none", "dry_run", "pilot", "full_recommendation"] = "none"
    command_spec_generated: Literal[False] = False
    executable_training_started: Literal[False] = False


class RecipeMaterializer:
    """Turn a prior into the highest recipe stage allowed by local maturity."""

    def materialize(
        self,
        prior: RecipePrior,
        *,
        component_contracts: Mapping[str, ComponentContract] | Iterable[ComponentContract],
    ) -> RecipeMaterialization:
        contracts = _contract_mapping(component_contracts)
        missing = sorted(set(prior.component_ids) - set(contracts))
        if missing:
            return _implementation_action(
                prior,
                "implementation_proposal",
                missing,
                "Paper components do not yet have local ComponentContract records.",
                ["missing_component_contracts:" + ",".join(missing)],
            )
        selected = [contracts[item] for item in prior.component_ids]
        if prior.yolo26_compatibility == "incompatible":
            return RecipeMaterialization(
                prior_id=prior.prior_id,
                status="rejected",
                reasons=["paper_prior_is_incompatible_with_yolo26"],
            )
        current_status = _implementation_status(selected)
        missing_adapter = [
            item.component_id
            for item in selected
            if maturity_rank(item.maturity) >= maturity_rank("adapter_implemented")
            and (not item.adapter_class or not item.implementation_path)
        ]
        if current_status == "metadata_only":
            return _implementation_action(
                prior,
                "implementation_proposal",
                prior.component_ids,
                "Metadata-only components require a reviewed contract and adapter design.",
                ["metadata_only_components_cannot_materialize_recipe"],
            )
        if current_status == "adapter_required" or missing_adapter:
            return _implementation_action(
                prior,
                "implementation_request",
                prior.component_ids,
                "A concrete local adapter implementation is required before dry-run materialization.",
                ["adapter_required_before_recipe_materialization"],
                sorted(set([*prior.required_adapter, *missing_adapter])),
            )
        return _materialized_recipe(prior, current_status)


def _implementation_action(
    prior: RecipePrior,
    status: Literal["implementation_proposal", "implementation_request"],
    component_ids: list[str],
    reason: str,
    reasons: list[str],
    required_adapters: list[str] | None = None,
) -> RecipeMaterialization:
    return RecipeMaterialization(
        prior_id=prior.prior_id,
        status=status,
        implementation_action=RecipeImplementationAction(
            action_type=status,
            component_ids=component_ids,
            required_adapters=required_adapters or prior.required_adapter,
            reason=reason,
        ),
        reasons=reasons,
    )


def _materialized_recipe(prior: RecipePrior, status: str) -> RecipeMaterialization:
    recipe = _recipe_from_prior(prior, status)
    mapping = {
        "adapter_implemented": (
            "dry_run_recipe",
            "dry_run",
            "normal",
            "adapter_is_implemented_but_has_not_passed_smoke_gate",
        ),
        "smoke_passed": (
            "pilot_recipe",
            "pilot",
            "normal",
            "all_components_passed_smoke_gate; pilot_only",
        ),
        "pilot_reproduced": (
            "prioritized_pilot_recipe",
            "pilot",
            "high",
            "local_pilot_reproduction_supports_higher_candidate_priority",
        ),
        "full_reproduced": (
            "full_candidate_recommendation",
            "full_recommendation",
            "high",
            "local_full_reproduction_allows_recommendation_only; full execution still requires consent",
        ),
    }
    materialization_status, allowed_stage, priority, reason = mapping[status]
    return RecipeMaterialization(
        prior_id=prior.prior_id,
        status=materialization_status,
        recipe=recipe,
        reasons=[reason],
        recommendation_priority=priority,
        allowed_stage=allowed_stage,
    )


def _contract_mapping(
    contracts: Mapping[str, ComponentContract] | Iterable[ComponentContract],
) -> dict[str, ComponentContract]:
    if isinstance(contracts, Mapping):
        return dict(contracts)
    return {item.component_id: item for item in contracts}


def _implementation_status(contracts: list[ComponentContract]) -> str:
    minimum = min(contracts, key=lambda item: maturity_rank(item.maturity)).maturity
    if minimum == "metadata_only":
        return "metadata_only"
    if minimum == "reference_code_available":
        return "adapter_required"
    if minimum in {"adapter_implemented", "unit_tested"}:
        return "adapter_implemented"
    if minimum == "smoke_passed":
        return "smoke_passed"
    if minimum == "pilot_reproduced":
        return "pilot_reproduced"
    return "full_reproduced"


def _recipe_from_prior(prior: RecipePrior, maturity: str) -> RecipeSpec:
    common = {
        "recipe_id": f"materialized_{prior.prior_id}",
        "version": "v1",
        "target_error_facts": prior.target_error_facts,
        "target_metrics": prior.target_metrics,
        "component_ids": prior.component_ids,
        "train_overrides": {"imgsz": 640},
        "fixed_variables": {"imgsz": 640},
        "primary_changed_variable": prior.suggested_changed_variables[0],
        "compatibility_requirements": ["fixed_imgsz_640", "yolo26_compatible"],
        "expected_effects": prior.expected_paper_effect,
        "evidence_prior": [item.model_dump(mode="json") for item in prior.evidence_prior],
        "stop_conditions": ["target_error_fact_not_improved", "guard_metric_regression"],
        "promotion_requirements": ["verified_local_paired_delta", "matching_protocol_hash"],
        "maturity": maturity,
    }
    is_coupled = len(prior.component_ids) > 1 or len(prior.suggested_changed_variables) > 1
    if is_coupled:
        return CoupledRecipe(
            **common,
            coupled_variables=prior.suggested_changed_variables,
            coupling_reason=prior.coupling_reason,
            coupling_source_papers=prior.paper_ids,
            internal_ablation_plan=prior.internal_ablation_plan,
        )
    return AtomicRecipe(**common)


__all__ = [
    "MaterializationStatus",
    "RecipeImplementationAction",
    "RecipeMaterialization",
    "RecipeMaterializer",
]
