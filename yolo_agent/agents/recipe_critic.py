"""Deterministic evidence and execution critic for component recipes."""

from __future__ import annotations

from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.policy_memory import PolicyMemoryRecord
from yolo_agent.recipes.schemas import AtomicRecipe, CoupledRecipe, RecipeSpec


CriticDecision = Literal["accepted", "rejected", "needs_implementation"]
FindingSeverity = Literal["error", "warning", "info"]


class RecipeCriticFinding(BaseModel):
    code: str
    severity: FindingSeverity
    message: str
    component_id: str | None = None


class RecipeCriticReport(BaseModel):
    recipe_id: str
    decision: CriticDecision
    accepted: bool
    findings: list[RecipeCriticFinding] = Field(default_factory=list)
    matched_error_facts: list[str] = Field(default_factory=list)
    required_adapters: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)

    @property
    def blocked_by(self) -> list[str]:
        return [item.code for item in self.findings if item.severity == "error"]


class RecipeCritic:
    """Check recipe evidence, contracts, compatibility, and guard metrics."""

    def critique(
        self,
        recipe: RecipeSpec,
        *,
        error_facts: Iterable[ErrorFact],
        component_contracts: Iterable[ComponentContract],
        compatibility: dict[str, bool | dict[str, Any]],
        local_evidence: Iterable[PolicyMemoryRecord | dict[str, Any]] = (),
    ) -> RecipeCriticReport:
        facts = list(error_facts)
        contracts = {item.component_id: item for item in component_contracts}
        findings: list[RecipeCriticFinding] = []
        required_adapters: list[str] = []

        matched = [fact for fact in facts if _fact_matches(recipe, fact)]
        if not matched:
            findings.append(RecipeCriticFinding(code="missing_bound_error_facts", severity="error", message="Recipe does not match any supplied local error fact."))

        for component_id in recipe.component_ids:
            contract = contracts.get(component_id)
            if contract is None:
                findings.append(RecipeCriticFinding(code="unknown_component", severity="error", message=f"Component {component_id} is not registered.", component_id=component_id))
                continue
            if not contract.can_execute:
                required_adapters.append(contract.adapter_class or f"adapter_for:{component_id}")
                findings.append(RecipeCriticFinding(code="component_maturity_insufficient", severity="error", message=f"Component {component_id} maturity is {contract.maturity}; smoke_passed is required.", component_id=component_id))
            if not contract.adapter_class or not contract.implementation_path:
                required_adapters.append(contract.adapter_class or f"adapter_for:{component_id}")
                findings.append(RecipeCriticFinding(code="adapter_required", severity="error", message=f"Component {component_id} needs an implemented adapter.", component_id=component_id))
            compatible, reason = _compatibility_for(component_id, compatibility)
            if not compatible:
                findings.append(RecipeCriticFinding(code="compatibility_failed", severity="error", message=reason or f"Compatibility failed for {component_id}.", component_id=component_id))

        if recipe.fixed_variables.get("imgsz") != 640 or recipe.train_overrides.get("imgsz", 640) != 640:
            findings.append(RecipeCriticFinding(code="violates_fixed_imgsz", severity="error", message="Recipe must preserve imgsz=640."))
        if isinstance(recipe, AtomicRecipe) and (recipe.coupled_variables or len(recipe.component_ids) > 1):
            findings.append(RecipeCriticFinding(code="atomic_recipe_changes_multiple_variables", severity="error", message="Atomic recipe changes multiple variables/components."))
        if isinstance(recipe, CoupledRecipe) and not recipe.coupling_reason:
            findings.append(RecipeCriticFinding(code="missing_coupling_reason", severity="error", message="Coupled recipe requires coupling_reason."))
        if not recipe.stop_conditions:
            findings.append(RecipeCriticFinding(code="missing_stop_condition", severity="error", message="Recipe must define a pilot stop condition."))
        if not _has_guard(recipe, "latency"):
            findings.append(RecipeCriticFinding(code="missing_latency_guard", severity="error", message="Recipe lacks a latency guard metric or stop condition."))
        if not _has_guard(recipe, "model_size"):
            findings.append(RecipeCriticFinding(code="missing_model_size_guard", severity="error", message="Recipe lacks a model-size guard metric or stop condition."))

        negatives = _negative_local_evidence(recipe, local_evidence)
        for item in negatives:
            findings.append(RecipeCriticFinding(code="local_negative_evidence", severity="warning", message=item))

        errors = [item for item in findings if item.severity == "error"]
        implementation_errors = {"component_maturity_insufficient", "adapter_required"}
        decision: CriticDecision = "accepted"
        if errors:
            decision = "needs_implementation" if all(item.code in implementation_errors for item in errors) else "rejected"
        return RecipeCriticReport(
            recipe_id=recipe.recipe_id,
            decision=decision,
            accepted=not errors,
            findings=findings,
            matched_error_facts=[f"{item.fact_type}:{item.subject}" for item in matched],
            required_adapters=sorted(set(required_adapters)),
            negative_evidence=negatives,
        )


def _fact_matches(recipe: RecipeSpec, fact: ErrorFact) -> bool:
    values = {fact.fact_type, fact.subject, fact.metric_name or "", fact.area or "", fact.class_name or ""}
    for target in recipe.target_error_facts:
        if all(str(value) in values for key, value in target.items() if key in {"fact_type", "subject", "metric_name", "area", "class_name"} and value is not None):
            return True
    return False


def _compatibility_for(component_id: str, compatibility: dict[str, bool | dict[str, Any]]) -> tuple[bool, str]:
    value = compatibility.get(component_id, compatibility.get("overall"))
    if isinstance(value, bool):
        return value, ""
    if isinstance(value, dict):
        ok = bool(value.get("compatible", value.get("ok", False)))
        reasons = value.get("blocked_by") or value.get("errors") or []
        return ok, ", ".join(str(item) for item in reasons)
    return False, f"missing compatibility result for {component_id}"


def _has_guard(recipe: RecipeSpec, name: str) -> bool:
    text = " ".join([
        *recipe.target_metrics,
        *recipe.stop_conditions,
        *recipe.promotion_requirements,
        *[str(key) for key in recipe.inference_cost],
        *[str(key) for key in recipe.training_cost],
    ]).lower()
    return name in text


def _negative_local_evidence(recipe: RecipeSpec, evidence: Iterable[PolicyMemoryRecord | dict[str, Any]]) -> list[str]:
    actions = {recipe.recipe_id, *recipe.component_ids}
    negatives: list[str] = []
    for raw in evidence:
        item = raw.model_dump(mode="json") if isinstance(raw, PolicyMemoryRecord) else raw
        if str(item.get("action")) not in actions:
            continue
        delta = item.get("effect_delta", item.get("delta"))
        trend = str(item.get("trend", ""))
        if trend == "regressed" or isinstance(delta, (int, float)) and float(delta) < 0:
            negatives.append(f"{item.get('action')} regressed {item.get('metric_name') or item.get('target')}: {delta}")
    return negatives


__all__ = ["RecipeCritic", "RecipeCriticFinding", "RecipeCriticReport"]
