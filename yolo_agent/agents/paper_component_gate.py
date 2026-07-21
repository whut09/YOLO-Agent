"""Deterministic eligibility gate for paper-derived component candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.adapters.base import ComponentAdapter
from yolo_agent.components.compatibility import CompatibilityResult
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import MaturityName, maturity_rank
from yolo_agent.core.decision_ledger import DecisionLedger, DecisionLedgerRecord
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.recipes.paper_priors import RecipePrior
from yolo_agent.recipes.schemas import AtomicRecipe, CoupledRecipe, RecipeSpec
from yolo_agent.research.component_aliases import normalize_component_id
from yolo_agent.research.snapshot import ResearchSnapshot


ExecutionClass = Literal[
    "paper_only",
    "implementation_request",
    "dry_run_only",
    "smoke_candidate",
    "pilot_candidate",
    "full_candidate",
]
GateDecision = Literal[
    "eligible",
    "blocked",
    "evidence_required",
    "implementation_required",
    "dry_run_required",
    "smoke_required",
]


class PaperEligibilityConstraints(BaseModel):
    """Runtime invariants that proposals and LLM output cannot override."""

    model_config = ConfigDict(extra="forbid")
    imgsz: int = 640
    detector_family: str = "yolo26"
    head_mode: str = "one_to_one"
    regression_mode: str = "dfl_free"
    postprocess: str = "nms_free"
    matched_baseline: bool = False
    matched_baseline_protocol_hash: str | None = None
    research_snapshot_hash: str | None = None
    candidate_metrics_source: Literal["none", "paper_claim", "local_verified"] = "none"
    full_confirmed: bool = False


class PaperEligibilityBudget(BaseModel):
    """Current bounded cost envelope for one candidate decision."""

    model_config = ConfigDict(extra="forbid")
    max_gpu_hours: float = Field(gt=0.0)
    gpu_hours_used: float = Field(default=0.0, ge=0.0)
    estimated_candidate_gpu_hours: float | None = Field(default=None, gt=0.0)

    @property
    def remaining_gpu_hours(self) -> float:
        return max(0.0, self.max_gpu_hours - self.gpu_hours_used)


class PaperComponentGateResult(BaseModel):
    """Auditable gate output consumed before any paper candidate is queued."""

    model_config = ConfigDict(extra="forbid")
    eligible: bool
    decision: GateDecision
    blocked_by: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    required_adapter: list[str] = Field(default_factory=list)
    changed_variables: list[str] = Field(default_factory=list)
    paper_prior: list[dict[str, Any]] = Field(default_factory=list)
    local_evidence: list[dict[str, Any]] = Field(default_factory=list)
    execution_class: ExecutionClass
    eligibility_token: str | None = None

    def assert_queue_eligible(self) -> None:
        """Reject queue materialization unless this exact gate accepted training."""
        if not self.eligible or self.execution_class not in {"pilot_candidate", "full_candidate"}:
            raise PermissionError(
                f"Paper component candidate is not queue eligible: {self.decision}; "
                f"blocked_by={self.blocked_by}"
            )


class PaperComponentEligibilityGate:
    """Single deterministic authority for paper candidate queue eligibility."""

    policy_version = "paper_component_eligibility.v1"

    def __init__(self, decision_ledger: DecisionLedger) -> None:
        self.decision_ledger = decision_ledger

    def evaluate(
        self,
        *,
        run_id: str,
        recipe: RecipePrior | RecipeSpec,
        component_contracts: Mapping[str, ComponentContract] | Iterable[ComponentContract],
        component_adapters: Mapping[str, ComponentAdapter] | ComponentAdapter | None,
        compatibility: CompatibilityResult,
        maturity: MaturityName | Mapping[str, MaturityName] | None,
        fixed_constraints: PaperEligibilityConstraints,
        research_snapshot: ResearchSnapshot,
        current_error_facts: Iterable[ErrorFact],
        objective: OptimizationObjective,
        budget: PaperEligibilityBudget,
    ) -> PaperComponentGateResult:
        contracts = _contract_mapping(component_contracts)
        component_ids = list(recipe.component_ids)
        adapters = _adapter_mapping(component_adapters, component_ids)
        facts = [fact for fact in current_error_facts if fact.evidence_role == "current_observation"]
        variables = _changed_variables(recipe)
        paper_prior = _paper_prior(recipe)
        local_evidence = [_fact_summary(fact) for fact in facts]
        blocked: list[str] = []
        required_evidence: list[str] = []
        required_adapter: list[str] = []

        missing_contracts = sorted(set(component_ids) - set(contracts))
        if missing_contracts:
            blocked.extend(f"missing_component_contract:{item}" for item in missing_contracts)
        selected = [contracts[item] for item in component_ids if item in contracts]
        maturity_map = _effective_maturity(selected, maturity, blocked)
        execution_class = _execution_class(selected, maturity_map, adapters)

        _check_snapshot(recipe, research_snapshot, fixed_constraints, blocked)
        _check_fixed_imgsz(recipe, objective, fixed_constraints, blocked)
        _check_compatibility(recipe, selected, compatibility, fixed_constraints, adapters, blocked)
        _check_recipe_shape(recipe, variables, blocked)
        _check_adapters(selected, adapters, maturity_map, required_adapter, blocked)
        _check_target_evidence(recipe, facts, required_evidence, blocked)
        _check_paper_metric_source(fixed_constraints, blocked)

        if execution_class in {"pilot_candidate", "full_candidate"}:
            _check_matched_baseline(recipe, facts, objective, fixed_constraints, required_evidence, blocked)
            _check_budget(recipe, objective, budget, required_evidence, blocked)
        if execution_class == "full_candidate" and objective.full_requires_confirmation and not fixed_constraints.full_confirmed:
            blocked.append("full_run_confirmation_required")
            required_evidence.append("full_run_consent")

        blocked = sorted(set(blocked))
        required_evidence = sorted(set(required_evidence))
        required_adapter = sorted(set(required_adapter))
        decision = _decision(execution_class, blocked, required_evidence)
        eligible = not blocked and not required_evidence and execution_class in {
            "pilot_candidate",
            "full_candidate",
        }
        token = _eligibility_token(run_id, recipe, objective, research_snapshot) if eligible else None
        result = PaperComponentGateResult(
            eligible=eligible,
            decision="eligible" if eligible else decision,
            blocked_by=blocked,
            required_evidence=required_evidence,
            required_adapter=required_adapter,
            changed_variables=variables,
            paper_prior=paper_prior,
            local_evidence=local_evidence,
            execution_class=execution_class,
            eligibility_token=token,
        )
        self._record(run_id, recipe, objective, budget, research_snapshot, result)
        return result

    def _record(
        self,
        run_id: str,
        recipe: RecipePrior | RecipeSpec,
        objective: OptimizationObjective,
        budget: PaperEligibilityBudget,
        snapshot: ResearchSnapshot,
        result: PaperComponentGateResult,
    ) -> None:
        self.decision_ledger.append(DecisionLedgerRecord(
            run_id=run_id,
            policy_id=_recipe_id(recipe),
            decision_type="paper_component_eligibility",
            proposal={
                "recipe_type": type(recipe).__name__,
                "component_ids": list(recipe.component_ids),
                "changed_variables": result.changed_variables,
                "paper_prior": result.paper_prior,
            },
            decision=result.decision,
            blocked_by=result.blocked_by,
            missing_evidence=result.required_evidence,
            budget_bucket=result.execution_class,
            budget_reason=(
                f"estimated={budget.estimated_candidate_gpu_hours}; "
                f"remaining={budget.remaining_gpu_hours}"
            ),
            rationale=(
                "Deterministic PaperComponentEligibilityGate decision; "
                "LLM proposals have no override authority."
            ),
            input_summary={
                "objective_hash": objective.objective_hash,
                "research_snapshot_hash": snapshot.snapshot_hash,
                "execution_class": result.execution_class,
                "eligible": result.eligible,
                "eligibility_token": result.eligibility_token,
                "local_evidence_count": len(result.local_evidence),
                "paper_prior_count": len(result.paper_prior),
            },
            policy_version=self.policy_version,
        ))


def _contract_mapping(
    contracts: Mapping[str, ComponentContract] | Iterable[ComponentContract],
) -> dict[str, ComponentContract]:
    if isinstance(contracts, Mapping):
        return dict(contracts)
    return {item.component_id: item for item in contracts}


def _adapter_mapping(
    adapters: Mapping[str, ComponentAdapter] | ComponentAdapter | None,
    component_ids: list[str],
) -> dict[str, ComponentAdapter]:
    if adapters is None:
        return {}
    if isinstance(adapters, Mapping):
        return dict(adapters)
    return {component_ids[0]: adapters} if len(component_ids) == 1 else {}


def _effective_maturity(
    contracts: list[ComponentContract],
    supplied: MaturityName | Mapping[str, MaturityName] | None,
    blocked: list[str],
) -> dict[str, MaturityName]:
    output: dict[str, MaturityName] = {}
    for contract in contracts:
        declared = supplied.get(contract.component_id) if isinstance(supplied, Mapping) else supplied
        if declared is not None and declared != contract.maturity:
            blocked.append(f"maturity_mismatch:{contract.component_id}:{declared}!={contract.maturity}")
        output[contract.component_id] = (
            min((contract.maturity, declared), key=maturity_rank)
            if declared is not None else contract.maturity
        )
    return output


def _execution_class(
    contracts: list[ComponentContract],
    maturity: dict[str, MaturityName],
    adapters: dict[str, ComponentAdapter],
) -> ExecutionClass:
    if not contracts or any(maturity.get(item.component_id) == "metadata_only" for item in contracts):
        return "paper_only"
    if any(
        maturity_rank(maturity[item.component_id]) <= maturity_rank("reference_code_available")
        or item.component_id not in adapters
        for item in contracts
    ):
        return "implementation_request"
    minimum = min(maturity.values(), key=maturity_rank)
    if minimum == "adapter_implemented":
        return "dry_run_only"
    if minimum == "unit_tested":
        return "smoke_candidate"
    if minimum in {"smoke_passed", "pilot_reproduced"}:
        return "pilot_candidate"
    return "full_candidate"


def _check_snapshot(
    recipe: RecipePrior | RecipeSpec,
    snapshot: ResearchSnapshot,
    constraints: PaperEligibilityConstraints,
    blocked: list[str],
) -> None:
    if not snapshot.frozen:
        blocked.append("research_snapshot_not_frozen")
    if snapshot.paper_intelligence != "available":
        blocked.append("paper_intelligence_unavailable")
    if constraints.research_snapshot_hash not in {None, snapshot.snapshot_hash}:
        blocked.append("research_snapshot_constraint_mismatch")
    if isinstance(recipe, RecipePrior) and recipe.research_snapshot_hash != snapshot.snapshot_hash:
        blocked.append("recipe_prior_snapshot_mismatch")


def _check_fixed_imgsz(
    recipe: RecipePrior | RecipeSpec,
    objective: OptimizationObjective,
    constraints: PaperEligibilityConstraints,
    blocked: list[str],
) -> None:
    values = [constraints.imgsz, objective.fixed_imgsz]
    if isinstance(recipe, RecipePrior):
        values.append(recipe.baseline_protocol.get("imgsz"))
    else:
        values.extend([recipe.fixed_variables.get("imgsz"), recipe.train_overrides.get("imgsz", 640)])
        if recipe.train_overrides.get("allow_imgsz_increase") is True:
            blocked.append("automatic_imgsz_increase_forbidden")
    if any(value != 640 for value in values):
        blocked.append("fixed_imgsz_640_violation")


def _check_compatibility(
    recipe: RecipePrior | RecipeSpec,
    contracts: list[ComponentContract],
    compatibility: CompatibilityResult,
    constraints: PaperEligibilityConstraints,
    adapters: dict[str, ComponentAdapter],
    blocked: list[str],
) -> None:
    if not compatibility.ok:
        blocked.extend(f"compatibility_error:{item}" for item in compatibility.errors)
    yolo26 = compatibility.yolo26 or {}
    blocked.extend(f"yolo26_compatibility:{item}" for item in yolo26.get("blocked_by", []))
    if bool(yolo26.get("incompatible")):
        blocked.append("yolo26_incompatible")
    if isinstance(recipe, RecipePrior) and recipe.yolo26_compatibility == "incompatible":
        blocked.append("recipe_prior_yolo26_incompatible")

    component_ids = [normalize_component_id(item.component_id) for item in contracts]
    uses_nms = any(item.category == "nms" for item in contracts) or any(
        "nms" in item and "nms_free" not in item for item in component_ids
    )
    if isinstance(recipe, RecipeSpec):
        uses_nms = uses_nms or any(
            "nms" in normalize_component_id(item) and "nms_free" not in normalize_component_id(item)
            for item in recipe.inference_actions
        )
    if constraints.head_mode == "one_to_one" and (uses_nms or constraints.postprocess not in {"none", "nms_free"}):
        blocked.append("one_to_one_head_default_nms_forbidden")

    for contract in contracts:
        source_constraints = contract.tensor_input_contract.get("compatibility_constraints", {})
        requires_dfl = bool(source_constraints.get("requires_dfl") or source_constraints.get("dfl_dependent"))
        if constraints.regression_mode == "dfl_free" and requires_dfl:
            blocked.append(f"dfl_dependent_component_on_dfl_free_regression:{contract.component_id}")
        anchor_based = bool(
            source_constraints.get("anchor_based") or source_constraints.get("requires_anchors")
        )
        if contract.category == "assigner" and anchor_based and contract.component_id not in adapters:
            blocked.append(f"anchor_based_assigner_requires_adapter:{contract.component_id}")


def _check_recipe_shape(
    recipe: RecipePrior | RecipeSpec,
    variables: list[str],
    blocked: list[str],
) -> None:
    if not variables:
        blocked.append("changed_variable_missing")
        return
    if any(normalize_component_id(item) in {"imgsz", "image_size"} for item in variables):
        blocked.append("imgsz_cannot_be_changed_variable")
    multi_component = len(recipe.component_ids) > 1
    multi_variable = len(variables) > 1
    if isinstance(recipe, RecipePrior):
        if (multi_component or multi_variable) and (
            not recipe.coupling_reason or not recipe.internal_ablation_plan
        ):
            blocked.append("unexplained_component_combination")
        return
    if isinstance(recipe, AtomicRecipe) and (multi_component or multi_variable):
        blocked.append("atomic_recipe_changes_multiple_components_or_variables")
    elif isinstance(recipe, CoupledRecipe):
        if not recipe.coupling_reason or not recipe.internal_ablation_plan:
            blocked.append("coupled_recipe_missing_reason_or_ablation")
    elif multi_component or multi_variable:
        if not recipe.coupling_reason or not recipe.internal_ablation_plan:
            blocked.append("unexplained_component_combination")


def _check_adapters(
    contracts: list[ComponentContract],
    adapters: dict[str, ComponentAdapter],
    maturity: dict[str, MaturityName],
    required_adapter: list[str],
    blocked: list[str],
) -> None:
    for contract in contracts:
        effective = maturity[contract.component_id]
        if effective == "metadata_only":
            blocked.append(f"metadata_only_component:{contract.component_id}")
        adapter = adapters.get(contract.component_id)
        if adapter is None:
            required_adapter.append(contract.adapter_class or contract.component_id)
            blocked.append(f"missing_component_adapter:{contract.component_id}")
        elif contract.adapter_class and type(adapter).__name__ != contract.adapter_class:
            blocked.append(f"adapter_class_mismatch:{contract.component_id}")
        if maturity_rank(effective) < maturity_rank("smoke_passed"):
            blocked.append(f"adapter_smoke_not_passed:{contract.component_id}:{effective}")


def _check_target_evidence(
    recipe: RecipePrior | RecipeSpec,
    facts: list[ErrorFact],
    required_evidence: list[str],
    blocked: list[str],
) -> None:
    if not recipe.target_error_facts:
        blocked.append("recipe_has_no_target_error_fact")
        required_evidence.append("target_error_fact")
        return
    matching = [fact for fact in facts if any(_fact_matches_target(fact, target) for target in recipe.target_error_facts)]
    if not matching:
        blocked.append("no_current_target_error_fact")
        required_evidence.append("current_target_error_fact")


def _check_paper_metric_source(
    constraints: PaperEligibilityConstraints,
    blocked: list[str],
) -> None:
    if constraints.candidate_metrics_source == "paper_claim":
        blocked.append("paper_claim_used_as_candidate_metric")


def _check_matched_baseline(
    recipe: RecipePrior | RecipeSpec,
    facts: list[ErrorFact],
    objective: OptimizationObjective,
    constraints: PaperEligibilityConstraints,
    required_evidence: list[str],
    blocked: list[str],
) -> None:
    if not constraints.matched_baseline:
        blocked.append("matched_baseline_missing")
        required_evidence.append("matched_baseline_control")
        return
    if constraints.matched_baseline_protocol_hash != objective.baseline_protocol_hash:
        blocked.append("matched_baseline_protocol_mismatch")
        required_evidence.append("same_protocol_matched_baseline")
    fact_protocols = {fact.protocol_hash for fact in facts if fact.protocol_hash}
    if objective.baseline_protocol_hash not in fact_protocols:
        blocked.append("target_error_fact_protocol_mismatch")
        required_evidence.append("same_protocol_target_error_fact")
    if isinstance(recipe, RecipePrior):
        prior_protocols = set(recipe.baseline_protocol.get("protocol_hashes", []))
        if objective.baseline_protocol_hash not in prior_protocols:
            blocked.append("recipe_prior_baseline_protocol_mismatch")


def _check_budget(
    recipe: RecipePrior | RecipeSpec,
    objective: OptimizationObjective,
    budget: PaperEligibilityBudget,
    required_evidence: list[str],
    blocked: list[str],
) -> None:
    estimated = budget.estimated_candidate_gpu_hours
    if estimated is None and isinstance(recipe, RecipeSpec):
        raw = recipe.training_cost.get("gpu_hours")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            estimated = float(raw)
    if estimated is None:
        required_evidence.append("candidate_gpu_hour_estimate")
        return
    objective_remaining = max(0.0, objective.max_gpu_hours - budget.gpu_hours_used)
    allowed = min(budget.remaining_gpu_hours, objective_remaining)
    if estimated > allowed:
        blocked.append(f"candidate_cost_exceeds_budget:{estimated}>{allowed}")


def _changed_variables(recipe: RecipePrior | RecipeSpec) -> list[str]:
    if isinstance(recipe, RecipePrior):
        return list(dict.fromkeys(recipe.suggested_changed_variables))
    return list(dict.fromkeys([recipe.primary_changed_variable, *recipe.coupled_variables]))


def _paper_prior(recipe: RecipePrior | RecipeSpec) -> list[dict[str, Any]]:
    if isinstance(recipe, RecipePrior):
        return [item.model_dump(mode="json") for item in recipe.evidence_prior]
    return [
        dict(item)
        for item in recipe.evidence_prior
        if item.get("evidence_level") in {"paper_claim", "paper_prior"}
    ]


def _fact_summary(fact: ErrorFact) -> dict[str, Any]:
    return {
        "run_id": fact.run_id,
        "candidate_id": fact.candidate_id,
        "node_id": fact.node_id,
        "fact_type": fact.fact_type,
        "subject": fact.subject,
        "metric_name": fact.metric_name,
        "protocol_hash": fact.protocol_hash,
        "evidence_role": fact.evidence_role,
        "source": fact.source,
    }


def _fact_matches_target(fact: ErrorFact, target: dict[str, Any]) -> bool:
    strict_fields = ("node_id", "fact_type", "subject", "class_name", "area", "metric_name")
    compared = False
    for field in strict_fields:
        expected = target.get(field)
        if expected in {None, ""}:
            continue
        compared = True
        if normalize_component_id(str(getattr(fact, field, "") or "")) != normalize_component_id(str(expected)):
            return False
    return compared


def _decision(
    execution_class: ExecutionClass,
    blocked: list[str],
    required_evidence: list[str],
) -> GateDecision:
    if execution_class in {"paper_only", "implementation_request"}:
        return "implementation_required"
    if execution_class == "dry_run_only":
        return "dry_run_required"
    if execution_class == "smoke_candidate":
        return "smoke_required"
    if required_evidence:
        return "evidence_required"
    return "blocked" if blocked else "eligible"


def _recipe_id(recipe: RecipePrior | RecipeSpec) -> str:
    return recipe.prior_id if isinstance(recipe, RecipePrior) else recipe.recipe_id


def _eligibility_token(
    run_id: str,
    recipe: RecipePrior | RecipeSpec,
    objective: OptimizationObjective,
    snapshot: ResearchSnapshot,
) -> str:
    payload = {
        "run_id": run_id,
        "recipe_id": _recipe_id(recipe),
        "objective_hash": objective.objective_hash,
        "snapshot_hash": snapshot.snapshot_hash,
        "policy_version": PaperComponentEligibilityGate.policy_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ExecutionClass",
    "GateDecision",
    "PaperComponentEligibilityGate",
    "PaperComponentGateResult",
    "PaperEligibilityBudget",
    "PaperEligibilityConstraints",
]
