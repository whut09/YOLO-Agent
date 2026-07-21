from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.paper_component_gate import (
    PaperComponentEligibilityGate,
    PaperEligibilityBudget,
    PaperEligibilityConstraints,
)
from yolo_agent.components.adapters.dummy import DummyAdapter
from yolo_agent.components.compatibility import CompatibilityResult
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.decision_ledger import DecisionLedger
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.recipes.paper_priors import RecipePrior, RecipePriorEvidence
from yolo_agent.recipes.schemas import AtomicRecipe, CoupledRecipe
from yolo_agent.research.snapshot import ResearchSnapshot


def _objective() -> OptimizationObjective:
    return OptimizationObjective(
        baseline_run_id="run",
        baseline_candidate_id="baseline",
        baseline_protocol_hash="protocol-1",
    )


def _snapshot() -> ResearchSnapshot:
    return ResearchSnapshot.model_construct(
        snapshot_hash="snapshot-1",
        paper_intelligence="available",
        frozen=True,
    )


def _fact(*, role: str = "current_observation") -> ErrorFact:
    return ErrorFact(
        run_id="run",
        candidate_id="candidate",
        node_id="node",
        fact_type="area_metric",
        subject="small object AP is low",
        metric_name="ap_small",
        area="small",
        severity="high",
        protocol_hash="protocol-1",
        evidence_role=role,
    )


def _contract(
    component_id: str = "sampling.small_object",
    *,
    maturity: str = "smoke_passed",
    category: str = "sampling",
    compatibility_constraints: dict[str, bool] | None = None,
) -> ComponentContract:
    return ComponentContract(
        component_id=component_id,
        display_name=component_id,
        category=category,
        implementation_path="yolo_agent.components.adapters.dummy",
        adapter_class="DummyAdapter",
        maturity=maturity,
        fixed_imgsz_compatible=True,
        supported_detector_families=["yolo26"],
        tensor_input_contract={"compatibility_constraints": compatibility_constraints or {}},
    )


def _compatibility(*, ok: bool = True, blocked: list[str] | None = None) -> CompatibilityResult:
    return CompatibilityResult(
        ok=ok,
        errors=[] if ok else blocked or ["compatibility_error"],
        yolo26={
            "compatible": ok,
            "incompatible": not ok,
            "blocked_by": blocked or [],
            "required_adapters": [],
        },
    )


def _recipe(
    *,
    components: list[str] | None = None,
    variable: str = "data.sampler",
    maturity: str = "smoke_passed",
    evidence: list[dict[str, str]] | None = None,
) -> AtomicRecipe:
    return AtomicRecipe(
        recipe_id="paper_recipe",
        version="v1",
        target_error_facts=[{
            "fact_type": "area_metric",
            "subject": "small object AP is low",
            "metric_name": "ap_small",
        }],
        target_metrics=["ap_small"],
        component_ids=components or ["sampling.small_object"],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable=variable,
        evidence_prior=evidence or [{"evidence_level": "paper_claim", "paper_id": "paper-1"}],
        maturity=maturity,
    )


def _constraints(**updates: object) -> PaperEligibilityConstraints:
    values: dict[str, object] = {
        "imgsz": 640,
        "matched_baseline": True,
        "matched_baseline_protocol_hash": "protocol-1",
        "research_snapshot_hash": "snapshot-1",
        "candidate_metrics_source": "local_verified",
    }
    values.update(updates)
    return PaperEligibilityConstraints.model_validate(values)


def _gate(tmp_path: Path) -> PaperComponentEligibilityGate:
    return PaperComponentEligibilityGate(DecisionLedger(tmp_path / "decision_ledger.jsonl"))


def _evaluate(
    tmp_path: Path,
    *,
    recipe: AtomicRecipe | CoupledRecipe | RecipePrior | None = None,
    contract: ComponentContract | None = None,
    contracts: dict[str, ComponentContract] | None = None,
    adapters: dict[str, object] | object | None = None,
    compatibility: CompatibilityResult | None = None,
    constraints: PaperEligibilityConstraints | None = None,
    facts: list[ErrorFact] | None = None,
    budget: PaperEligibilityBudget | None = None,
    maturity: str | dict[str, str] | None = None,
):
    default_contract = contract or _contract()
    selected = contracts or {default_contract.component_id: default_contract}
    return _gate(tmp_path).evaluate(
        run_id="run",
        recipe=recipe or _recipe(),
        component_contracts=selected,
        component_adapters=adapters if adapters is not None else {item: DummyAdapter() for item in selected},
        compatibility=compatibility or _compatibility(),
        maturity=maturity,
        fixed_constraints=constraints or _constraints(),
        research_snapshot=_snapshot(),
        current_error_facts=facts if facts is not None else [_fact()],
        objective=_objective(),
        budget=budget or PaperEligibilityBudget(max_gpu_hours=4, estimated_candidate_gpu_hours=1),
    )


def test_smoke_passed_component_is_pilot_candidate_and_writes_ledger(tmp_path: Path) -> None:
    result = _evaluate(tmp_path)

    assert result.eligible is True
    assert result.decision == "eligible"
    assert result.execution_class == "pilot_candidate"
    assert result.changed_variables == ["data.sampler"]
    assert result.paper_prior[0]["evidence_level"] == "paper_claim"
    assert result.local_evidence[0]["evidence_role"] == "current_observation"
    assert result.eligibility_token
    result.assert_queue_eligible()
    records = DecisionLedger(tmp_path / "decision_ledger.jsonl").read()
    assert len(records) == 1
    assert records[0].decision_type == "paper_component_eligibility"
    assert records[0].decision == "eligible"


def test_full_reproduced_is_only_a_full_recommendation(tmp_path: Path) -> None:
    result = _evaluate(
        tmp_path,
        contract=_contract(maturity="full_reproduced"),
        recipe=_recipe(maturity="full_reproduced"),
        constraints=_constraints(full_confirmed=True),
    )

    assert result.eligible is True
    assert result.execution_class == "full_candidate"
    result.assert_queue_eligible()


def test_full_candidate_still_requires_explicit_consent(tmp_path: Path) -> None:
    result = _evaluate(
        tmp_path,
        contract=_contract(maturity="full_reproduced"),
        recipe=_recipe(maturity="full_reproduced"),
    )

    assert result.eligible is False
    assert result.execution_class == "full_candidate"
    assert "full_run_confirmation_required" in result.blocked_by
    assert "full_run_consent" in result.required_evidence


@pytest.mark.parametrize(
    ("maturity", "execution_class", "decision"),
    [
        ("metadata_only", "paper_only", "implementation_required"),
        ("reference_code_available", "implementation_request", "implementation_required"),
        ("adapter_implemented", "dry_run_only", "dry_run_required"),
        ("unit_tested", "smoke_candidate", "smoke_required"),
    ],
)
def test_maturity_gate_never_queues_unready_component(
    tmp_path: Path,
    maturity: str,
    execution_class: str,
    decision: str,
) -> None:
    result = _evaluate(
        tmp_path,
        contract=_contract(maturity=maturity),
        adapters={"sampling.small_object": DummyAdapter()},
    )

    assert result.eligible is False
    assert result.execution_class == execution_class
    assert result.decision == decision
    with pytest.raises(PermissionError):
        result.assert_queue_eligible()


def test_missing_adapter_is_implementation_request(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, adapters={})

    assert result.eligible is False
    assert result.execution_class == "implementation_request"
    assert "missing_component_adapter:sampling.small_object" in result.blocked_by
    assert result.required_adapter == ["DummyAdapter"]


def test_supplied_maturity_cannot_override_contract_maturity(tmp_path: Path) -> None:
    result = _evaluate(
        tmp_path,
        contract=_contract(maturity="metadata_only"),
        maturity="smoke_passed",
    )

    assert result.eligible is False
    assert "maturity_mismatch:sampling.small_object:smoke_passed!=metadata_only" in result.blocked_by
    assert result.execution_class == "paper_only"


def test_compatibility_result_cannot_be_bypassed(tmp_path: Path) -> None:
    result = _evaluate(
        tmp_path,
        compatibility=_compatibility(ok=False, blocked=["one_to_one_head_uses_nms_recipe"]),
    )

    assert result.eligible is False
    assert "yolo26_compatibility:one_to_one_head_uses_nms_recipe" in result.blocked_by


def test_fixed_imgsz_violation_is_blocked(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, constraints=_constraints(imgsz=1280))

    assert result.eligible is False
    assert "fixed_imgsz_640_violation" in result.blocked_by


def test_dfl_free_regression_blocks_dfl_component(tmp_path: Path) -> None:
    contract = _contract(
        "loss.localization.dfl",
        category="bbox_regression_loss",
        compatibility_constraints={"requires_dfl": True},
    )
    recipe = _recipe()
    recipe = recipe.model_copy(update={"component_ids": [contract.component_id]})
    result = _evaluate(
        tmp_path,
        recipe=recipe,
        contract=contract,
        adapters={contract.component_id: DummyAdapter()},
    )

    assert "dfl_dependent_component_on_dfl_free_regression:loss.localization.dfl" in result.blocked_by


def test_one_to_one_head_blocks_default_nms(tmp_path: Path) -> None:
    contract = _contract("postprocess.nms", category="nms")
    recipe = _recipe()
    recipe = recipe.model_copy(update={"component_ids": [contract.component_id]})
    result = _evaluate(
        tmp_path,
        recipe=recipe,
        contract=contract,
        adapters={contract.component_id: DummyAdapter()},
    )

    assert "one_to_one_head_default_nms_forbidden" in result.blocked_by


def test_anchor_assigner_without_adapter_is_blocked(tmp_path: Path) -> None:
    contract = _contract(
        "assigner.anchor_based",
        category="assigner",
        compatibility_constraints={"anchor_based": True},
    )
    recipe = _recipe()
    recipe = recipe.model_copy(update={"component_ids": [contract.component_id]})
    result = _evaluate(tmp_path, recipe=recipe, contract=contract, adapters={})

    assert "anchor_based_assigner_requires_adapter:assigner.anchor_based" in result.blocked_by


def test_paper_claim_cannot_be_candidate_metric(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, constraints=_constraints(candidate_metrics_source="paper_claim"))

    assert result.eligible is False
    assert "paper_claim_used_as_candidate_metric" in result.blocked_by


def test_missing_target_error_fact_is_evidence_required(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, facts=[])

    assert result.eligible is False
    assert result.decision == "evidence_required"
    assert "current_target_error_fact" in result.required_evidence


def test_missing_matched_baseline_is_evidence_required(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, constraints=_constraints(matched_baseline=False))

    assert result.eligible is False
    assert result.decision == "evidence_required"
    assert "matched_baseline_control" in result.required_evidence


def test_atomic_recipe_with_multiple_changes_is_blocked(tmp_path: Path) -> None:
    contracts = {
        "sampling.small_object": _contract(),
        "head.p2_small_object": _contract("head.p2_small_object", category="detection_head"),
    }
    recipe = _recipe(components=list(contracts), variable="data.sampler")
    result = _evaluate(
        tmp_path,
        recipe=recipe,
        contracts=contracts,
        adapters={item: DummyAdapter() for item in contracts},
    )

    assert "atomic_recipe_changes_multiple_components_or_variables" in result.blocked_by


def test_coupled_recipe_can_pass_structural_gate(tmp_path: Path) -> None:
    contracts = {
        "sampling.small_object": _contract(),
        "head.p2_small_object": _contract("head.p2_small_object", category="detection_head"),
    }
    recipe = CoupledRecipe(
        recipe_id="coupled",
        version="v1",
        target_error_facts=_recipe().target_error_facts,
        component_ids=list(contracts),
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="data.sampler",
        coupled_variables=["data.sampler", "model.head"],
        coupling_reason="The paper method explicitly couples sampling and P2.",
        coupling_source_papers=["paper-1"],
        internal_ablation_plan=[{"variant": item} for item in ["baseline", "A", "B", "A+B"]],
        maturity="smoke_passed",
    )
    result = _evaluate(
        tmp_path,
        recipe=recipe,
        contracts=contracts,
        adapters={item: DummyAdapter() for item in contracts},
    )

    assert result.eligible is True
    assert result.changed_variables == ["data.sampler", "model.head"]


def test_budget_overrun_is_blocked(tmp_path: Path) -> None:
    result = _evaluate(
        tmp_path,
        budget=PaperEligibilityBudget(
            max_gpu_hours=2,
            gpu_hours_used=1.5,
            estimated_candidate_gpu_hours=1,
        ),
    )

    assert result.eligible is False
    assert any(item.startswith("candidate_cost_exceeds_budget:") for item in result.blocked_by)


def test_missing_cost_estimate_requests_budget_evidence(tmp_path: Path) -> None:
    result = _evaluate(
        tmp_path,
        budget=PaperEligibilityBudget(max_gpu_hours=2),
    )

    assert result.eligible is False
    assert result.decision == "evidence_required"
    assert "candidate_gpu_hour_estimate" in result.required_evidence


def test_recipe_prior_is_supported_and_keeps_local_evidence_separate(tmp_path: Path) -> None:
    prior = RecipePrior(
        prior_id="prior-1",
        research_snapshot_hash="snapshot-1",
        paper_ids=["paper-1"],
        component_ids=["sampling.small_object"],
        target_error_facts=_recipe().target_error_facts,
        target_metrics=["ap_small"],
        suggested_changed_variables=["data.sampler"],
        baseline_protocol={"imgsz": 640, "protocol_hashes": ["protocol-1"]},
        evidence_prior=[RecipePriorEvidence(
            paper_id="paper-1", claim="paper claim", source_location="paper.md#1", evidence_level="paper_claim"
        )],
        expected_paper_effect={"ap_small": "+1.2"},
        implementation_status="smoke_passed",
        yolo26_compatibility="compatible",
        confidence=0.8,
        source_locations=["paper.md#1"],
    )
    result = _evaluate(tmp_path, recipe=prior)

    assert result.eligible is True
    assert result.paper_prior[0]["evidence_level"] == "paper_claim"
    assert result.local_evidence[0]["evidence_role"] == "current_observation"
