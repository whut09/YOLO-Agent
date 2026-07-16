from yolo_agent.agents.recipe_critic import RecipeCritic
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.policy_memory import PolicyMemoryRecord
from yolo_agent.recipes.schemas import AtomicRecipe, CoupledRecipe


def _fact() -> ErrorFact:
    return ErrorFact(run_id="run", candidate_id="base", node_id="node", fact_type="area_metric", subject="ap_small", area="small", metric_name="ap_small", value=0.2, severity="high")


def _contract(component_id="sampling.small", maturity="smoke_passed") -> ComponentContract:
    return ComponentContract(component_id=component_id, display_name=component_id, category="sampling", implementation_path="local", adapter_class="SmallAdapter", maturity=maturity, fixed_imgsz_compatible=True)


def _atomic(**updates) -> AtomicRecipe:
    data = {
        "recipe_id": "small_sampling", "version": "v1", "primary_changed_variable": "sampling",
        "component_ids": ["sampling.small"],
        "target_error_facts": [{"fact_type": "area_metric", "area": "small"}],
        "target_metrics": ["ap_small", "latency_ms", "model_size_mb"],
        "fixed_variables": {"imgsz": 640}, "train_overrides": {"imgsz": 640},
        "stop_conditions": ["pilot_no_ap_small_gain", "latency_regressed", "model_size_regressed"],
        "promotion_requirements": ["latency_guard", "model_size_guard"],
    }
    data.update(updates)
    return AtomicRecipe.model_validate(data)


def test_recipe_critic_accepts_grounded_executable_recipe() -> None:
    report = RecipeCritic().critique(_atomic(), error_facts=[_fact()], component_contracts=[_contract()], compatibility={"sampling.small": True})
    assert report.accepted and report.decision == "accepted"
    assert report.matched_error_facts == ["area_metric:ap_small"]


def test_recipe_critic_requires_adapter_for_metadata_component() -> None:
    report = RecipeCritic().critique(_atomic(), error_facts=[_fact()], component_contracts=[_contract(maturity="metadata_only")], compatibility={"sampling.small": True})
    assert not report.accepted
    assert report.decision == "needs_implementation"
    assert "component_maturity_insufficient" in report.blocked_by
    assert report.required_adapters == ["SmallAdapter"]


def test_recipe_critic_rejects_ungrounded_atomic_multi_component_and_missing_guards() -> None:
    recipe = _atomic(component_ids=["sampling.small", "head.p2"], target_error_facts=[{"fact_type": "class_confusion_pair"}], target_metrics=["ap_small"], stop_conditions=[] , promotion_requirements=[])
    report = RecipeCritic().critique(recipe, error_facts=[_fact()], component_contracts=[_contract(), _contract("head.p2")], compatibility={"sampling.small": True, "head.p2": False})
    assert {"missing_bound_error_facts", "compatibility_failed", "atomic_recipe_changes_multiple_variables", "missing_stop_condition", "missing_latency_guard", "missing_model_size_guard"} <= set(report.blocked_by)


def test_recipe_critic_reports_local_negative_evidence() -> None:
    memory = PolicyMemoryRecord(run_id="run", action="sampling.small", target="ap_small", metric_name="ap_small", delta=-0.01, trend="regressed")
    report = RecipeCritic().critique(_atomic(), error_facts=[_fact()], component_contracts=[_contract()], compatibility={"sampling.small": True}, local_evidence=[memory])
    assert report.accepted
    assert report.negative_evidence
    assert any(item.code == "local_negative_evidence" for item in report.findings)


def test_recipe_critic_requires_explicit_compatibility_result() -> None:
    report = RecipeCritic().critique(_atomic(), error_facts=[_fact()], component_contracts=[_contract()], compatibility={})
    assert "compatibility_failed" in report.blocked_by


def test_recipe_critic_checks_coupling_reason_even_for_untrusted_construct() -> None:
    recipe = CoupledRecipe.model_construct(**{
        **_atomic().model_dump(), "kind": "coupled", "recipe_id": "coupled", "component_ids": ["sampling.small", "head.p2"],
        "coupled_variables": ["sampling", "head"], "coupling_reason": None,
        "coupling_source_papers": ["paper:x"], "internal_ablation_plan": [{"name": "a"}],
    })
    report = RecipeCritic().critique(recipe, error_facts=[_fact()], component_contracts=[_contract(), _contract("head.p2")], compatibility={"sampling.small": True, "head.p2": True})
    assert "missing_coupling_reason" in report.blocked_by
