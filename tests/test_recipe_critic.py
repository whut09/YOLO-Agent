from yolo_agent.agents.recipe_critic import (
    RecipeCritic,
    error_fact_id,
    recipe_matches_error_fact,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.policy_memory import PolicyMemoryRecord
from yolo_agent.recipes.schemas import AtomicRecipe, CoupledRecipe
from tests.maturity_helpers import with_smoke_artifact


def _fact() -> ErrorFact:
    return ErrorFact(run_id="run", candidate_id="base", node_id="node", fact_type="area_metric", subject="ap_small", area="small", metric_name="ap_small", value=0.2, severity="high")


def _contract(component_id="sampling.small", maturity="smoke_passed") -> ComponentContract:
    contract = ComponentContract(component_id=component_id, display_name=component_id, category="sampling", implementation_path="local", adapter_class="SmallAdapter", maturity=maturity, fixed_imgsz_compatible=True)
    return with_smoke_artifact(contract) if maturity == "smoke_passed" else contract


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
    assert "incomplete_coupled_ablation_plan" in report.blocked_by


def test_recipe_critic_requires_assignment_shadow_for_active_quality_pair() -> None:
    assigner = ComponentContract(
        component_id="assigner.task_aligned",
        display_name="Task aligned",
        category="assigner",
        implementation_path="local",
        adapter_class="TaskAlignedAdapter",
        maturity="smoke_passed",
        fixed_imgsz_compatible=True,
    )
    quality = ComponentContract(
        component_id="loss.quality.correlation",
        display_name="Correlation quality loss",
        category="classification_loss",
        implementation_path="local",
        adapter_class="QualityAdapter",
        maturity="smoke_passed",
        fixed_imgsz_compatible=True,
    )
    recipe = CoupledRecipe(
        recipe_id="task-aligned-correlation",
        version="v1",
        target_error_facts=[{"fact_type": "localization_error"}],
        target_metrics=["map50_95", "latency_ms", "model_size_mb"],
        component_ids=[assigner.component_id, quality.component_id],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="assignment.policy",
        coupled_variables=["assignment.policy", "loss.quality.weight"],
        coupling_reason="Assignment and quality alignment require separate arms.",
        coupling_source_papers=["paper:assignment-quality"],
        internal_ablation_plan=[
            {"name": "baseline", "components": []},
            {"name": "A", "components": [assigner.component_id]},
            {"name": "B", "components": [quality.component_id]},
            {"name": "A+B", "components": [assigner.component_id, quality.component_id]},
        ],
        stop_conditions=["latency_guard", "model_size_guard"],
    )
    report = RecipeCritic().critique(
        recipe,
        error_facts=[
            ErrorFact(
                run_id="run",
                candidate_id="base",
                node_id="node",
                fact_type="localization_error",
                subject="all",
            )
        ],
        component_contracts=[assigner, quality],
        compatibility={assigner.component_id: True, quality.component_id: True},
    )

    assert "assignment_shadow_evidence_required" in report.blocked_by


def test_error_fact_binding_is_field_exact_and_requires_fact_constraints() -> None:
    fact = _fact()
    wrong_field = _atomic(target_error_facts=[{"subject": "area_metric"}])
    metadata_only = _atomic(target_error_facts=[{"component": "sampling.small"}])
    exact = _atomic(
        target_error_facts=[
            {
                "fact_type": "area_metric",
                "area": "small",
                "metric_name": "ap_small",
                "severity": "high",
            }
        ]
    )

    assert recipe_matches_error_fact(wrong_field, fact) is False
    assert recipe_matches_error_fact(metadata_only, fact) is False
    assert recipe_matches_error_fact(exact, fact) is True


def test_recipe_matches_multiple_fact_patterns_and_reports_concrete_ids() -> None:
    recipe = _atomic(
        recipe_id="quality_multi_pattern",
        target_error_facts=[
            {"fact_type": "confidence_localization_mismatch"},
            {"fact_type": "localization_error"},
        ],
    )
    facts = [
        ErrorFact(
            run_id="run",
            candidate_id="base",
            node_id="node-confidence",
            fact_type="confidence_localization_mismatch",
            subject="person",
        ),
        ErrorFact(
            run_id="run",
            candidate_id="base",
            node_id="node-localization",
            fact_type="localization_error",
            subject="person",
        ),
    ]

    report = RecipeCritic().critique(
        recipe,
        error_facts=facts,
        component_contracts=[_contract()],
        compatibility={"sampling.small": True},
    )

    assert report.accepted
    assert report.matched_error_fact_ids == [error_fact_id(fact) for fact in facts]


def test_abstract_iou_aware_classification_requires_implementation_request() -> None:
    recipe = _atomic(
        recipe_id="abstract_iou_aware",
        component_ids=["loss.quality.iou_aware_classification"],
        target_error_facts=[{"fact_type": "localization_error"}],
    )
    contract = _contract("loss.quality.iou_aware_classification")

    report = RecipeCritic().critique(
        recipe,
        error_facts=[
            ErrorFact(
                run_id="run",
                candidate_id="base",
                node_id="node",
                fact_type="localization_error",
                subject="person",
            )
        ],
        component_contracts=[contract],
        compatibility={contract.component_id: True},
    )

    assert report.decision == "needs_implementation"
    assert "abstract_quality_component_requires_implementation_request" in report.blocked_by
