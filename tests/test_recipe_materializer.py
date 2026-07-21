from __future__ import annotations

import pytest

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.recipes.paper_priors import RecipePrior, RecipePriorEvidence
from yolo_agent.recipes.recipe_materializer import RecipeMaterializer
from yolo_agent.recipes.schemas import AtomicRecipe, CoupledRecipe


def _prior(
    *,
    compatibility: str = "compatible",
    components: list[str] | None = None,
    variables: list[str] | None = None,
    coupling: bool = False,
) -> RecipePrior:
    component_ids = components or ["sampling.small_object"]
    changed_variables = variables or ["data.sampler"]
    return RecipePrior(
        prior_id="paper-prior-1",
        research_snapshot_hash="snapshot-1",
        paper_ids=["paper-small"],
        component_ids=component_ids,
        target_error_facts=[{
            "run_id": "run",
            "candidate_id": "candidate",
            "node_id": "node",
            "fact_type": "area_metric",
            "subject": "small objects",
            "metric_name": "ap_small",
            "severity": "high",
            "protocol_hash": "protocol-1",
        }],
        target_metrics=["ap_small"],
        suggested_changed_variables=changed_variables,
        baseline_protocol={"imgsz": 640, "protocol_hashes": ["protocol-1"]},
        evidence_prior=[RecipePriorEvidence(
            paper_id="paper-small",
            claim="reported AP_small gain",
            source_location="notes/paper.md#method",
            evidence_level="paper_claim",
        )],
        expected_paper_effect={"ap_small": "+1.2"},
        implementation_status="smoke_passed",
        yolo26_compatibility=compatibility,
        confidence=0.75,
        source_locations=["notes/paper.md#method"],
        coupling_reason="Components are reported as a coupled method." if coupling else None,
        internal_ablation_plan=(
            [{"variant": name} for name in ["baseline", "A", "B", "A+B"]]
            if coupling else []
        ),
    )


def _contract(
    maturity: str,
    *,
    component_id: str = "sampling.small_object",
    adapter: bool | None = None,
) -> ComponentContract:
    has_adapter = maturity not in {"metadata_only", "reference_code_available"} if adapter is None else adapter
    return ComponentContract(
        component_id=component_id,
        display_name=component_id,
        category="sampling",
        implementation_path="yolo_agent.components.adapters.test" if has_adapter else None,
        adapter_class="TestAdapter" if has_adapter else None,
        maturity=maturity,
        fixed_imgsz_compatible=True,
        supported_detector_families=["yolo26"],
    )


@pytest.mark.parametrize(
    ("maturity", "expected_status", "allowed_stage", "recipe_type"),
    [
        ("metadata_only", "implementation_proposal", "none", None),
        ("reference_code_available", "implementation_request", "none", None),
        ("adapter_implemented", "dry_run_recipe", "dry_run", AtomicRecipe),
        ("smoke_passed", "pilot_recipe", "pilot", AtomicRecipe),
        ("pilot_reproduced", "prioritized_pilot_recipe", "pilot", AtomicRecipe),
        ("full_reproduced", "full_candidate_recommendation", "full_recommendation", AtomicRecipe),
    ],
)
def test_materialization_is_strictly_gated_by_local_maturity(
    maturity: str,
    expected_status: str,
    allowed_stage: str,
    recipe_type: type | None,
) -> None:
    result = RecipeMaterializer().materialize(
        _prior(),
        component_contracts={"sampling.small_object": _contract(maturity)},
    )

    assert result.status == expected_status
    assert result.allowed_stage == allowed_stage
    assert result.command_spec_generated is False
    assert result.executable_training_started is False
    if recipe_type is None:
        assert result.recipe is None
        assert result.implementation_action is not None
    else:
        assert isinstance(result.recipe, recipe_type)
        assert result.recipe.fixed_variables == {"imgsz": 640}
        assert result.recipe.train_overrides == {"imgsz": 640}
        assert result.recipe.target_error_facts
        assert result.recipe.primary_changed_variable == "data.sampler"


def test_adapter_maturity_without_real_adapter_returns_implementation_request() -> None:
    result = RecipeMaterializer().materialize(
        _prior(),
        component_contracts={
            "sampling.small_object": _contract("adapter_implemented", adapter=False),
        },
    )
    assert result.status == "implementation_request"
    assert result.recipe is None
    assert result.implementation_action.required_adapters == ["sampling.small_object"]


def test_incompatible_prior_is_rejected_before_materialization() -> None:
    result = RecipeMaterializer().materialize(
        _prior(compatibility="incompatible"),
        component_contracts={"sampling.small_object": _contract("full_reproduced")},
    )
    assert result.status == "rejected"
    assert result.recipe is None
    assert result.allowed_stage == "none"


def test_coupled_prior_materializes_only_with_reason_and_internal_ablation() -> None:
    prior = _prior(
        components=["sampling.small_object", "head.p2_small_object"],
        variables=["data.sampler", "model.head"],
        coupling=True,
    )
    result = RecipeMaterializer().materialize(
        prior,
        component_contracts={
            "sampling.small_object": _contract("smoke_passed"),
            "head.p2_small_object": _contract(
                "smoke_passed", component_id="head.p2_small_object"
            ),
        },
    )
    assert result.status == "pilot_recipe"
    assert isinstance(result.recipe, CoupledRecipe)
    assert result.recipe.coupling_reason == prior.coupling_reason
    assert len(result.recipe.internal_ablation_plan) == 4


def test_missing_contract_only_generates_implementation_proposal() -> None:
    result = RecipeMaterializer().materialize(_prior(), component_contracts={})
    assert result.status == "implementation_proposal"
    assert result.recipe is None
    assert result.implementation_action.component_ids == ["sampling.small_object"]


def test_materialized_output_never_contains_command_spec_or_training_action() -> None:
    result = RecipeMaterializer().materialize(
        _prior(),
        component_contracts={"sampling.small_object": _contract("smoke_passed")},
    )
    serialized = str(result.model_dump(mode="json"))
    assert "CommandSpec" not in serialized
    assert "run_training" not in serialized
