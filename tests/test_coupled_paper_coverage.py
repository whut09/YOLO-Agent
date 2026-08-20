from __future__ import annotations

import hashlib

import pytest

from yolo_agent.recipes.coupled_library import (
    CouplingEvidence,
    EvidenceBoundCoupledRecipeLibrary,
    ExplicitCoupledCombinationGenerator,
)
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)


def _evidence(
    components: list[str],
    *,
    paper_ids: list[str] | None = None,
    error_fact_types: list[str] | None = None,
    shadow: bool = False,
    dispositions: dict[str, str] | None = None,
) -> CouplingEvidence:
    return CouplingEvidence(
        evidence_kind="local_diagnosis",
        source_id="fixture:coupled-paper-coverage",
        component_ids=components,
        reason="The verified diagnosis identifies complementary mechanisms with an interaction to measure.",
        source_locations=["tests/test_coupled_paper_coverage.py#fixture"],
        paper_ids=paper_ids or ["paper:fixture"],
        error_fact_ids=["fact:coupling"],
        error_fact_types=error_fact_types or ["localization_error"],
        assignment_shadow_passed=shadow,
        shadow_evidence_ids=["shadow:passed"] if shadow else [],
        mechanism_ids=[item.replace(".", ":") for item in components],
        paper_specific_configuration={
            item: {"changed_variable": f"{item}.weight"} for item in components
        },
        required_evidence=["verified_coupling_diagnosis"],
        component_dispositions=dispositions or {},
        verified=True,
    )


@pytest.mark.parametrize(
    ("components", "facts", "shadow"),
    [
        (["loss.hard_negative_classification", "sampling.hard_negative_replay"], ["background_false_positive_class"], False),
        (["neck.rtmdet_large_kernel", "loss.quality.correlation"], ["localization_error"], False),
        (["neck.rtmdet_large_kernel", "loss.quality.pseudo_iou"], ["localization_error"], False),
        (["assigner.task_aligned", "loss.quality.correlation"], ["assignment_conflict"], True),
        (["assigner.task_aligned", "loss.quality.pseudo_iou"], ["assignment_conflict"], True),
        (["assigner.optimal_transport", "loss.quality.correlation"], ["assignment_conflict"], True),
        (["assigner.optimal_transport", "loss.quality.pseudo_iou"], ["assignment_conflict"], True),
        (["distillation.yolo26_teacher_student", "sampling.class_balanced"], ["capacity_gap"], False),
        (["domain_adaptation.feature_alignment", "domain_adaptation.domain_distillation"], ["representation_gap"], False),
        (["domain_adaptation.contrastive_alignment", "distillation.yolo26_teacher_student"], ["representation_gap"], False),
    ],
)
def test_required_coupled_pairs_materialize_as_four_arm_recipes(
    components: list[str], facts: list[str], shadow: bool
) -> None:
    result = EvidenceBoundCoupledRecipeLibrary().materialize(
        component_ids=components,
        evidence=_evidence(components, error_fact_types=facts, shadow=shadow),
    )

    assert result.decision == "materialized"
    assert result.recipe is not None
    assert result.recipe.kind == "coupled"
    assert result.recipe.component_ids == components
    assert result.recipe.combination_fingerprint
    assert [item["arm_id"] for item in result.recipe.internal_ablation_plan] == [
        "baseline",
        "arm_A",
        "arm_B",
        "arm_A_plus_B",
    ]
    assert all(
        item.get("matched_control_arm_id") == "baseline"
        for item in result.recipe.internal_ablation_plan[1:]
    )
    assert result.recipe.mechanism_ids
    assert result.recipe.paper_specific_configuration
    assert result.recipe.required_evidence


def test_assignment_shadow_failure_is_explicit_and_does_not_enter_active_queue() -> None:
    result = EvidenceBoundCoupledRecipeLibrary().materialize(
        component_ids=["assigner.optimal_transport", "loss.quality.pseudo_iou"],
        evidence=_evidence(
            ["assigner.optimal_transport", "loss.quality.pseudo_iou"],
            error_fact_types=["assignment_conflict"],
            shadow=False,
        ),
    )

    assert result.decision == "rejected"
    assert result.disposition == "implementation_request"
    assert "assignment_shadow_evidence_required" in result.blocked_by


def test_blocked_component_blocks_only_its_coupled_combination() -> None:
    result = EvidenceBoundCoupledRecipeLibrary().materialize(
        component_ids=["neck.rtmdet_large_kernel", "loss.quality.correlation"],
        evidence=_evidence(
            ["neck.rtmdet_large_kernel", "loss.quality.correlation"],
            dispositions={"neck.rtmdet_large_kernel": "blocked_runtime"},
        ),
    )

    assert result.decision == "rejected"
    assert result.disposition == "blocked_runtime"
    assert "component_blocked_runtime:neck.rtmdet_large_kernel" in result.blocked_by


def test_83_paper_provenance_is_retained_even_when_execution_is_shared() -> None:
    paper_ids = [f"fixture:paper-{index:03d}" for index in range(83)]
    records = [
        PaperExecutionSpec(
            paper_id=paper_id,
            profile_id=f"profile:{paper_id}",
            title=f"Fixture paper {index:03d}",
            source_locations=["tests/test_coupled_paper_coverage.py"],
            original_method_name="verified coupled method",
            original_method_family="object_detection",
            canonical_component_ids=["neck.rtmdet_large_kernel", "loss.quality.correlation"],
            paper_specific_mechanism_ids=["rtmdet_large_kernel", "quality_correlation"],
            required_evidence=["verified_coupling_diagnosis"],
            recipe_ids=["coupled__rtmdet_quality"],
            execution_fingerprint=hashlib.sha256(b"shared-coupled-execution").hexdigest(),
            current_disposition="queued",
            disposition_reason="queued as one shared execution with paper provenance preserved",
        )
        for index, paper_id in enumerate(paper_ids)
    ]
    inventory = PaperExecutionInventory(
        source_method_coverage_hash=hashlib.sha256(b"coverage").hexdigest(),
        all_paper_count=728,
        compatible_paper_count=83,
        exact_reproduction_candidates=0,
        records=records,
    ).with_hash()

    assert len(inventory.records) == 83
    assert len({item.paper_id for item in inventory.records}) == 83
    assert len({item.execution_fingerprint for item in inventory.records}) == 1
    assert inventory.exact_reproduction_candidates == 0


def test_same_execution_merges_paper_ids_but_different_configuration_does_not() -> None:
    components = ["neck.rtmdet_large_kernel", "loss.quality.correlation"]
    first = _evidence(components, paper_ids=["paper:a"])
    second = _evidence(components, paper_ids=["paper:b"])
    merged = ExplicitCoupledCombinationGenerator().generate(
        [(components, first), (components, second)]
    )

    assert len(merged) == 1
    assert merged[0].recipe is not None
    assert merged[0].paper_ids == ["paper:a", "paper:b"]
    assert merged[0].recipe.paper_ids == ["paper:a", "paper:b"]
