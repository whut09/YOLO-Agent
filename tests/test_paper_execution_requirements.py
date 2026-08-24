from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.executable_coverage import (
    ExecutablePaperCoverageAuditor,
    method_coverage_file_hash,
)
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.paper_execution_inventory import (
    PaperExecutionInventoryBuilder,
)
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirement,
)
from yolo_agent.research.paper_execution_requirements import (
    PaperExecutionRequirementsBuilder,
    build_paper_execution_requirements,
)
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.resources import ResourcePaths


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_COVERAGE = ROOT / "research" / "production" / "paper_method_coverage.yaml"
GENERIC_MECHANISMS = {
    "distillation.yolo26_teacher_student",
    "domain_adaptation.general",
    "quality_alignment.general",
}


@pytest.fixture(scope="module")
def production_inventory() -> PaperExecutionInventory:
    method_coverage = PaperMethodCoverageReport.from_yaml(PRODUCTION_COVERAGE)
    aliases = ComponentAliasResolver.from_yaml()
    executable = ExecutablePaperCoverageAuditor(contracts=aliases.contracts).build(
        method_coverage,
        source_method_coverage_hash=method_coverage_file_hash(PRODUCTION_COVERAGE),
        source_taxonomy_hash="requirements-matrix-test",
    )
    recipes = RecipeRegistry.from_paths(
        [
            ResourcePaths.RECIPE_BUNDLES,
            *sorted(ResourcePaths.RECIPES_DIR.glob("*.yaml")),
        ],
        strict=False,
    )
    return PaperExecutionInventoryBuilder().build(
        method_coverage,
        executable,
        PaperRegistry(ROOT / "research").list(),
        recipes.list(),
        expected_compatible_count=83,
    )


@pytest.fixture(scope="module")
def requirements(production_inventory: PaperExecutionInventory):  # type: ignore[no-untyped-def]
    return PaperExecutionRequirementsBuilder().build(
        production_inventory,
        source_inventory_path=ROOT / "runs" / "coverage-audit" / "paper_execution_inventory.yaml",
    )


def test_matrix_preserves_all_83_papers_once(
    production_inventory: PaperExecutionInventory, requirements
) -> None:  # type: ignore[no-untyped-def]
    expected = {item.paper_id for item in production_inventory.records}
    actual = [item.paper_id for item in requirements.requirements]

    assert requirements.compatible_paper_count == 83
    assert len(actual) == 83
    assert len(set(actual)) == 83
    assert set(actual) == expected
    assert actual == sorted(actual)


def test_generic_mechanisms_never_become_primary_requirements(requirements) -> None:  # type: ignore[no-untyped-def]
    assert all(
        item.paper_specific_mechanism not in GENERIC_MECHANISMS
        for item in requirements.requirements
    )
    assert all(
        not item.paper_specific_mechanism.endswith(".general")
        for item in requirements.requirements
    )


def test_all_distillation_papers_have_specific_branch_or_unresolved_blocker(
    production_inventory: PaperExecutionInventory, requirements
) -> None:  # type: ignore[no-untyped-def]
    source_ids = {
        item.paper_id
        for item in production_inventory.records
        if "distillation.yolo26_teacher_student" in item.canonical_component_ids
    }
    rows = [item for item in requirements.requirements if item.paper_id in source_ids]

    assert len(source_ids) == 32
    assert len(rows) == 32
    for item in rows:
        mechanisms = set(item.paper_specific_mechanism_ids)
        assert mechanisms
        assert "distillation.yolo26_teacher_student" not in mechanisms
        distillation_ids = {
            mechanism
            for mechanism in mechanisms
            if mechanism.startswith("distillation.")
            or mechanism.endswith("_distillation")
            or mechanism in {
                "cross_domain_teacher",
                "source_free_teacher",
                "teacher_ensemble",
            }
        }
        assert distillation_ids
        if any(".unresolved_" in mechanism for mechanism in distillation_ids):
            assert item.execution_route == "implementation_request"
            assert "unresolved" in (item.exact_blocker or "") or (
                "distillation_branch_unmapped" in (item.exact_blocker or "")
            )
            assert not item.training_candidate_allowed
        else:
            assert "frozen_teacher_checkpoint" in item.required_teacher_assets
            assert not item.training_candidate_allowed


def test_all_domain_papers_have_specific_branch_and_explicit_domain_assets(
    production_inventory: PaperExecutionInventory, requirements
) -> None:  # type: ignore[no-untyped-def]
    source_ids = {
        item.paper_id
        for item in production_inventory.records
        if "domain_adaptation.general" in item.canonical_component_ids
    }
    rows = [item for item in requirements.requirements if item.paper_id in source_ids]

    assert len(source_ids) == 40
    assert len(rows) == 40
    assert all(item.required_domain_assets for item in rows)
    assert all("target_domain_dataset" in item.required_domain_assets for item in rows)
    assert all(item.required_manifest_assets for item in rows)
    assert all(item.execution_route in {"evidence_recovery", "implementation_request"} for item in rows)
    assert all(not item.training_candidate_allowed for item in rows)
    assert all(item.paper_specific_mechanism != "domain_adaptation.general" for item in rows)
    assert all(
        "never use COCO train/val as domains" in item.recovery_action
        for item in rows
    )


def test_domain_distillation_paper_preserves_both_requirement_families(
    requirements,
) -> None:  # type: ignore[no-untyped-def]
    item = next(
        row
        for row in requirements.requirements
        if row.paper_id == "ecva:eccv2024:11254"
    )

    assert "cross_domain_teacher" in item.paper_specific_mechanism_ids
    assert any(
        mechanism.startswith("distillation.")
        for mechanism in item.paper_specific_mechanism_ids
    )
    assert not item.training_candidate_allowed
    assert "frozen_teacher_checkpoint" in item.required_teacher_assets
    assert "teacher_checkpoint_missing" in (item.exact_blocker or "")


def test_sahi_is_inference_only(requirements) -> None:  # type: ignore[no-untyped-def]
    item = next(
        row
        for row in requirements.requirements
        if row.paper_specific_mechanism == "inference.sahi_slicing"
    )

    assert item.execution_route == "inference"
    assert not item.training_candidate_allowed
    assert item.required_runtime_payload["training"] is False
    assert item.exact_blocker == "inference_only_not_training_candidate"


def test_independent_training_mechanisms_are_not_collapsed(requirements) -> None:  # type: ignore[no-untyped-def]
    mechanisms = {
        item.paper_specific_mechanism for item in requirements.requirements
    }

    assert {
        "assigner.optimal_transport",
        "assigner.task_aligned",
        "assigner.dynamic_smooth_label",
        "loss.quality.correlation",
        "loss.quality.pseudo_iou",
        "loss.calibration.bpc",
        "feature_pyramid.multi_scale",
        "attention.spatial",
    }.issubset(mechanisms)


def test_unknown_mechanism_fails_closed() -> None:
    record = PaperExecutionSpec(
        paper_id="fixture:unknown",
        profile_id="profile:unknown",
        title="Unknown paper mechanism",
        source_locations=["fixture"],
        canonical_component_ids=["unknown.paper.component"],
        execution_fingerprint=hashlib.sha256(b"unknown").hexdigest(),
        current_disposition="implementation_request",
        disposition_reason="paper mechanism is unresolved",
    )
    inventory = PaperExecutionInventory(
        source_method_coverage_hash="a" * 64,
        all_paper_count=1,
        compatible_paper_count=1,
        exact_reproduction_candidates=0,
        records=[record],
    ).with_hash()

    item = PaperExecutionRequirementsBuilder().build(
        inventory, source_inventory_path="fixture.yaml"
    ).requirements[0]

    assert item.execution_route == "implementation_request"
    assert item.paper_specific_mechanism.startswith("paper.unresolved_")
    assert item.required_adapter is None
    assert not item.training_candidate_allowed
    assert item.exact_blocker == "paper_specific_mechanism_unresolved"


def test_builder_writes_a_roundtrippable_matrix(
    tmp_path: Path, production_inventory: PaperExecutionInventory
) -> None:
    inventory_path = tmp_path / "paper_execution_inventory.yaml"
    output_path = tmp_path / "paper_execution_requirements.yaml"
    production_inventory.to_yaml(inventory_path)

    matrix = build_paper_execution_requirements(inventory_path, output_path)

    assert matrix.compatible_paper_count == 83
    assert output_path.is_file()
    assert "compatible_paper_count: 83" in output_path.read_text(encoding="utf-8")


def test_requirements_build_is_semantically_deterministic(
    production_inventory: PaperExecutionInventory,
) -> None:
    builder = PaperExecutionRequirementsBuilder()
    first = builder.build(
        production_inventory,
        source_inventory_path="paper_execution_inventory.yaml",
    )
    second = builder.build(
        production_inventory,
        source_inventory_path="paper_execution_inventory.yaml",
    )
    assert first.model_dump(mode="json", exclude={"generated_at"}) == second.model_dump(
        mode="json",
        exclude={"generated_at"},
    )
    assert all(
        item.required_dataset_protocol["mechanism_ids"]
        == sorted(set(item.required_dataset_protocol["mechanism_ids"]))
        for item in first.requirements
    )


def test_requirement_schema_rejects_generic_and_unsafe_training() -> None:
    base = {
        "paper_id": "fixture:paper",
        "paper_specific_mechanism": "loss.quality.correlation",
        "paper_specific_mechanism_ids": ["loss.quality.correlation"],
        "execution_route": "blocked_runtime",
        "compatible_with_yolo26": True,
        "training_candidate_allowed": False,
        "exact_blocker": "payload_missing",
        "recovery_action": "provide payload",
        "current_disposition": "blocked_runtime",
        "execution_fingerprint": "b" * 64,
    }
    with pytest.raises(ValueError, match="generic mechanism"):
        PaperExecutionRequirement.model_validate(
            {
                **base,
                "paper_specific_mechanism": "domain_adaptation.general",
                "paper_specific_mechanism_ids": ["domain_adaptation.general"],
            }
        )
    with pytest.raises(ValueError, match="training candidate requires an adapter"):
        PaperExecutionRequirement.model_validate(
            {
                **base,
                "execution_route": "training",
                "training_candidate_allowed": True,
                "exact_blocker": None,
            }
        )
