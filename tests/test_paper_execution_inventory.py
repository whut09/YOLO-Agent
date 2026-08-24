from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)
from yolo_agent.research.paper_execution_inventory import (
    PaperExecutionInventoryBuilder,
    render_paper_execution_inventory_markdown,
    write_paper_execution_inventory_artifacts,
)
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.executable_coverage import (
    ExecutablePaperCoverageAuditor,
    method_coverage_file_hash,
)


def _profile(paper_id: str) -> PaperMethodProfile:
    return PaperMethodProfile(
        profile_id=f"profile:{paper_id}",
        paper_id=paper_id,
        source_locations=["paper_record"],
    )


def _decision(paper_id: str) -> PaperImplementationDecision:
    return PaperImplementationDecision(
        paper_id=paper_id,
        profile_id=f"profile:{paper_id}",
        decision="new_method_profile",
        reasons=["fixture"],
    )


def test_compatible_method_pairs_preserve_every_requested_paper() -> None:
    report = PaperMethodCoverageReport(
        paper_count=3,
        profile_count=3,
        profiles=[_profile("paper-c"), _profile("paper-a"), _profile("paper-b")],
        decisions=[_decision("paper-a"), _decision("paper-b"), _decision("paper-c")],
    )

    pairs = PaperExecutionInventoryBuilder.compatible_method_pairs(
        report,
        ["paper-c", "paper-a"],
    )

    assert [profile.paper_id for profile, _ in pairs] == ["paper-a", "paper-c"]


def test_compatible_method_pairs_reject_missing_decision() -> None:
    report = PaperMethodCoverageReport(
        paper_count=1,
        profile_count=1,
        profiles=[_profile("paper-a")],
        decisions=[],
    )

    with pytest.raises(ValueError, match="missing decisions: paper-a"):
        PaperExecutionInventoryBuilder.compatible_method_pairs(report, ["paper-a"])


def test_compatible_method_pairs_reject_duplicate_profiles() -> None:
    profile = _profile("paper-a")
    report = PaperMethodCoverageReport(
        paper_count=1,
        profile_count=2,
        profiles=[profile, profile],
        decisions=[_decision("paper-a")],
    )

    with pytest.raises(ValueError, match="duplicate method profiles"):
        PaperExecutionInventoryBuilder.compatible_method_pairs(report, ["paper-a"])


def test_compatible_method_pairs_reject_profile_decision_mismatch() -> None:
    profile = _profile("paper-a")
    decision = _decision("paper-a").model_copy(update={"profile_id": "profile:other"})
    report = PaperMethodCoverageReport(
        paper_count=1,
        profile_count=1,
        profiles=[profile],
        decisions=[decision],
    )

    with pytest.raises(ValueError, match="profile/decision identity mismatch"):
        PaperExecutionInventoryBuilder.compatible_method_pairs(report, ["paper-a"])


def _inventory() -> PaperExecutionInventory:
    spec = PaperExecutionSpec(
        paper_id="paper|one",
        profile_id="profile-one",
        title="Method | One",
        source_locations=["paper_record"],
        canonical_component_ids=["loss.quality.correlation"],
        paper_specific_mechanism_ids=["loss.quality.correlation"],
        runtime_ready_adapters=["loss.quality.correlation"],
        recipe_ids=["quality-recipe"],
        execution_fingerprint=hashlib.sha256(b"paper-one").hexdigest(),
        current_disposition="runtime_ready",
        disposition_reason="runtime evidence is available",
    )
    return PaperExecutionInventory(
        source_method_coverage_hash="a" * 64,
        source_maturity_hash="b" * 64,
        all_paper_count=728,
        compatible_paper_count=1,
        exact_reproduction_candidates=0,
        records=[spec],
    ).with_hash()


def test_inventory_markdown_preserves_counts_and_escapes_cells() -> None:
    markdown = render_paper_execution_inventory_markdown(_inventory())

    assert "YOLO26-compatible papers: 1" in markdown
    assert "Exact reproduction candidates: 0" in markdown
    assert "`paper\\|one`" in markdown
    assert "Method \\| One" in markdown
    assert "runtime_ready" in markdown
    assert "## Disposition Counts" in markdown
    assert "| `deferred_budget` | 0 |" in markdown


def test_inventory_artifacts_roundtrip(tmp_path: Path) -> None:
    inventory = _inventory()
    yaml_path, markdown_path = write_paper_execution_inventory_artifacts(
        inventory,
        yaml_path=tmp_path / "paper_execution_inventory.yaml",
        markdown_path=tmp_path / "paper_execution_inventory.md",
    )

    loaded = PaperExecutionInventory.from_yaml(yaml_path)
    assert loaded.inventory_hash == inventory.inventory_hash
    assert len(loaded.records) == 1
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Paper Execution Inventory"
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_production_inventory_freezes_all_compatible_papers() -> None:
    from yolo_agent.recipes.registry import RecipeRegistry
    from yolo_agent.resources import ResourcePaths
    from yolo_agent.research.method_profiles import PaperMethodCoverageReport
    from yolo_agent.research.paper_registry import PaperRegistry

    method_path = Path("research/production/paper_method_coverage.yaml")
    resolver = ComponentAliasResolver.from_yaml()
    method_coverage = PaperMethodCoverageReport.from_yaml(method_path)
    executable_coverage = ExecutablePaperCoverageAuditor(
        contracts=resolver.contracts,
    ).build(
        method_coverage,
        source_method_coverage_hash=method_coverage_file_hash(method_path),
        source_taxonomy_hash="t" * 64,
    )
    recipes = RecipeRegistry.from_paths(
        [ResourcePaths.RECIPE_BUNDLES, *sorted(ResourcePaths.RECIPES_DIR.glob("*.yaml"))],
        strict=False,
    )

    inventory = PaperExecutionInventoryBuilder().build(
        method_coverage,
        executable_coverage,
        PaperRegistry("research").list(),
        recipes.list(),
        expected_compatible_count=83,
    )

    assert inventory.compatible_paper_count == 83
    assert len(inventory.records) == 83
    assert len({item.paper_id for item in inventory.records}) == 83
    assert all(item.paper_mechanism_resolutions for item in inventory.records)
    assert all(
        not resolution.executable_candidate
        for item in inventory.records
        for resolution in item.paper_mechanism_resolutions
        if not resolution.resolved
    )
    assert inventory.exact_reproduction_candidates == 0
    assert inventory.disposition_counts["runtime_ready"] == 0
    assert inventory.disposition_counts["implementation_request"] == 65
    assert inventory.disposition_counts["blocked_runtime"] == 18
    assert all(
        item.current_disposition != "runtime_ready"
        for item in inventory.records
        if set(item.generic_component_ids)
        & {
            "domain_adaptation.general",
            "distillation.yolo26_teacher_student",
        }
    )
    assert inventory.generic_mechanism_counts == {
        "distillation.yolo26_teacher_student": 32,
        "domain_adaptation.general": 40,
        "quality_alignment.general": 2,
    }
    generic_only = [
        item
        for item in inventory.records
        if item.generic_component_ids and not item.paper_specific_mechanism_ids
    ]
    assert len(generic_only) == 65
    assert sum(len(item.generic_component_ids) for item in generic_only) == 65
    assert {item.current_disposition for item in generic_only} <= {
        "evidence_recovery",
        "implementation_request",
    }
    resolved_mechanisms = {
        mechanism
        for item in inventory.records
        for mechanism in item.paper_specific_mechanism_ids
    }
    assert {
        "feature_distillation",
        "localization_distillation",
        "logits_distillation",
        "pseudo_label_adaptation",
        "relation_distillation",
        "source_free_adaptation",
    }.issubset(resolved_mechanisms)
    multi_mechanism = next(
        item
        for item in inventory.records
        if item.paper_id == "arxiv:2108.07755"
    )
    assert len(multi_mechanism.paper_specific_mechanism_ids) == 2
    assert len({
        resolution.execution_fingerprint
        for resolution in multi_mechanism.paper_mechanism_resolutions
    }) == 2
