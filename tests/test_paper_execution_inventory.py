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


def _inventory() -> PaperExecutionInventory:
    spec = PaperExecutionSpec(
        paper_id="paper|one",
        profile_id="profile-one",
        title="Method | One",
        source_locations=["paper_record"],
        paper_specific_mechanism_ids=["loss.quality.correlation"],
        runtime_ready_adapters=["loss.quality.correlation"],
        recipe_ids=["quality-recipe"],
        execution_fingerprint=hashlib.sha256(b"paper-one").hexdigest(),
        current_disposition="runtime_ready",
        disposition_reason="runtime evidence is available",
    )
    return PaperExecutionInventory(
        source_method_coverage_hash="m" * 64,
        source_maturity_hash="r" * 64,
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
