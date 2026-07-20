"""Tests for paper-component coverage and adapter accounting."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.component_coverage import ComponentCoverageAnalyzer, ComponentCoverageReport
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.production_pipeline import ResearchProductionPipeline
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.research.snapshot import load_research_snapshot


def _paper(paper_id: str, *components: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=2025,
        component_ids=list(components),
        source="test",
    )


def test_coverage_distinguishes_adapters_priors_incompatible_and_unknown() -> None:
    analyzer = ComponentCoverageAnalyzer(ComponentAliasResolver.from_yaml())
    report = analyzer.analyze_papers([
        _paper(
            "paper-1",
            "small_object_sampling",
            "p2_head",
            "distillation",
            "open_vocabulary_detection",
            "unknown_component",
        ),
    ])

    assert report.total_paper_components == 5
    assert report.resolved == 4
    assert report.unresolved == 1
    assert report.executable == 3
    assert report.adapter_required == 0
    assert report.incompatible == 1
    assert report.real_adapter_components == [
        "distillation.yolo26_teacher_student",
        "head.p2_small_object",
        "sampling.small_object",
    ]
    assert report.paper_prior_only_components == ["detection_head.open_vocabulary"]
    assert report.unresolved_components == ["unknown_component"]


def test_canonical_component_retains_multiple_paper_sources() -> None:
    analyzer = ComponentCoverageAnalyzer(ComponentAliasResolver.from_yaml())
    report = analyzer.analyze_papers([
        _paper("paper-a", "small_object_sampling"),
        _paper("paper-b", "small-object-sampling"),
    ])

    assert report.total_paper_components == 2
    assert report.resolved == 2
    assert report.canonical_component_count == 1
    assert report.executable == 1
    assert report.canonical_paper_sources["sampling.small_object"] == ["paper-a", "paper-b"]


def test_coverage_report_round_trips_yaml(tmp_path: Path) -> None:
    analyzer = ComponentCoverageAnalyzer(ComponentAliasResolver.from_yaml())
    report = analyzer.analyze_papers([_paper("paper-1", "feature_pyramid", "missing")])
    path = analyzer.write_report(tmp_path / "component_coverage_report.yaml", report)

    loaded = ComponentCoverageReport.from_yaml(path)

    assert loaded.model_dump(mode="json") == report.model_dump(mode="json")


def test_production_pipeline_freezes_alias_and_coverage_without_promoting_contract(tmp_path: Path) -> None:
    root = tmp_path / "research"
    PaperRegistry(root).add(_paper("paper-1", "small_object_sampling", "unmapped_component"))

    result = ResearchProductionPipeline(root).run()

    assert result.status == "completed", result.errors
    coverage_path = root / "production" / "component_coverage_report.yaml"
    alias_path = root / "production" / "component_alias_resolutions.yaml"
    assert coverage_path.is_file()
    assert alias_path.is_file()
    coverage = yaml.safe_load(coverage_path.read_text(encoding="utf-8"))
    assert coverage["executable"] == 1
    assert coverage["unresolved"] == 1

    contracts = yaml.safe_load((root / "production" / "component_contracts.yaml").read_text(encoding="utf-8"))
    assert contracts["components"]["sampling.small_object"]["maturity"] == "metadata_only"
    assert contracts["components"]["unmapped_component"]["maturity"] == "metadata_only"
    queue = yaml.safe_load((root / "production" / "reproduction_queue.yaml").read_text(encoding="utf-8"))
    assert {item["component_id"] for item in queue["items"]} == {
        "sampling.small_object",
        "unmapped_component",
    }

    snapshot, snapshot_dir = load_research_snapshot(root)  # type: ignore[misc]
    assert snapshot.alias_resolution_version != "not_available"
    assert snapshot.coverage_version != "not_available"
    assert "component_alias_resolutions" in snapshot.artifacts
    assert "component_coverage" in snapshot.artifacts
    assert snapshot.verify(snapshot_dir) == []
