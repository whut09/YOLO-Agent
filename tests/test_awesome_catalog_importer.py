"""Offline tests for Awesome-object-detection catalog import."""

from __future__ import annotations

import json
from pathlib import Path

from yolo_agent.research.awesome_catalog_importer import AwesomeCatalogImporter, import_awesome_catalog
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.schemas import PaperRecord


def _write_catalog(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "data" / "papers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "paper_id": "arxiv:2501.00001",
        "title": "Small Object Detection Fixture",
        "year": 2025,
        "publication": "CVPR",
        "category": "Small, Aerial, and Oriented Detection",
        "paper_url": "https://arxiv.org/abs/2501.00001",
        "official_code_url": "https://github.com/example/small-object",
        "summary": "Improves small object recall with multi-scale features.",
        "task_families": ["object_detection", "small_object_detection"],
        "detector_family": "one_stage",
        "component_ids": ["multi_scale_features", "small_object_sampling"],
        "applicability": "recipe_idea_only",
        "harness_hints": ["Try this only when AP_small is low."],
    }
    row.update(overrides)
    return row


def test_import_maps_catalog_fields_and_preserves_paper_prior(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "awesome", [_row(extra_field="audit-me", note_path="docs/paper.md")])
    result = import_awesome_catalog(catalog.parents[1], registry_root=tmp_path / "research", source_commit="commit-1")

    assert result.imported_count == 1
    assert result.would_import_count == 1
    assert result.conflict_count == 0
    assert result.unknown_fields["arxiv:2501.00001"] == ["extra_field"]
    paper = PaperRegistry(tmp_path / "research").get("arxiv:2501.00001")
    assert paper is not None
    assert paper.source == "awesome_object_detection"
    assert paper.evidence_level == "paper_prior"
    assert paper.abstract.startswith("Improves small object")
    assert paper.code_license == "unknown"
    assert "small_object_detection" in paper.task_families
    assert paper.component_ids == ["multi_scale_features", "small_object_sampling"]
    assert paper.component_categories == ["feature_pyramid", "sampling", "slicing"]
    assert paper.provenance is not None
    assert paper.provenance.source_commit == "commit-1"
    assert paper.provenance.original_category == "Small, Aerial, and Oriented Detection"
    assert paper.provenance.original_harness_hints == ["Try this only when AP_small is low."]
    assert paper.provenance.original_note_path == "docs/paper.md"


def test_import_is_idempotent_for_same_source_hash(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "awesome", [_row()])
    root = tmp_path / "research"
    first = import_awesome_catalog(catalog, registry_root=root, source_commit="commit-1")
    second = import_awesome_catalog(catalog, registry_root=root, source_commit="commit-1")

    assert first.imported_count == 1
    assert second.imported_count == 0
    assert any(item["reason"] == "unchanged" for item in second.skipped)
    assert len(PaperRegistry(root).list()) == 1


def test_same_source_update_retains_prior_provenance_history(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "awesome", [_row()])
    root = tmp_path / "research"
    import_awesome_catalog(catalog, registry_root=root, source_commit="commit-1")
    _write_catalog(tmp_path / "awesome", [_row(summary="Updated summary", year=2026)])
    result = import_awesome_catalog(catalog, registry_root=root, source_commit="commit-2")

    assert result.imported_count == 1
    paper = PaperRegistry(root).get("arxiv:2501.00001")
    assert paper is not None and paper.provenance is not None
    assert paper.abstract == "Updated summary"
    assert len(paper.provenance.history) == 1
    assert paper.provenance.history[0]["source_commit"] == "commit-1"


def test_import_does_not_overwrite_record_owned_by_another_source(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "awesome", [_row()])
    root = tmp_path / "research"
    registry = PaperRegistry(root)
    registry.add(PaperRecord(
        paper_id="arxiv:2501.00001",
        title="Manually curated record",
        year=2025,
        source="manual",
    ))

    result = AwesomeCatalogImporter(registry).import_source(catalog, source_commit="commit-1")

    assert result.imported_count == 0
    assert result.conflict_count == 1
    preserved = PaperRegistry(root).get("arxiv:2501.00001")
    assert preserved is not None
    assert preserved.title == "Manually curated record"


def test_import_does_not_replace_other_source_with_same_title(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "awesome", [_row(paper_id="awesome-id")])
    root = tmp_path / "research"
    registry = PaperRegistry(root)
    registry.add(PaperRecord(
        paper_id="manual-id",
        title="Small Object Detection Fixture",
        year=2025,
        source="manual",
    ))

    result = AwesomeCatalogImporter(registry).import_source(catalog, source_commit="commit-1")

    assert result.conflict_count == 1
    assert PaperRegistry(root).get("manual-id") is not None
    assert PaperRegistry(root).get("awesome-id") is None


def test_dry_run_does_not_write_registry(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "awesome", [_row()])
    root = tmp_path / "research"

    result = import_awesome_catalog(catalog, registry_root=root, dry_run=True, source_commit="commit-1")

    assert result.dry_run is True
    assert result.would_import_count == 1
    assert result.imported_count == 0
    assert result.records[0].paper_id == "arxiv:2501.00001"
    assert not (root / "papers.jsonl").exists()


def test_direct_adapter_candidate_remains_a_non_executable_paper_prior(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "awesome", [_row(applicability="direct_adapter_candidate")])

    result = import_awesome_catalog(catalog, registry_root=None, dry_run=True, source_commit="commit-1")

    paper = result.records[0]
    assert paper.applicability == "direct_adapter_candidate"
    assert paper.evidence_level == "paper_prior"
    assert paper.benchmarks == []
    assert paper.claimed_effects == []


def test_malformed_rows_are_skipped_and_valid_rows_import(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "awesome", [
        {"paper_id": "missing-title", "year": 2025},
        _row(paper_id="valid-paper"),
        "not a mapping",  # type: ignore[list-item]
    ])

    result = import_awesome_catalog(catalog, registry_root=tmp_path / "research", source_commit="commit-1")

    assert result.imported_count == 1
    assert result.skipped_count == 2
    assert {item["reason"] for item in result.skipped} >= {"record_not_mapping", "missing_fields:title"}


def test_file_input_and_catalog_hash_are_supported(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "awesome", [_row()])
    result = import_awesome_catalog(catalog, registry_root=None, source_commit="commit-1")

    assert result.source_path == catalog.resolve().as_posix()
    assert len(result.catalog_hash) == 64
    assert result.source_commit == "commit-1"
