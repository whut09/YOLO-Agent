"""Offline tests for Awesome catalog synchronization and frozen snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yolo_agent.cli import main
from yolo_agent.agents.auto_optimization_loop import _prepare_child_training_context
from yolo_agent.core.artifact_manifest import sha256_file
from yolo_agent.core.run_context import RunContext
from yolo_agent.research.awesome_snapshot_builder import AwesomeSnapshotBuilder
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.research.snapshot import bind_research_snapshot, load_research_snapshot


def _write_catalog(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "data" / "papers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _paper_row(paper_id: str = "paper-1", summary: str = "Small object detection prior") -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "year": 2025,
        "category": "Small, Aerial, and Oriented Detection",
        "paper_url": f"https://example.invalid/{paper_id}",
        "summary": summary,
        "task_families": ["object_detection"],
        "detector_family": "one_stage",
        "component_ids": ["small_object_sampling"],
        "applicability": "recipe_idea_only",
        "harness_hints": ["Use when AP_small is low."],
    }


def test_same_source_commit_and_catalog_hash_produce_stable_snapshot(tmp_path: Path) -> None:
    source = _write_catalog(tmp_path / "awesome", [_paper_row()])
    root = tmp_path / "research"
    builder = AwesomeSnapshotBuilder(root)

    first = builder.build(source=source, source_commit="commit-1")
    second = builder.build(source=source, source_commit="commit-1")

    assert first.status == "completed", first.errors
    assert second.status == "completed", second.errors
    assert first.snapshot_hash == second.snapshot_hash
    assert first.import_result is not None
    assert first.import_result.catalog_record_count == 1
    loaded = load_research_snapshot(root)
    assert loaded is not None
    snapshot, _ = loaded
    assert snapshot.source_commit == "commit-1"
    assert snapshot.source_catalog_hash == first.import_result.catalog_hash
    assert snapshot.importer_version == "awesome_object_detection.v1"
    assert snapshot.component_count >= 1
    assert snapshot.recipe_count >= 1
    assert snapshot.maturity_summary.metadata_only >= 1


def test_same_catalog_is_stable_across_independent_research_roots(tmp_path: Path) -> None:
    source = _write_catalog(tmp_path / "awesome", [_paper_row()])

    first = AwesomeSnapshotBuilder(tmp_path / "research-a").build(
        source=source,
        source_commit="commit-1",
    )
    second = AwesomeSnapshotBuilder(tmp_path / "research-b").build(
        source=source,
        source_commit="commit-1",
    )

    assert first.status == "completed", first.errors
    assert second.status == "completed", second.errors
    assert first.snapshot_hash == second.snapshot_hash


def test_frozen_artifact_hashes_verify_physical_files(tmp_path: Path) -> None:
    source = _write_catalog(tmp_path / "awesome", [_paper_row()])
    root = tmp_path / "research"
    built = AwesomeSnapshotBuilder(root).build(source=source, source_commit="commit-1")
    snapshot, snapshot_dir = load_research_snapshot(root)  # type: ignore[misc]

    assert built.status == "completed", built.errors
    assert snapshot.verify(snapshot_dir) == []
    for artifact in snapshot.artifacts.values():
        assert sha256_file(snapshot_dir / artifact.path) == artifact.sha256


def test_catalog_content_change_changes_snapshot_hash(tmp_path: Path) -> None:
    source = _write_catalog(tmp_path / "awesome", [_paper_row()])
    root = tmp_path / "research"
    first = AwesomeSnapshotBuilder(root).build(source=source, source_commit="commit-1")
    _write_catalog(tmp_path / "awesome", [_paper_row(summary="Changed paper prior")])
    second = AwesomeSnapshotBuilder(root).build(source=source, source_commit="commit-1")

    assert first.snapshot_hash != second.snapshot_hash
    assert first.import_result is not None and second.import_result is not None
    assert first.import_result.catalog_hash != second.import_result.catalog_hash


def test_source_commit_change_changes_snapshot_hash(tmp_path: Path) -> None:
    source = _write_catalog(tmp_path / "awesome", [_paper_row()])
    root = tmp_path / "research"
    first = AwesomeSnapshotBuilder(root).build(source=source, source_commit="commit-1")
    second = AwesomeSnapshotBuilder(root).build(source=source, source_commit="commit-2")

    assert first.snapshot_hash != second.snapshot_hash
    assert load_research_snapshot(root) is not None
    snapshot, _ = load_research_snapshot(root)  # type: ignore[misc]
    assert snapshot.source_commit == "commit-2"


def test_empty_catalog_is_explicitly_unavailable_and_rules_can_continue(tmp_path: Path) -> None:
    source = _write_catalog(tmp_path / "awesome", [])
    root = tmp_path / "research"

    result = AwesomeSnapshotBuilder(root).build(source=source, source_commit="empty-commit")

    assert result.status == "completed", result.errors
    assert result.paper_intelligence == "unavailable"
    assert result.unavailable_reason == "empty_catalog"
    assert result.snapshot_hash
    snapshot, _ = load_research_snapshot(root)  # type: ignore[misc]
    assert snapshot.paper_intelligence == "unavailable"
    assert snapshot.unavailable_reason == "empty_catalog"
    assert PaperRegistry(root).list() == []
    binding = bind_research_snapshot(root, expected_hash=result.snapshot_hash)
    assert binding.paper_intelligence == "unavailable"
    assert binding.unavailable_reason == "empty_catalog"
    assert binding.research_network_allowed is False


def test_dry_run_does_not_modify_registry_or_source_manifest(tmp_path: Path) -> None:
    source = _write_catalog(tmp_path / "awesome", [_paper_row()])
    root = tmp_path / "research"

    result = AwesomeSnapshotBuilder(root).import_catalog(source, dry_run=True, source_commit="commit-1")

    assert result.catalog_record_count == 1
    assert result.imported_count == 0
    assert not (root / "papers.jsonl").exists()
    assert not (root / "sources" / "awesome_object_detection.yaml").exists()


def test_snapshot_binding_rejects_mismatch(tmp_path: Path) -> None:
    source = _write_catalog(tmp_path / "awesome", [_paper_row()])
    root = tmp_path / "research"
    result = AwesomeSnapshotBuilder(root).build(source=source, source_commit="commit-1")

    assert result.snapshot_hash
    with pytest.raises(ValueError, match="snapshot changed"):
        bind_research_snapshot(root, expected_hash="wrong-snapshot-hash")


def test_live_registry_changes_do_not_mutate_bound_snapshot(tmp_path: Path) -> None:
    source = _write_catalog(tmp_path / "awesome", [_paper_row()])
    root = tmp_path / "research"
    built = AwesomeSnapshotBuilder(root).build(source=source, source_commit="commit-1")
    snapshot, snapshot_dir = load_research_snapshot(root)  # type: ignore[misc]
    assert snapshot.snapshot_hash == built.snapshot_hash

    PaperRegistry(root).add(PaperRecord(paper_id="later-paper", title="Later paper", year=2026))

    frozen_papers = PaperRegistry(snapshot_dir).list()
    assert [paper.paper_id for paper in frozen_papers] == ["paper-1"]
    binding = bind_research_snapshot(root, expected_hash=built.snapshot_hash, snapshot_path=snapshot_dir)
    assert binding.research_snapshot_hash == built.snapshot_hash


def test_child_training_context_inherits_parent_snapshot_hash(tmp_path: Path) -> None:
    task = tmp_path / "task.yaml"
    data = tmp_path / "data.yaml"
    task.write_text("task_type: detect\n", encoding="utf-8")
    data.write_text("path: .\n", encoding="utf-8")
    parent_context = RunContext(
        run_id="parent",
        run_root=tmp_path / "runs",
        task_path=task,
        data_yaml=data,
        metadata={
            "research_snapshot_hash": "snapshot-hash",
            "research_snapshot_path": (tmp_path / "research" / "snapshots" / "snapshot-hash").as_posix(),
            "research_snapshot_verified": True,
            "training_model": "yolo26n.pt",
        },
    )
    child_context = RunContext(
        run_id="child",
        run_root=tmp_path / "runs",
        task_path=task,
        data_yaml=data,
    )
    parent_context.ensure_dirs()
    child_context.ensure_dirs()

    _prepare_child_training_context(
        SimpleNamespace(context=child_context),  # type: ignore[arg-type]
        SimpleNamespace(context=parent_context),  # type: ignore[arg-type]
        "pilot",
    )

    assert child_context.metadata["research_snapshot_hash"] == "snapshot-hash"
    assert child_context.metadata["research_snapshot_path"].endswith("snapshot-hash")
    assert child_context.metadata["research_snapshot_verified"] is True


def test_training_phase_blocks_catalog_import_and_paper_scout(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    source = _write_catalog(tmp_path / "awesome", [_paper_row()])
    monkeypatch.setenv("YOLO_AGENT_RUNTIME_PHASE", "training")

    with pytest.raises(RuntimeError, match="disabled during training"):
        AwesomeSnapshotBuilder(tmp_path / "research").import_catalog(source)


def test_research_cli_import_and_build_snapshot(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    source = _write_catalog(tmp_path / "awesome", [_paper_row()])
    root = tmp_path / "research"

    assert main(["research", "import-awesome", "--source", str(source), "--root", str(root), "--source-commit", "commit-1"]) == 0
    import_output = capsys.readouterr().out
    assert "Records:" in import_output
    assert (root / "sources" / "awesome_object_detection.yaml").is_file()

    assert main(["research", "build-snapshot", "--root", str(root), "--source", "awesome_object_detection"]) == 0
    build_output = capsys.readouterr().out
    assert "Snapshot:" in build_output
    assert "Paper AI:" in build_output
    assert (root / "latest_snapshot.yaml").is_file()
