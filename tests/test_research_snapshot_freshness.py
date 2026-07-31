from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from yolo_agent.research.snapshot import (
    ResearchSnapshot,
    ResearchSnapshotArtifact,
    preflight_research_snapshot,
    research_snapshot_hash,
)


def _base_payload(schema_version: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "paper_intelligence": "available",
        "unavailable_reason": None,
        "papers_version": "papers-v1",
        "component_registry_version": "components-v1",
        "recipe_registry_version": "recipes-v1",
        "source_repository": "awesome_object_detection",
        "source_commit": "commit-1",
        "source_catalog_hash": "catalog-v1",
        "importer_version": "importer-v1",
        "alias_resolution_version": "aliases-v1",
        "coverage_version": "coverage-v1",
        "paper_evidence_version": "paper-evidence-v1",
        "classifications_version": "classifications-v1",
        "extractions_version": "extractions-v1",
        "compatibility_version": "compatibility-v1",
        "reproduction_queue_version": "queue-v1",
        "paper_count": 1,
        "component_count": 1,
        "recipe_count": 1,
        "maturity_summary": {
            "metadata_only": 1,
            "recipe_idea_only": 0,
            "adapter_implemented": 0,
            "runtime_integrated": 0,
            "unit_tested": 0,
            "smoke_passed": 0,
            "gpu_certified": 0,
            "pilot_reproduced": 0,
            "full_reproduced": 0,
            "confirmed_multi_seed": 0,
        },
    }
    return payload


def test_legacy_snapshot_without_method_coverage_is_stale() -> None:
    payload = _base_payload("research_snapshot.v6")

    snapshot = ResearchSnapshot(
        **payload,
        snapshot_hash=research_snapshot_hash(payload),
    )

    assert snapshot.snapshot_status == "stale_snapshot"
    assert "paper_method_coverage_missing" in snapshot.stale_reasons
    assert "legacy_snapshot_schema:research_snapshot.v6" in snapshot.stale_reasons


def test_v7_snapshot_requires_both_runtime_research_artifacts() -> None:
    payload = _base_payload("research_snapshot.v7")
    payload.update({
        "paper_method_coverage_version": "method-coverage-v1",
        "effective_maturity_version": "effective-maturity-v1",
    })
    artifact = ResearchSnapshotArtifact(
        name="fixture",
        path="fixture.yaml",
        sha256="0" * 64,
        size_bytes=1,
    )

    snapshot = ResearchSnapshot(
        **payload,
        snapshot_hash=research_snapshot_hash(payload),
        artifacts={
            "paper_method_coverage": artifact.model_copy(
                update={"name": "paper_method_coverage"}
            ),
            "effective_component_maturity": artifact.model_copy(
                update={"name": "effective_component_maturity"}
            ),
        },
    )

    assert snapshot.snapshot_status == "current"
    assert snapshot.stale_reasons == []


def test_preflight_rejects_stale_snapshot_and_accepts_current_snapshot(
    tmp_path: Path,
) -> None:
    stale_payload = _base_payload("research_snapshot.v6")
    stale = ResearchSnapshot(
        **stale_payload,
        snapshot_hash=research_snapshot_hash(stale_payload),
    )
    stale_dir = tmp_path / "snapshots" / stale.snapshot_hash
    stale_dir.mkdir(parents=True)
    stale.to_yaml(stale_dir / "snapshot.yaml")
    _write_pointer(tmp_path, stale, stale_dir)

    stale_result = preflight_research_snapshot(tmp_path)

    assert stale_result.ok is False
    assert stale_result.status == "stale_snapshot"
    assert "paper_method_coverage_missing" in stale_result.reasons

    current_payload = _base_payload("research_snapshot.v7")
    current_payload.update({
        "paper_method_coverage_version": "method-coverage-v1",
        "effective_maturity_version": "effective-maturity-v1",
    })
    current_dir = tmp_path / "snapshots" / "current"
    artifacts: dict[str, ResearchSnapshotArtifact] = {}
    for name in ("paper_method_coverage", "effective_component_maturity"):
        path = current_dir / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"name: {name}\n", encoding="utf-8")
        artifacts[name] = ResearchSnapshotArtifact(
            name=name,
            path=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )
    current = ResearchSnapshot(
        **current_payload,
        snapshot_hash=research_snapshot_hash(current_payload),
        artifacts=artifacts,
    )
    current.to_yaml(current_dir / "snapshot.yaml")
    _write_pointer(tmp_path, current, current_dir)

    current_result = preflight_research_snapshot(tmp_path)

    assert current_result.ok is True
    assert current_result.binding is not None
    assert current_result.binding.research_snapshot_hash == current.snapshot_hash


def _write_pointer(root: Path, snapshot: ResearchSnapshot, snapshot_dir: Path) -> None:
    (root / "latest_snapshot.yaml").write_text(
        yaml.safe_dump({
            "schema_version": "research_snapshot_pointer.v1",
            "snapshot_hash": snapshot.snapshot_hash,
            "snapshot_path": snapshot_dir.resolve().as_posix(),
        }),
        encoding="utf-8",
    )
