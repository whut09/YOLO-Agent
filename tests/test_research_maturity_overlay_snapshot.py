from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from yolo_agent.components.contracts import load_contracts
from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
)
from yolo_agent.components.maturity_registry_schemas import ComponentEvidenceOverlay
from yolo_agent.research.production_pipeline import ResearchProductionPipeline
from yolo_agent.research.maturity_snapshot import EffectiveComponentMaturityManifest
from yolo_agent.research.snapshot import load_research_snapshot
from yolo_agent.resources import ResourcePaths


COMPONENT_ID = "sampling.small_object"


def _artifact(tmp_path: Path, target: str) -> ComponentMaturityArtifact:
    artifact_type = {
        "runtime_integrated": "runtime_payload",
        "unit_tested": "unit_test_report",
        "smoke_passed": "smoke_report",
    }[target]
    path = tmp_path / f"{target}.yaml"
    path.write_text(f"{target}: passed\n", encoding="utf-8")
    return ComponentMaturityArtifact(
        component_id=COMPONENT_ID,
        target_maturity=target,
        artifact_type=artifact_type,
        artifact_path=path,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status="passed",
        producer="pytest",
        protocol_hash="protocol-1",
    )


def test_research_snapshot_freezes_effective_overlay_and_evidence(
    tmp_path: Path,
) -> None:
    source_contract = load_contracts(
        ResourcePaths.COMPONENTS_DIR / "sampling" / "small_object_sampling.yaml"
    )[0]
    source_artifacts = [
        _artifact(tmp_path, "runtime_integrated"),
        _artifact(tmp_path, "unit_tested"),
        _artifact(tmp_path, "smoke_passed"),
    ]
    registry = ComponentMaturityRegistry(tmp_path / "maturity-registry.yaml")
    registry.upsert(
        ComponentEvidenceOverlay(
            component_id=COMPONENT_ID,
            adapter_hash=adapter_source_hash(source_contract),
            code_commit="commit-1",
            ultralytics_version="8.4.87",
            protocol_hash="protocol-1",
            artifacts=source_artifacts,
        )
    )
    research_root = tmp_path / "research"

    result = ResearchProductionPipeline(
        research_root,
        maturity_registry=registry,
        maturity_protocol_hash="protocol-1",
        maturity_ultralytics_version="8.4.87",
    ).run(include_local_implementations=True)

    assert result.status == "completed"
    assert result.maturity_summary.smoke_passed >= 1
    loaded = load_research_snapshot(research_root, result.snapshot_path)
    assert loaded is not None
    snapshot, snapshot_dir = loaded
    assert not snapshot.verify(snapshot_dir)
    maturity_artifacts = [
        item
        for name, item in snapshot.artifacts.items()
        if name.startswith(f"component_maturity_{COMPONENT_ID}")
    ]
    assert len(maturity_artifacts) == 3
    effective_manifest = EffectiveComponentMaturityManifest.from_yaml(
        snapshot_dir / "effective_component_maturity.yaml"
    )
    frozen_identity = effective_manifest.by_component()[COMPONENT_ID]
    assert frozen_identity.adapter_hash == adapter_source_hash(source_contract)
    assert frozen_identity.ultralytics_version == "8.4.87"
    assert frozen_identity.protocol_hash == "protocol-1"
    assert frozen_identity.effective_maturity == "smoke_passed"
    assert frozen_identity.runtime_execution_ready is True
    assert {item.snapshot_artifact_name for item in frozen_identity.artifacts} == {
        name for name in snapshot.artifacts if name.startswith(f"component_maturity_{COMPONENT_ID}")
    }

    frozen_contract = next(
        item
        for item in load_contracts(snapshot_dir / "component_contracts.yaml")
        if item.component_id == COMPONENT_ID
    )
    assert frozen_contract.maturity == "smoke_passed"
    assert frozen_contract.can_execute
    assert all(
        "component_maturity_evidence" in str(item.artifact_path)
        for item in frozen_contract.maturity_artifacts
    )

    source_artifacts[0].artifact_path.write_text("tampered: true\n", encoding="utf-8")
    unchanged = load_research_snapshot(research_root, result.snapshot_path)
    assert unchanged is not None
    assert unchanged[0].snapshot_hash == result.snapshot_hash
    assert not unchanged[0].verify(unchanged[1])
    assert frozen_contract.can_execute


def test_overlay_or_adapter_change_creates_new_snapshot_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_contract = load_contracts(
        ResourcePaths.COMPONENTS_DIR / "sampling" / "small_object_sampling.yaml"
    )[0]
    artifacts = [
        _artifact(tmp_path, "runtime_integrated"),
        _artifact(tmp_path, "unit_tested"),
        _artifact(tmp_path, "smoke_passed"),
    ]
    registry = ComponentMaturityRegistry(tmp_path / "maturity-registry.yaml")
    identity = {
        "component_id": COMPONENT_ID,
        "adapter_hash": adapter_source_hash(source_contract),
        "code_commit": "commit-1",
        "ultralytics_version": "8.4.87",
        "protocol_hash": "protocol-1",
    }
    registry.upsert(ComponentEvidenceOverlay(**identity, artifacts=artifacts))
    research_root = tmp_path / "research"
    pipeline = ResearchProductionPipeline(
        research_root,
        maturity_registry=registry,
        maturity_protocol_hash="protocol-1",
        maturity_ultralytics_version="8.4.87",
    )
    first = pipeline.run(include_local_implementations=True)

    failed_path = tmp_path / "gpu-certified-failed.yaml"
    failed_path.write_text("status: failed\n", encoding="utf-8")
    failed = ComponentMaturityArtifact(
        component_id=COMPONENT_ID,
        target_maturity="gpu_certified",
        artifact_type="gpu_certification_report",
        artifact_path=failed_path,
        artifact_sha256=hashlib.sha256(failed_path.read_bytes()).hexdigest(),
        status="failed",
        producer="pytest",
        protocol_hash="protocol-1",
    )
    registry.upsert(ComponentEvidenceOverlay(**identity, artifacts=[failed]))
    second = pipeline.run(include_local_implementations=True)

    assert first.snapshot_hash != second.snapshot_hash

    monkeypatch.setattr(
        "yolo_agent.research.production_pipeline.adapter_source_hash",
        lambda _: "f" * 64,
    )
    third = pipeline.run(include_local_implementations=True)

    assert second.snapshot_hash != third.snapshot_hash
