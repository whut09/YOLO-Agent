from __future__ import annotations

import hashlib
from pathlib import Path

from yolo_agent.components.contracts import load_contracts
from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
)
from yolo_agent.components.maturity_registry_schemas import ComponentEvidenceOverlay
from yolo_agent.research.production_pipeline import ResearchProductionPipeline
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
