from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.components.maturity_registry_schemas import (
    ComponentEvidenceOverlay,
    ComponentMaturityRegistryDocument,
    ComponentOverlayResolution,
)


def _artifact(tmp_path: Path, component_id: str = "sampling.small") -> ComponentMaturityArtifact:
    path = tmp_path / "runtime.yaml"
    path.write_text("runtime: ready\n", encoding="utf-8")
    return ComponentMaturityArtifact(
        component_id=component_id,
        target_maturity="runtime_integrated",
        artifact_type="runtime_payload",
        artifact_path=path,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status="passed",
        producer="pytest",
        protocol_hash="protocol-1",
    )


def test_overlay_identity_and_evidence_hash_are_stable(tmp_path: Path) -> None:
    values = {
        "component_id": "sampling.small",
        "adapter_hash": "a" * 64,
        "code_commit": "commit-1",
        "ultralytics_version": "8.4.87",
        "protocol_hash": "protocol-1",
        "artifacts": [_artifact(tmp_path)],
    }
    first = ComponentEvidenceOverlay(**values)
    second = ComponentEvidenceOverlay(**values)

    assert first.identity_key == second.identity_key
    assert first.evidence_hash == second.evidence_hash
    changed = second.model_copy(update={"protocol_hash": "protocol-2"})
    assert changed.identity_key != first.identity_key


def test_overlay_rejects_artifacts_from_another_component(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must match component_id"):
        ComponentEvidenceOverlay(
            component_id="sampling.small",
            adapter_hash="a" * 64,
            code_commit="commit-1",
            ultralytics_version="8.4.87",
            protocol_hash="protocol-1",
            artifacts=[_artifact(tmp_path, "head.p2")],
        )


def test_registry_document_and_resolution_round_trip(tmp_path: Path) -> None:
    overlay = ComponentEvidenceOverlay(
        component_id="sampling.small",
        adapter_hash="a" * 64,
        code_commit="commit-1",
        ultralytics_version="8.4.87",
        protocol_hash="protocol-1",
        artifacts=[_artifact(tmp_path)],
    )
    document = ComponentMaturityRegistryDocument(overlays=[overlay])
    path = document.to_yaml(tmp_path / "registry.yaml")

    loaded = ComponentMaturityRegistryDocument.from_yaml(path)
    assert loaded.overlays[0].evidence_hash == overlay.evidence_hash

    resolution = ComponentOverlayResolution(
        status="applied",
        component_id="sampling.small",
        source_maturity="adapter_implemented",
        effective_maturity="runtime_integrated",
        overlay_identity_key=overlay.identity_key,
        applied_artifact_hashes=[overlay.artifacts[0].artifact_sha256],
    )
    assert resolution.effective_maturity == "runtime_integrated"
