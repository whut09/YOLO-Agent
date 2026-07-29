from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from yolo_agent.components.contracts import load_contracts
from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
)
from yolo_agent.components.maturity_registry_schemas import ComponentEvidenceOverlay


COMPONENT_ID = "dummy.overlay"


def _source(path: Path) -> Path:
    payload = {
        "components": {
            COMPONENT_ID: {
                "display_name": "Dummy overlay",
                "category": "augmentation",
                "implementation_path": "yolo_agent.components.adapters.dummy",
                "adapter_class": "DummyAdapter",
                "maturity": "adapter_implemented",
                "fixed_imgsz_compatible": True,
            }
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _artifact(tmp_path: Path, target: str, *, status: str = "passed") -> ComponentMaturityArtifact:
    artifact_type = {
        "runtime_integrated": "runtime_payload",
        "unit_tested": "unit_test_report",
        "smoke_passed": "smoke_report",
    }[target]
    path = tmp_path / f"{target}-{status}.yaml"
    path.write_text(f"{target}: {status}\n", encoding="utf-8")
    return ComponentMaturityArtifact(
        component_id=COMPONENT_ID,
        target_maturity=target,
        artifact_type=artifact_type,
        artifact_path=path,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=status,
        producer="pytest",
        protocol_hash="protocol-1",
    )


def _registry(tmp_path: Path, source: Path) -> ComponentMaturityRegistry:
    source_contract = load_contracts(source)[0]
    registry = ComponentMaturityRegistry(tmp_path / "maturity-registry.yaml")
    registry.upsert(
        ComponentEvidenceOverlay(
            component_id=COMPONENT_ID,
            adapter_hash=adapter_source_hash(source_contract),
            code_commit="commit-1",
            ultralytics_version="8.4.87",
            protocol_hash="protocol-1",
            artifacts=[
                _artifact(tmp_path, "runtime_integrated"),
                _artifact(tmp_path, "unit_tested"),
                _artifact(tmp_path, "smoke_passed"),
            ],
        )
    )
    return registry


def test_contract_loader_merges_matching_valid_overlay_without_editing_source(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "components.yaml")
    original = source.read_bytes()
    registry = _registry(tmp_path, source)

    conservative = load_contracts(source)[0]
    effective = load_contracts(
        source,
        maturity_registry=registry,
        protocol_hash="protocol-1",
        ultralytics_version="8.4.87",
    )[0]

    assert conservative.maturity == "adapter_implemented"
    assert conservative.maturity_artifacts == []
    assert effective.maturity == "smoke_passed"
    assert effective.can_execute
    assert source.read_bytes() == original


def test_contract_loader_rejects_protocol_or_runtime_mismatch(tmp_path: Path) -> None:
    source = _source(tmp_path / "components.yaml")
    registry = _registry(tmp_path, source)

    wrong_protocol = load_contracts(
        source,
        maturity_registry=registry,
        protocol_hash="protocol-2",
        ultralytics_version="8.4.87",
    )[0]
    wrong_runtime = load_contracts(
        source,
        maturity_registry=registry,
        protocol_hash="protocol-1",
        ultralytics_version="9.0.0",
    )[0]

    assert wrong_protocol.maturity == "adapter_implemented"
    assert wrong_runtime.maturity == "adapter_implemented"


def test_contract_loader_downgrades_when_overlay_artifact_hash_is_invalid(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "components.yaml")
    registry = _registry(tmp_path, source)
    document = registry.load()
    unit = next(
        item
        for item in document.overlays[0].artifacts
        if item.target_maturity == "unit_tested"
    )
    unit.artifact_path.write_text("tampered: true\n", encoding="utf-8")

    effective = load_contracts(
        source,
        maturity_registry=registry,
        protocol_hash="protocol-1",
        ultralytics_version="8.4.87",
    )[0]

    assert effective.maturity == "runtime_integrated"
    assert not effective.can_execute
