from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.components.maturity_registry import ComponentMaturityRegistry
from yolo_agent.components.maturity_registry_schemas import ComponentEvidenceOverlay


COMPONENT_ID = "sampling.small"


def _contract() -> ComponentContract:
    return ComponentContract(
        component_id=COMPONENT_ID,
        display_name="Small sampling",
        category="sampling",
        implementation_path="yolo_agent.components.adapters.dummy",
        adapter_class="DummyAdapter",
        maturity="adapter_implemented",
    )


def _artifact(
    tmp_path: Path,
    target: str,
    *,
    status: str = "passed",
    content: str | None = None,
) -> ComponentMaturityArtifact:
    artifact_types = {
        "runtime_integrated": "runtime_payload",
        "unit_tested": "unit_test_report",
        "smoke_passed": "smoke_report",
    }
    path = tmp_path / f"{target}-{status}.yaml"
    path.write_text(content or f"{target}: {status}\n", encoding="utf-8")
    return ComponentMaturityArtifact(
        component_id=COMPONENT_ID,
        target_maturity=target,
        artifact_type=artifact_types[target],
        artifact_path=path,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=status,
        producer="pytest",
        protocol_hash="protocol-1",
    )


def _overlay(tmp_path: Path, *, component_id: str = COMPONENT_ID) -> ComponentEvidenceOverlay:
    artifact = _artifact(tmp_path, "runtime_integrated")
    if component_id != COMPONENT_ID:
        artifact = artifact.model_copy(update={"component_id": component_id})
    return ComponentEvidenceOverlay(
        component_id=component_id,
        adapter_hash="a" * 64,
        code_commit="commit-1",
        ultralytics_version="8.4.87",
        protocol_hash="protocol-1",
        artifacts=[artifact],
    )


def test_registry_write_is_atomic_and_repeated_upsert_is_idempotent(
    tmp_path: Path,
) -> None:
    registry = ComponentMaturityRegistry(tmp_path / "registry.yaml")
    overlay = _overlay(tmp_path)

    first = registry.upsert(overlay)
    content = registry.path.read_bytes()
    second = registry.upsert(overlay)

    assert first.evidence_hash == second.evidence_hash
    assert registry.path.read_bytes() == content
    assert len(registry.load().overlays) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_file_lock_preserves_concurrent_component_updates(tmp_path: Path) -> None:
    registry = ComponentMaturityRegistry(tmp_path / "registry.yaml")
    for directory in (tmp_path / "first", tmp_path / "second"):
        directory.mkdir(exist_ok=True)
    overlays = [
        _overlay(tmp_path / "first", component_id="sampling.small"),
        _overlay(tmp_path / "second", component_id="head.p2"),
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(registry.upsert, overlays))

    assert {item.component_id for item in registry.load().overlays} == {
        "sampling.small",
        "head.p2",
    }


def test_valid_adjacent_artifacts_promote_and_failed_artifact_is_retained(
    tmp_path: Path,
) -> None:
    registry = ComponentMaturityRegistry(tmp_path / "registry.yaml")
    registry.upsert(
        _overlay(tmp_path).model_copy(
            update={
                "artifacts": [
                    _artifact(tmp_path, "runtime_integrated"),
                    _artifact(tmp_path, "unit_tested"),
                    _artifact(tmp_path, "smoke_passed", status="failed"),
                ]
            }
        )
    )

    effective, resolution = registry.apply(
        _contract(),
        adapter_hash="a" * 64,
        ultralytics_version="8.4.87",
        protocol_hash="protocol-1",
    )

    assert effective.maturity == "unit_tested"
    assert [item.status for item in effective.maturity_artifacts] == [
        "passed",
        "passed",
        "failed",
    ]
    assert len(resolution.applied_artifact_hashes) == 2
    assert len(resolution.retained_artifact_hashes) == 1


def test_invalid_artifact_hash_downgrades_to_last_valid_stage(tmp_path: Path) -> None:
    registry = ComponentMaturityRegistry(tmp_path / "registry.yaml")
    runtime = _artifact(tmp_path, "runtime_integrated")
    unit = _artifact(tmp_path, "unit_tested")
    smoke = _artifact(tmp_path, "smoke_passed")
    registry.upsert(
        _overlay(tmp_path).model_copy(
            update={"artifacts": [runtime, unit, smoke]}
        )
    )
    unit.artifact_path.write_text("tampered: true\n", encoding="utf-8")

    effective, resolution = registry.apply(
        _contract(),
        adapter_hash="a" * 64,
        ultralytics_version="8.4.87",
        protocol_hash="protocol-1",
    )

    assert effective.maturity == "runtime_integrated"
    assert not effective.can_execute
    assert any("artifact_invalid:unit_tested" in item for item in resolution.invalid_artifacts)
    assert smoke.artifact_sha256 in resolution.retained_artifact_hashes


def test_adapter_runtime_or_protocol_mismatch_does_not_apply_overlay(
    tmp_path: Path,
) -> None:
    registry = ComponentMaturityRegistry(tmp_path / "registry.yaml")
    registry.upsert(_overlay(tmp_path))

    for values in (
        {
            "adapter_hash": "b" * 64,
            "ultralytics_version": "8.4.87",
            "protocol_hash": "protocol-1",
        },
        {
            "adapter_hash": "a" * 64,
            "ultralytics_version": "9.0.0",
            "protocol_hash": "protocol-1",
        },
        {
            "adapter_hash": "a" * 64,
            "ultralytics_version": "8.4.87",
            "protocol_hash": "protocol-2",
        },
    ):
        effective, resolution = registry.apply(_contract(), **values)
        assert effective.maturity == "adapter_implemented"
        assert resolution.status == "no_match"
