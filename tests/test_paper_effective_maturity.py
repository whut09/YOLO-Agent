from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import yaml

from yolo_agent.agents.auto_optimization_loop import _load_execution_contracts
from yolo_agent.agents.paper_recipe_materialization.maturity import (
    EffectiveMaturityResolver,
)
from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
    installed_ultralytics_version,
)
from yolo_agent.research.maturity_snapshot import (
    EffectiveComponentMaturityManifest,
    FrozenComponentMaturity,
    FrozenMaturityArtifact,
)
from yolo_agent.resources import ResourcePaths
from tests.paper_materialization_fixtures import contract


def _promoted_contract(tmp_path: Path, *, protocol_hash: str = "cert-protocol"):
    source = contract(maturity="adapter_implemented")
    artifact_path = tmp_path / "runtime.yaml"
    artifact_path.write_text("runtime: passed\n", encoding="utf-8")
    updated = source
    artifact_types = {
        "runtime_integrated": "runtime_payload",
        "unit_tested": "unit_test_report",
        "smoke_passed": "smoke_report",
    }
    for maturity in ("runtime_integrated", "unit_tested", "smoke_passed"):
        artifact = ComponentMaturityArtifact(
            component_id=source.component_id,
            target_maturity=maturity,  # type: ignore[arg-type]
            artifact_type=artifact_types[maturity],  # type: ignore[arg-type]
            artifact_path=artifact_path,
            artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            status="passed",
            producer="offline-test",
            protocol_hash=protocol_hash,
        )
        updated = updated.model_copy(
            update={
                "maturity": maturity,
                "maturity_artifacts": [*updated.maturity_artifacts, artifact],
            }
        )
    return source, updated


def test_valid_machine_overlay_promotes_effective_contract(tmp_path: Path) -> None:
    source, promoted = _promoted_contract(tmp_path)
    registry = ComponentMaturityRegistry(tmp_path / "registry.yaml")
    registry.record_contract(
        promoted,
        adapter_hash=adapter_source_hash(source),
        code_commit="test",
        ultralytics_version="test-ultralytics",
        protocol_hash="cert-protocol",
    )

    resolved = EffectiveMaturityResolver(
        registry,
        ultralytics_version="test-ultralytics",
        certification_protocol_hash="cert-protocol",
    ).resolve({source.component_id: source})[source.component_id]

    assert resolved.valid_for_training is True
    assert resolved.effective_maturity == "smoke_passed"
    assert resolved.evidence_source == "machine_overlay"
    assert resolved.adapter_hash == adapter_source_hash(source)
    assert len(resolved.maturity_artifact_hashes) == 1


def test_wrong_adapter_hash_or_protocol_cannot_authorize_training(
    tmp_path: Path,
) -> None:
    source, promoted = _promoted_contract(tmp_path)
    registry = ComponentMaturityRegistry(tmp_path / "registry.yaml")
    registry.record_contract(
        promoted,
        adapter_hash="0" * 64,
        code_commit="test",
        ultralytics_version="test-ultralytics",
        protocol_hash="other-protocol",
    )

    resolved = EffectiveMaturityResolver(
        registry,
        ultralytics_version="test-ultralytics",
        certification_protocol_hash="cert-protocol",
    ).resolve({source.component_id: source})[source.component_id]

    assert resolved.valid_for_training is False
    assert resolved.effective_maturity == "adapter_implemented"
    assert "valid_maturity_overlay_required:no_match" in resolved.rejection_reasons


def test_frozen_artifact_backed_contract_remains_valid_without_live_registry() -> None:
    frozen = contract(maturity="smoke_passed")

    resolved = EffectiveMaturityResolver().resolve(
        {frozen.component_id: frozen}
    )[frozen.component_id]

    assert resolved.valid_for_training is True
    assert resolved.evidence_source == "frozen_snapshot_artifact"


def test_frozen_runtime_identity_overrides_live_registry_and_fails_on_hash_change(
    tmp_path: Path,
) -> None:
    frozen = contract(maturity="smoke_passed")
    artifact = frozen.maturity_artifacts[0]
    identity = FrozenComponentMaturity(
        component_id=frozen.component_id,
        adapter_hash=adapter_source_hash(frozen),
        code_commit="snapshot-commit",
        ultralytics_version="snapshot-ultralytics",
        protocol_hash="unavailable",
        effective_maturity="smoke_passed",
        runtime_execution_ready=True,
        artifacts=[FrozenMaturityArtifact(
            snapshot_artifact_name="component_maturity_dummy_smoke_passed_0",
            target_maturity=artifact.target_maturity,
            artifact_type=artifact.artifact_type,
            artifact_sha256=artifact.artifact_sha256,
            status=artifact.status,
            mock=artifact.mock,
        )],
    )
    live_registry = ComponentMaturityRegistry(tmp_path / "live-registry.yaml")

    resolved = EffectiveMaturityResolver(
        live_registry,
        ultralytics_version="snapshot-ultralytics",
        frozen_identities={frozen.component_id: identity},
    ).resolve({frozen.component_id: frozen})[frozen.component_id]

    assert resolved.valid_for_training is True
    assert resolved.evidence_source == "frozen_snapshot_artifact"
    assert resolved.overlay_status == "frozen_snapshot"

    changed = identity.model_copy(update={"adapter_hash": "0" * 64})
    rejected = EffectiveMaturityResolver(
        live_registry,
        ultralytics_version="snapshot-ultralytics",
        frozen_identities={frozen.component_id: changed},
    ).resolve({frozen.component_id: frozen})[frozen.component_id]

    assert rejected.valid_for_training is False
    assert any(
        reason.startswith("frozen_adapter_hash_mismatch:")
        for reason in rejected.rejection_reasons
    )


def test_frozen_effective_contract_is_not_downgraded_by_source_yaml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = contract(maturity="adapter_implemented")
    frozen = contract(maturity="smoke_passed")
    source_dir = tmp_path / "source-components"
    snapshot_dir = tmp_path / "snapshot"
    source_dir.mkdir()
    snapshot_dir.mkdir()
    _write_contract(source_dir / "dummy.yaml", source)
    _write_contract(snapshot_dir / "component_contracts.yaml", frozen)
    artifact = frozen.maturity_artifacts[0]
    EffectiveComponentMaturityManifest(entries=[FrozenComponentMaturity(
        component_id=frozen.component_id,
        adapter_hash=adapter_source_hash(frozen),
        code_commit="snapshot-commit",
        ultralytics_version=installed_ultralytics_version(),
        protocol_hash="unavailable",
        effective_maturity="smoke_passed",
        runtime_execution_ready=True,
        artifacts=[FrozenMaturityArtifact(
            snapshot_artifact_name="component_maturity_dummy_smoke_passed_0",
            target_maturity=artifact.target_maturity,
            artifact_type=artifact.artifact_type,
            artifact_sha256=artifact.artifact_sha256,
            status=artifact.status,
            mock=artifact.mock,
        )],
    )]).to_yaml(snapshot_dir / "effective_component_maturity.yaml")
    monkeypatch.setattr(
        ResourcePaths,
        "COMPONENT_COMPATIBILITY",
        tmp_path / "missing-compatibility.yaml",
    )
    monkeypatch.setattr(ResourcePaths, "COMPONENTS_DIR", source_dir)
    live_registry = tmp_path / "live-maturity-registry.yaml"
    live_registry.write_text("overlays: [invalid\n", encoding="utf-8")
    context = SimpleNamespace(
        run_root=tmp_path / "runs",
        metadata={
            "research_snapshot_path": snapshot_dir.as_posix(),
            "research_snapshot_verified": True,
            "component_maturity_registry": live_registry.as_posix(),
        },
    )

    loaded = _load_execution_contracts(SimpleNamespace(context=context))

    assert len(loaded) == 1
    assert loaded[0].maturity == "smoke_passed"
    assert loaded[0].can_execute is True
    effective = context.metadata["effective_component_maturity"][source.component_id]
    assert effective["valid_for_training"] is True
    assert effective["evidence_source"] == "frozen_snapshot_artifact"


def _write_contract(path: Path, item) -> None:
    payload = item.model_dump(mode="json", exclude={"component_id"})
    path.write_text(
        yaml.safe_dump({"components": {item.component_id: payload}}, sort_keys=False),
        encoding="utf-8",
    )
