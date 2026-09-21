"""Prompt-18H: source provenance collector tests.

Pins the deduplication (shared primitives -> one hash per implementation
identity), the reuse of :func:`adapter_source_hash`, the runtime
dependency closure (non-MRO helpers included), determinism, and the
fail-closed behavior for unresolvable components.  CPU-only, training-free.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from yolo_agent.research.release_source_provenance import (
    ReleaseSourceProvenanceError,
    collect_release_source_hashes,
    implementation_identity,
    runtime_dependency_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "runs/paper-readiness/paper_implementation_registry.yaml"


@pytest.fixture(scope="module")
def registry_payload() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8-sig")) or {}


def test_collects_deduplicated_adapter_hashes(registry_payload: dict) -> None:
    hashes, dependencies, unresolvable = collect_release_source_hashes(registry_payload)
    assert unresolvable == []
    assert hashes, "the 83-paper campaign must resolve real adapter identities"
    # The campaign binds 82 unique component ids onto 76 implementation
    # identities (shared adapters dedupe).
    assert len(hashes) == 76
    assert all(len(value) == 64 for value in hashes.values())


def test_shared_primitive_produces_one_identity(registry_payload: dict) -> None:
    """Multiple component ids on one implementation -> a single entry."""
    from yolo_agent.research.paper_runtime_preflight import _load_component_contracts

    contracts = _load_component_contracts()
    shared = [
        component_id
        for component_id, contract in contracts.items()
        if contract.implementation_path
        and contract.adapter_class
        and contract.implementation_path
        == "yolo_agent.components.adapters.losses.quality_alignment"
    ]
    assert len(shared) >= 2, "quality_alignment must be a shared primitive"
    hashes, _, _ = collect_release_source_hashes(registry_payload)
    identities = {implementation_identity(contracts[c]) for c in shared}
    assert len(identities) == 1
    assert next(iter(identities)) in hashes


def test_runtime_dependency_closure_covers_non_mro_helpers(registry_payload: dict) -> None:
    hashes, dependencies, _ = collect_release_source_hashes(registry_payload)
    # The non-MRO helpers the adapters import at runtime:
    for expected in (
        "yolo_agent.components.assignment",
        "yolo_agent.components.auxiliary_losses",
        "yolo_agent.components.distillation.losses",
        "yolo_agent.components.adapters.domain_adaptation.feature_alignment",
    ):
        assert expected in dependencies, f"{expected} missing from the closure"
    assert len(dependencies) >= 20
    assert all(len(value) == 64 for value in dependencies.values())


def test_dependency_closure_is_deterministic() -> None:
    first = runtime_dependency_files(["yolo_agent.components.assignment"])
    second = runtime_dependency_files(["yolo_agent.components.assignment"])
    assert first == second


def test_runtime_dependency_files_hashes_real_files() -> None:
    result = runtime_dependency_files(["yolo_agent.components.assignment"])
    file = Path(__import__("yolo_agent.components.assignment", fromlist=["x"]).__file__)
    expected = hashlib.sha256(file.read_bytes()).hexdigest()
    assert result["yolo_agent.components.assignment"] == expected


def test_type_checking_imports_are_excluded(tmp_path: Path) -> None:
    module = tmp_path / "typing_only_probe.py"
    module.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from yolo_agent.components.assignment import _ghost\n",
        encoding="utf-8",
    )
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        # Import the probe module (it must parse), then ask for its imports.
        from yolo_agent.research.release_source_provenance import _local_imports

        assert _local_imports(module) == set()
    finally:
        sys.path.pop(0)


def test_unresolvable_components_are_reported_not_pinned(registry_payload: dict) -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8-sig"))
    payload["records"].append({"paper_id": "arxiv:ghost", "component_ids": ["ghost.component"]})
    hashes, _, unresolvable = collect_release_source_hashes(payload)
    assert unresolvable == ["ghost.component"]
    # The honest records are still pinned:
    assert hashes


def test_broken_adapter_module_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered contract whose module cannot import fails closed."""
    from yolo_agent.components.contracts import ComponentContract
    from yolo_agent.research import release_source_provenance as module

    broken = ComponentContract(
        component_id="loss.probe",
        display_name="Probe",
        category="loss",
        implementation_path="yolo_agent.research.no_such_module_here",
        adapter_class="SomeAdapter",
    )
    monkeypatch.setattr(
        "yolo_agent.research.paper_runtime_preflight._load_component_contracts",
        lambda: {"loss.probe": broken},
    )
    with pytest.raises(ReleaseSourceProvenanceError, match="loss.probe"):
        collect_release_source_hashes(
            {"records": [{"paper_id": "arxiv:probe", "component_ids": ["loss.probe"]}]}
        )
    del module
