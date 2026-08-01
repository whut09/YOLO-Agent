"""Verified inputs for executable paper coverage audits."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
    installed_ultralytics_version,
)
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.maturity_snapshot import (
    EffectiveComponentMaturityManifest,
    FrozenComponentMaturity,
)
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.snapshot import ResearchSnapshot


class ExecutableCoverageInputs(BaseModel):
    """One verified method, contract, and maturity input set."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    method_coverage_path: Path
    contracts: dict[str, ComponentContract]
    maturity: EffectiveComponentMaturityManifest
    snapshot_hash: str | None = None


def load_snapshot_coverage_inputs(
    snapshot_dir: Path | str,
) -> ExecutableCoverageInputs:
    """Load only artifacts bound by one verified ResearchSnapshot."""
    root = Path(snapshot_dir)
    snapshot = ResearchSnapshot.from_snapshot_dir(root)
    failures = snapshot.verify(root)
    if failures:
        raise ValueError("invalid research snapshot: " + "; ".join(failures))
    method_path = _snapshot_artifact(snapshot, root, "paper_method_coverage")
    contracts_path = _snapshot_artifact(snapshot, root, "component_contracts")
    maturity_path = _snapshot_artifact(
        snapshot, root, "effective_component_maturity"
    )
    return ExecutableCoverageInputs(
        method_coverage_path=method_path,
        contracts={
            item.component_id: item for item in _load_contracts(contracts_path)
        },
        maturity=EffectiveComponentMaturityManifest.from_yaml(maturity_path),
        snapshot_hash=snapshot.snapshot_hash,
    )


def load_live_coverage_inputs(
    *,
    method_coverage_path: Path | str,
    maturity_registry_path: Path | str,
    resolver: ComponentAliasResolver | None = None,
    ultralytics_version: str | None = None,
) -> ExecutableCoverageInputs:
    """Resolve valid local overlays without mutating source component YAML."""
    effective_resolver = resolver or ComponentAliasResolver.from_yaml()
    contracts = dict(effective_resolver.contracts)
    registry = ComponentMaturityRegistry(maturity_registry_path)
    runtime_version = ultralytics_version or installed_ultralytics_version()
    entries: list[FrozenComponentMaturity] = []
    for component_id, contract in sorted(contracts.items()):
        try:
            adapter_hash = adapter_source_hash(contract)
        except (AttributeError, ImportError, TypeError, ValueError):
            continue
        effective, resolution, overlay = registry.resolve(
            contract,
            adapter_hash=adapter_hash,
            ultralytics_version=runtime_version,
        )
        if overlay is None or resolution.status != "applied":
            continue
        entries.append(
            FrozenComponentMaturity(
                component_id=component_id,
                adapter_hash=adapter_hash,
                code_commit=overlay.code_commit,
                ultralytics_version=overlay.ultralytics_version,
                protocol_hash=overlay.protocol_hash,
                overlay_identity_key=overlay.identity_key,
                overlay_evidence_hash=overlay.evidence_hash,
                effective_maturity=effective.maturity,
                runtime_execution_ready=effective.can_execute,
            )
        )
    return ExecutableCoverageInputs(
        method_coverage_path=Path(method_coverage_path),
        contracts=contracts,
        maturity=EffectiveComponentMaturityManifest(entries=entries),
    )


def load_method_coverage(
    inputs: ExecutableCoverageInputs,
) -> PaperMethodCoverageReport:
    return PaperMethodCoverageReport.from_yaml(inputs.method_coverage_path)


def _snapshot_artifact(
    snapshot: ResearchSnapshot,
    root: Path,
    name: str,
) -> Path:
    artifact = snapshot.artifacts.get(name)
    if artifact is None:
        raise ValueError(f"verified snapshot is missing required artifact: {name}")
    return root / artifact.path


def _load_contracts(path: Path) -> list[ComponentContract]:
    from yolo_agent.components.contracts import load_contracts

    return load_contracts(path)


__all__ = [
    "ExecutableCoverageInputs",
    "load_live_coverage_inputs",
    "load_method_coverage",
    "load_snapshot_coverage_inputs",
]
