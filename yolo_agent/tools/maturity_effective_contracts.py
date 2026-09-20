"""Effective contract loading shared by inventory and coverage tooling.

Component certification persists machine-local maturity overlays in
``runs/component_maturity_registry.yaml``.  Any pipeline that classifies paper
executability from contract maturity must consume the merged
source+overlay contracts, not the bare source contracts, or certified
non-mock smoke evidence is silently dropped from static dispositions.
"""

from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.component_runner import installed_ultralytics_version
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
)
from yolo_agent.research.component_aliases import ComponentAliasResolver


def load_effective_contracts(
    maturity_registry: Path | str = Path("runs/component_maturity_registry.yaml"),
    *,
    resolver: ComponentAliasResolver | None = None,
) -> dict[str, ComponentContract]:
    """Return source contracts merged with their matching local overlays.

    Contracts without an importable adapter, without a resolving overlay, or
    whose overlay resolution fails fall back to the source contract so the
    denominator never changes.
    """

    effective_resolver = resolver or ComponentAliasResolver.from_yaml()
    registry_path = Path(maturity_registry)
    if not registry_path.is_file():
        return dict(effective_resolver.contracts)
    registry = ComponentMaturityRegistry(registry_path)
    version = installed_ultralytics_version()
    effective: dict[str, ComponentContract] = {}
    for component_id, contract in effective_resolver.contracts.items():
        try:
            adapter_hash = adapter_source_hash(contract)
            merged, _, _ = registry.resolve(
                contract,
                adapter_hash=adapter_hash,
                ultralytics_version=version,
            )
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            effective[component_id] = contract
            continue
        effective[component_id] = merged
    return effective


__all__ = ["load_effective_contracts"]
