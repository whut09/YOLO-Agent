"""Audit the 13 paper component adapters without inflating local maturity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.adapters.base import ComponentAdapter
from yolo_agent.components.adapters.audit_contract import (
    EXPECTED_RUNTIME_ADAPTERS,
    PluginKind,
)
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.maturity import maturity_rank
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
    installed_ultralytics_version,
)
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.component_aliases import ComponentAliasResolver


class RuntimeAdapterAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: str
    adapter_class: str | None = None
    adapter_hash: str | None = None
    plugin_kind: PluginKind
    required_hook: str
    changed_variable: str
    source_maturity: str
    effective_maturity: str
    payload_implemented: bool
    runtime_observed: bool
    overlay_status: str
    blocked_by: list[str] = Field(default_factory=list)


class RuntimeAdapterAuditReport(BaseModel, YAMLModelMixin):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "runtime_adapter_audit.v1"
    expected_count: int = 13
    audited_count: int
    payload_implemented_count: int
    runtime_observed_count: int
    records: list[RuntimeAdapterAuditRecord]


def build_runtime_adapter_audit(
    *,
    registry_path: Path | str = Path("runs/component_maturity_registry.yaml"),
    protocol_hash: str | None = None,
) -> RuntimeAdapterAuditReport:
    """Inspect source implementations and valid machine-local maturity overlays."""
    resolver = ComponentAliasResolver.from_yaml()
    maturity_registry = ComponentMaturityRegistry(registry_path)
    ultralytics_version = installed_ultralytics_version()
    adapter_registry = ComponentAdapterRegistry()
    records: list[RuntimeAdapterAuditRecord] = []
    for component_id, expectation in EXPECTED_RUNTIME_ADAPTERS.items():
        contract = resolver.contracts.get(component_id)
        if contract is None:
            records.append(
                RuntimeAdapterAuditRecord(
                    component_id=component_id,
                    plugin_kind=expectation.plugin_kind,
                    required_hook=expectation.required_hook,
                    changed_variable=expectation.changed_variable,
                    source_maturity="missing",
                    effective_maturity="missing",
                    payload_implemented=False,
                    runtime_observed=False,
                    overlay_status="missing_contract",
                    blocked_by=["component_contract_missing"],
                )
            )
            continue
        blocked: list[str] = []
        try:
            adapter = adapter_registry.create_for_contract(contract)
            adapter_hash = adapter_source_hash(contract, adapter=adapter)
            payload_implemented = (
                type(adapter).build_runtime_payload
                is not ComponentAdapter.build_runtime_payload
            )
            effective, resolution = maturity_registry.apply(
                contract,
                adapter_hash=adapter_hash,
                ultralytics_version=ultralytics_version,
                protocol_hash=protocol_hash,
            )
        except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
            adapter = None
            adapter_hash = None
            payload_implemented = False
            effective = contract
            overlay_status = f"invalid:{exc}"
            blocked.append("adapter_or_overlay_invalid")
        else:
            overlay_status = resolution.status
        if not payload_implemented:
            blocked.append("typed_runtime_payload_not_implemented")
        runtime_observed = maturity_rank(effective.maturity) >= maturity_rank(
            "smoke_passed"
        )
        if not runtime_observed:
            blocked.append("artifact_backed_runtime_hook_not_observed")
        records.append(
            RuntimeAdapterAuditRecord(
                component_id=component_id,
                adapter_class=type(adapter).__name__ if adapter is not None else None,
                adapter_hash=adapter_hash,
                plugin_kind=expectation.plugin_kind,
                required_hook=expectation.required_hook,
                changed_variable=expectation.changed_variable,
                source_maturity=contract.maturity,
                effective_maturity=effective.maturity,
                payload_implemented=payload_implemented,
                runtime_observed=runtime_observed,
                overlay_status=overlay_status,
                blocked_by=blocked,
            )
        )
    return RuntimeAdapterAuditReport(
        audited_count=len(records),
        payload_implemented_count=sum(item.payload_implemented for item in records),
        runtime_observed_count=sum(item.runtime_observed for item in records),
        records=records,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit paper runtime adapters.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("runs/component_maturity_registry.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/runtime_adapter_audit.yaml"),
    )
    parser.add_argument("--protocol-hash")
    args = parser.parse_args(argv)
    report = build_runtime_adapter_audit(
        registry_path=args.registry,
        protocol_hash=args.protocol_hash,
    )
    report.to_yaml(args.output, exclude_none=True, sort_keys=False)
    print(
        f"Audited {report.audited_count} adapters: "
        f"payloads={report.payload_implemented_count} "
        f"runtime_observed={report.runtime_observed_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_RUNTIME_ADAPTERS",
    "RuntimeAdapterAuditRecord",
    "RuntimeAdapterAuditReport",
    "build_runtime_adapter_audit",
]
