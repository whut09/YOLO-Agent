"""Generate deterministic paper catalog and local adapter maturity coverage."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.maturity import MaturityName
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
    installed_ultralytics_version,
)
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.component_coverage import ComponentCoverageAnalyzer


class PaperCatalogAudit(BaseModel, YAMLModelMixin):
    """Committed identity of the frozen catalog used for public counts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_catalog_audit.v1"
    audited_at: date
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_repository: str
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    paper_count: int = Field(ge=0)


class PaperAdapterCoverageReport(BaseModel, YAMLModelMixin):
    """Public, machine-readable separation of catalog and runtime maturity."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_adapter_coverage.v2"
    audited_at: date
    snapshot_hash: str
    source_repository: str
    source_commit: str
    source_catalog_hash: str
    paper_count: int = Field(ge=0)
    local_component_count: int = Field(ge=0)
    implemented_component_id_count: int = Field(ge=0)
    unique_adapter_class_count: int = Field(ge=0)
    runtime_integrated_count: int = Field(ge=0)
    pilot_reproduced_count: int = Field(ge=0)
    maturity_counts: dict[MaturityName, int]
    implemented_component_ids: list[str]
    adapter_implementation_ids: list[str]
    runtime_integrated_ids: list[str]
    pilot_reproduced_ids: list[str]


class LocalPaperAdapterCoverageReport(BaseModel, YAMLModelMixin):
    """Machine-local effective maturity after applying valid registry overlays."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_adapter_local_coverage.v1"
    audited_at: date
    registry_path: Path
    ultralytics_version: str
    protocol_hash: str | None = None
    source_component_count: int = Field(ge=0)
    overlay_applied_count: int = Field(ge=0)
    maturity_counts: dict[MaturityName, int]
    components_by_maturity: dict[MaturityName, list[str]]
    runtime_integrated_ids: list[str]
    smoke_passed_ids: list[str]
    gpu_certified_ids: list[str]
    pilot_reproduced_ids: list[str]
    overlay_statuses: dict[str, str]


def build_report(audit: PaperCatalogAudit) -> PaperAdapterCoverageReport:
    """Audit tracked contracts without consulting the live research registry."""
    resolver = ComponentAliasResolver.from_yaml()
    coverage = ComponentCoverageAnalyzer(resolver).analyze_resolutions(
        [],
        paper_count=audit.paper_count,
    )
    implemented = sorted(
        component_id
        for component_id in resolver.contracts
        if resolver.adapter_verified(component_id)
    )
    adapter_implementations = sorted(
        {
            f"{contract.implementation_path}:{contract.adapter_class}"
            for component_id in implemented
            for contract in [resolver.contracts[component_id]]
        }
    )
    runtime_ids = sorted(
        component_id
        for maturity, component_ids in coverage.components_by_maturity.items()
        if maturity in {
            "runtime_integrated",
            "unit_tested",
            "smoke_passed",
            "gpu_certified",
            "pilot_reproduced",
            "full_reproduced",
            "confirmed_multi_seed",
        }
        for component_id in component_ids
    )
    pilot_ids = sorted(
        component_id
        for maturity, component_ids in coverage.components_by_maturity.items()
        if maturity in {"pilot_reproduced", "full_reproduced", "confirmed_multi_seed"}
        for component_id in component_ids
    )
    maturity_counts = {
        name: int(getattr(coverage, name))
        for name in (
            "metadata_only",
            "recipe_idea_only",
            "adapter_implemented",
            "runtime_integrated",
            "unit_tested",
            "smoke_passed",
            "gpu_certified",
            "pilot_reproduced",
            "full_reproduced",
            "confirmed_multi_seed",
        )
    }
    return PaperAdapterCoverageReport(
        audited_at=audit.audited_at,
        snapshot_hash=audit.snapshot_hash,
        source_repository=audit.source_repository,
        source_commit=audit.source_commit,
        source_catalog_hash=audit.source_catalog_hash,
        paper_count=audit.paper_count,
        local_component_count=coverage.local_component_count,
        implemented_component_id_count=len(implemented),
        unique_adapter_class_count=len(adapter_implementations),
        runtime_integrated_count=len(runtime_ids),
        pilot_reproduced_count=len(pilot_ids),
        maturity_counts=maturity_counts,
        implemented_component_ids=implemented,
        adapter_implementation_ids=adapter_implementations,
        runtime_integrated_ids=runtime_ids,
        pilot_reproduced_ids=pilot_ids,
    )


def build_local_report(
    audit: PaperCatalogAudit,
    *,
    registry_path: Path | str,
    protocol_hash: str | None = None,
) -> LocalPaperAdapterCoverageReport:
    """Apply valid machine-local evidence without changing tracked contracts."""
    resolver = ComponentAliasResolver.from_yaml()
    registry = ComponentMaturityRegistry(registry_path)
    ultralytics_version = installed_ultralytics_version()
    effective = []
    statuses: dict[str, str] = {}
    for component_id, contract in sorted(resolver.contracts.items()):
        try:
            updated, resolution = registry.apply(
                contract,
                adapter_hash=adapter_source_hash(contract),
                ultralytics_version=ultralytics_version,
                protocol_hash=protocol_hash,
            )
        except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
            updated = contract
            statuses[component_id] = f"invalid:{exc}"
        else:
            statuses[component_id] = resolution.status
        effective.append(updated)
    effective_resolver = ComponentAliasResolver(
        resolver.config,
        contracts=effective,
    )
    coverage = ComponentCoverageAnalyzer(effective_resolver).analyze_resolutions(
        [],
        paper_count=audit.paper_count,
    )
    maturity_counts = {
        name: int(getattr(coverage, name))
        for name in (
            "metadata_only",
            "recipe_idea_only",
            "adapter_implemented",
            "runtime_integrated",
            "unit_tested",
            "smoke_passed",
            "gpu_certified",
            "pilot_reproduced",
            "full_reproduced",
            "confirmed_multi_seed",
        )
    }
    by_maturity = coverage.components_by_maturity
    return LocalPaperAdapterCoverageReport(
        audited_at=audit.audited_at,
        registry_path=Path(registry_path).resolve(),
        ultralytics_version=ultralytics_version,
        protocol_hash=protocol_hash,
        source_component_count=coverage.local_component_count,
        overlay_applied_count=sum(status == "applied" for status in statuses.values()),
        maturity_counts=maturity_counts,
        components_by_maturity=by_maturity,
        runtime_integrated_ids=_at_or_above(by_maturity, "runtime_integrated"),
        smoke_passed_ids=_at_or_above(by_maturity, "smoke_passed"),
        gpu_certified_ids=_at_or_above(by_maturity, "gpu_certified"),
        pilot_reproduced_ids=_at_or_above(by_maturity, "pilot_reproduced"),
        overlay_statuses=statuses,
    )


def generate(
    *,
    audit_path: Path,
    report_path: Path,
    check: bool = False,
) -> bool:
    """Write the report, or return whether the committed report is current."""
    report = build_report(PaperCatalogAudit.from_yaml(audit_path))
    if check:
        return report_path.is_file() and PaperAdapterCoverageReport.from_yaml(
            report_path
        ) == report
    report.to_yaml(report_path, exclude_none=True, sort_keys=False)
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate paper adapter coverage.")
    parser.add_argument("--audit", type=Path, default=Path("configs/paper_catalog_audit.yaml"))
    parser.add_argument("--report", type=Path, default=Path("docs/paper-adapter-coverage.yaml"))
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--local-report", type=Path)
    parser.add_argument("--protocol-hash")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.registry is not None:
        local_path = args.local_report or Path(
            "runs/paper-adapter-coverage.local.yaml"
        )
        local = build_local_report(
            PaperCatalogAudit.from_yaml(args.audit),
            registry_path=args.registry,
            protocol_hash=args.protocol_hash,
        )
        local.to_yaml(local_path, exclude_none=True, sort_keys=False)
        print(f"Generated local paper adapter coverage report: {local_path}")
        return 0
    current = generate(audit_path=args.audit, report_path=args.report, check=args.check)
    if args.check and not current:
        print("Paper adapter coverage report is stale.")
        return 1
    print("Paper adapter coverage report is current." if current else "Generated paper adapter coverage report.")
    return 0


def _at_or_above(
    components_by_maturity: dict[MaturityName, list[str]],
    minimum: MaturityName,
) -> list[str]:
    order = (
        "metadata_only",
        "recipe_idea_only",
        "adapter_implemented",
        "runtime_integrated",
        "unit_tested",
        "smoke_passed",
        "gpu_certified",
        "pilot_reproduced",
        "full_reproduced",
        "confirmed_multi_seed",
    )
    threshold = order.index(minimum)
    return sorted(
        component_id
        for index, maturity in enumerate(order)
        if index >= threshold
        for component_id in components_by_maturity.get(maturity, [])
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LocalPaperAdapterCoverageReport",
    "PaperAdapterCoverageReport",
    "PaperCatalogAudit",
    "build_local_report",
    "build_report",
    "generate",
]
