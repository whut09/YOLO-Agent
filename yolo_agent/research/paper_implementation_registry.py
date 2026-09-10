"""Build and persist the frozen paper-level implementation audit."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import yaml

from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.maturity import maturity_rank
from yolo_agent.research.executable_coverage_schemas import (
    ExecutablePaperCoverageBaseline,
    PaperExecutableCoverageEntry,
)
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)
from yolo_agent.research.paper_implementation_readiness import (
    PaperImplementationReadinessEvaluator,
)
from yolo_agent.research.paper_implementation_schemas import (
    PaperImplementationRegistry,
    PaperImplementationSpec,
)


class PaperImplementationRegistryError(ValueError):
    """Raised when a frozen paper implementation audit cannot be built."""


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_profiles(
    report: PaperMethodCoverageReport | None,
) -> tuple[dict[str, PaperMethodProfile], dict[str, PaperImplementationDecision]]:
    if report is None:
        return {}, {}
    profiles: dict[str, PaperMethodProfile] = {}
    for item in report.profiles:
        if item.paper_id in profiles:
            raise PaperImplementationRegistryError(
                f"duplicate method profile for frozen lookup: {item.paper_id}"
            )
        profiles[item.paper_id] = item
    decisions: dict[str, PaperImplementationDecision] = {}
    for item in report.decisions:
        if item.paper_id in decisions:
            raise PaperImplementationRegistryError(
                f"duplicate implementation decision for frozen lookup: {item.paper_id}"
            )
        decisions[item.paper_id] = item
    return profiles, decisions


def _index_inventory(path: Path | None) -> dict[str, PaperExecutionSpec]:
    if path is None or not path.is_file():
        return {}
    inventory = PaperExecutionInventory.from_yaml(path)
    return {item.paper_id: item for item in inventory.records}


def _index_coverage(path: Path | None) -> dict[str, PaperExecutableCoverageEntry]:
    if path is None or not path.is_file():
        return {}
    report = ExecutablePaperCoverageBaseline.from_yaml(path)
    return {item.paper_id: item for item in report.entries}


def _load_contract_map(
    path: Path | None,
    *,
    overlay_path: Path | None = None,
) -> dict[str, ComponentContract]:
    """Load production contracts and enrich metadata rows from local contracts.

    The production registry is the maturity source of truth.  Some older
    production snapshots contain a metadata-only row for a component even
    though its checked-in contract already declares an implementation identity.
    Reading that identity here lets the paper audit find the real adapter while
    keeping maturity artifacts and their non-mock evidence authoritative.
    """

    contracts: dict[str, ComponentContract] = {}
    if path is not None and path.exists():
        contracts.update(
            {item.component_id: item for item in load_contracts(path)}
        )

    if overlay_path is None or not overlay_path.exists():
        return contracts

    overlay_files = (
        [overlay_path]
        if overlay_path.is_file()
        else sorted(overlay_path.rglob("*.yaml"))
    )
    for contract_path in overlay_files:
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8-sig")) or {}
        entries = raw.get("components", raw) if isinstance(raw, dict) else {}
        if not isinstance(entries, dict) or any(
            not isinstance(values, dict) for values in entries.values()
        ):
            continue
        for overlay in load_contracts(contract_path):
            current = contracts.get(overlay.component_id)
            if current is None:
                contracts[overlay.component_id] = overlay
                continue
            if _has_implementation_identity(current):
                continue
            if not _has_implementation_identity(overlay):
                continue
            contracts[overlay.component_id] = _merge_contract_overlay(
                current,
                overlay,
            )
    return contracts


def _has_implementation_identity(contract: ComponentContract) -> bool:
    return bool(contract.implementation_path and contract.adapter_class)


def _merge_contract_overlay(
    current: ComponentContract,
    overlay: ComponentContract,
) -> ComponentContract:
    """Use a concrete local contract without discarding audited artifacts."""

    if current.component_id != overlay.component_id:
        raise ValueError("cannot merge contracts for different component IDs")

    current_rank = maturity_rank(current.maturity)
    overlay_rank = maturity_rank(overlay.maturity)
    artifacts = list(current.maturity_artifacts)
    for artifact in overlay.maturity_artifacts:
        if artifact not in artifacts:
            artifacts.append(artifact)

    update = overlay.model_dump(
        mode="python",
        exclude={"component_id", "maturity", "maturity_artifacts"},
    )
    update["maturity"] = (
        current.maturity if current_rank > overlay_rank else overlay.maturity
    )
    update["maturity_artifacts"] = artifacts
    for field in ("source_papers", "tests_required", "known_risks"):
        update[field] = sorted(
            set(getattr(current, field)) | set(getattr(overlay, field))
        )
    return current.model_copy(update=update)


class PaperImplementationRegistryBuilder:
    """Construct one independent implementation spec for each frozen paper."""

    def __init__(
        self,
        *,
        manifest_path: Path | str = "configs/research/paper_83_manifest.yaml",
        method_coverage_path: Path | str = (
            "research/production/paper_method_coverage.yaml"
        ),
        inventory_path: Path | str | None = (
            "runs/coverage-audit/paper_execution_inventory.yaml"
        ),
        coverage_path: Path | str | None = (
            "research/production/coverage_baseline.yaml"
        ),
        contracts_path: Path | str | None = (
            "research/production/component_contracts.yaml"
        ),
        contract_overlay_path: Path | str | None = "configs/components",
        tests_root: Path | str | None = "tests",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.method_coverage_path = Path(method_coverage_path)
        self.inventory_path = Path(inventory_path) if inventory_path is not None else None
        self.coverage_path = Path(coverage_path) if coverage_path is not None else None
        self.contracts_path = Path(contracts_path) if contracts_path is not None else None
        self.contract_overlay_path = (
            Path(contract_overlay_path)
            if contract_overlay_path is not None
            else None
        )
        self.tests_root = tests_root

    def build(self) -> PaperImplementationRegistry:
        """Build from the manifest; all mutable sources are enrichment only."""

        manifest = Paper83Manifest.from_yaml(self.manifest_path)
        paper_ids = [item.paper_id for item in manifest.papers]
        if len(paper_ids) != 83 or len(set(paper_ids)) != 83:
            raise PaperImplementationRegistryError(
                f"frozen paper manifest must contain 83 unique IDs, got {len(paper_ids)}"
            )

        method_report = self._load_method_report()
        profiles, decisions = _index_profiles(method_report)
        inventory = _index_inventory(self.inventory_path)
        coverage = _index_coverage(self.coverage_path)
        contracts = _load_contract_map(
            self.contracts_path,
            overlay_path=self.contract_overlay_path,
        )
        evaluator = PaperImplementationReadinessEvaluator(
            manifest_membership_hash=manifest.campaign.membership_hash,
            contracts=contracts,
            tests_root=self.tests_root,
        )

        records: list[PaperImplementationSpec] = []
        for paper in manifest.papers:
            record = evaluator.evaluate(
                paper=paper,
                profile=profiles.get(paper.paper_id),
                decision=decisions.get(paper.paper_id),
                inventory=inventory.get(paper.paper_id),
                coverage=coverage.get(paper.paper_id),
            )
            records.append(record)

        return PaperImplementationRegistry(
            manifest_path=str(self.manifest_path.resolve()),
            manifest_membership_hash=manifest.campaign.membership_hash,
            paper_count=len(records),
            source_inventory_hash=(
                inventory_hash(inventory, self.inventory_path)
                if inventory
                else _file_sha256(self.inventory_path)
                if self.inventory_path is not None
                else None
            ),
            source_method_coverage_hash=(
                _file_sha256(self.method_coverage_path)
                if self.method_coverage_path.is_file()
                else None
            ),
            records=sorted(records, key=lambda item: item.paper_id),
        ).with_hash()

    def _load_method_report(self) -> PaperMethodCoverageReport | None:
        if not self.method_coverage_path.is_file():
            return None
        return PaperMethodCoverageReport.from_yaml(self.method_coverage_path)


def inventory_hash(
    indexed_records: Mapping[str, PaperExecutionSpec],
    path: Path | None,
) -> str | None:
    """Return the inventory's semantic hash when available, else file hash."""

    if path is not None and path.is_file():
        inventory = PaperExecutionInventory.from_yaml(path)
        if inventory.inventory_hash:
            return inventory.inventory_hash
    return _file_sha256(path) if path is not None else None


def build_paper_implementation_registry(
    **kwargs: object,
) -> PaperImplementationRegistry:
    """Functional builder for the CLI and integrations."""

    return PaperImplementationRegistryBuilder(**kwargs).build()


def render_paper_implementation_readiness_markdown(
    registry: PaperImplementationRegistry,
) -> str:
    """Render a concise status report without implying paper reproduction."""

    counts = registry.readiness_counts
    lines = [
        "# Paper Implementation Readiness",
        "",
        "This is a paper-level implementation audit. It is not an exact paper "
        "reproduction, a pilot result, or a training result.",
        "",
        f"- Frozen papers: {registry.paper_count}",
        f"- Manifest membership hash: `{registry.manifest_membership_hash}`",
        f"- Implementation ready: {registry.implementation_ready_count}",
        f"- Blocked: {registry.blocked_count}",
        f"- Not audited: {registry.not_audited_count}",
        "",
        "## Readiness Counts",
        "",
        "| Readiness | Papers |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in counts.items())
    lines.extend(
        [
            "",
            "## Per-Paper Audit",
            "",
            "| Paper ID | Profile | Readiness | Evidence class | Components | "
            "Adapters | Blockers |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in registry.records:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md_cell(item.paper_id)}`",
                    f"`{_md_cell(item.method_profile_id)}`",
                    item.readiness,
                    item.implementation_evidence_class,
                    _md_items(item.component_ids),
                    _md_items(item.adapter_ids),
                    _md_items(item.blockers),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "A paper is `implementation_ready` only when its own paper-specific "
        "configuration and non-mock validation satisfy the strict boundary. "
        "`pilot_reproduced` remains a later state."
    )
    return "\n".join(lines)


def write_paper_implementation_readiness_artifacts(
    registry: PaperImplementationRegistry,
    *,
    yaml_path: Path | str,
    markdown_path: Path | str,
) -> tuple[Path, Path]:
    """Persist the machine report and human report."""

    yaml_output = registry.to_yaml(yaml_path, exclude_none=True, sort_keys=False)
    markdown_output = Path(markdown_path)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_paper_implementation_readiness_markdown(registry),
        encoding="utf-8",
    )
    return yaml_output, markdown_output


def _md_items(values: list[str]) -> str:
    return _md_cell("<br>".join(values) if values else "none")


def _md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


__all__ = [
    "PaperImplementationRegistryBuilder",
    "PaperImplementationRegistryError",
    "build_paper_implementation_registry",
    "inventory_hash",
    "render_paper_implementation_readiness_markdown",
    "write_paper_implementation_readiness_artifacts",
]
