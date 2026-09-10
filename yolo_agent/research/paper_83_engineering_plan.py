"""Build a paper-shaped implementation plan for the frozen paper-83 campaign.

The plan is deliberately weaker than implementation readiness.  It records the
work required for every frozen paper and keeps shared primitives as
dependencies, but it never promotes a paper because a generic adapter exists.
This module is offline-only and does not create a trainer or evidence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import hashlib
import math
from pathlib import Path
import re

from yolo_agent.research.paper_83_campaign_schemas import (
    Paper83Manifest,
    Paper83Paper,
)
from yolo_agent.research.paper_83_implementation_schemas import (
    Paper83EngineeringBatch,
    Paper83EngineeringPlan,
    Paper83EngineeringPlanEntry,
)
from yolo_agent.research.paper_implementation_registry import (
    PaperImplementationRegistryBuilder,
)
from yolo_agent.research.paper_implementation_schemas import (
    PaperImplementationRegistry,
    PaperImplementationSpec,
)


class Paper83EngineeringPlanError(ValueError):
    """Raised when a complete plan cannot be derived from the frozen campaign."""


_COMPONENT_DOMAIN_PREFIXES: tuple[tuple[str, str], ...] = (
    ("domain_adaptation.", "domain_adaptation"),
    ("distillation.", "distillation"),
    ("inference.", "inference"),
    ("assigner.", "assignment"),
    ("detection_head.", "head"),
    ("head.", "head"),
    ("neck.", "neck"),
    ("feature_pyramid.", "feature_fusion"),
    ("attention.", "feature_fusion"),
    ("sampling.", "sampling"),
    ("loss.", "bbox_loss"),
)

_BLOCKER_PARTS: tuple[tuple[str, str], ...] = (
    ("paper_specific_mechanism_missing", "paper_specific_mechanism_interpretation"),
    ("paper_implementation_not_specific", "paper_specific_mechanism_interpretation"),
    ("missing_paper_config", "paper_specific_config"),
    ("missing_component_binding", "component_binding"),
    ("missing_adapter_binding", "adapter_binding"),
    ("missing_adapter_contract", "adapter_contract"),
    ("adapter_import_failed", "adapter_import"),
    ("missing_runtime", "runtime_hook_or_payload"),
    ("runtime_payload", "runtime_payload"),
    ("component_unit_evidence_missing", "unit_validation"),
    ("missing_test:unit", "unit_validation"),
    ("component_smoke_evidence_missing", "non_mock_smoke_validation"),
    ("missing_test:smoke", "non_mock_smoke_validation"),
    ("missing_test:compatibility", "yolo26_compatibility_validation"),
    ("compatibility_", "yolo26_compatibility_validation"),
    ("paper_specific_evidence_unbound", "paper_specific_evidence_provenance"),
)


def _unique(values: Iterable[object]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return result or "mixed"


def _component_domain(component_id: str) -> str | None:
    normalized = component_id.strip().casefold()
    for prefix, domain in _COMPONENT_DOMAIN_PREFIXES:
        if normalized.startswith(prefix):
            return domain
    return None


def classify_implementation_domain(spec: PaperImplementationSpec) -> str:
    """Classify a plan by explicit component identity, never title keywords."""

    domains = {
        domain
        for component_id in spec.component_ids
        if (domain := _component_domain(component_id)) is not None
    }
    if "domain_adaptation" in domains:
        return "domain_adaptation"
    if "distillation" in domains:
        return "distillation"
    if "inference" in domains:
        return "inference"
    return sorted(domains)[0] if domains else "other"


def _secondary_domains(spec: PaperImplementationSpec, primary: str) -> list[str]:
    domains = {
        domain
        for component_id in spec.component_ids
        if (domain := _component_domain(component_id)) is not None
    }
    return sorted(domains - {primary})


def _missing_parts(spec: PaperImplementationSpec) -> list[str]:
    parts: set[str] = set()
    for blocker in spec.blockers:
        for marker, part in _BLOCKER_PARTS:
            if marker in blocker:
                parts.add(part)
                break
    if not spec.paper_specific_mechanisms:
        parts.add("paper_specific_mechanism_interpretation")
    if spec.readiness != "implementation_ready" and not parts:
        parts.add("paper_specific_readiness_review")
    return sorted(parts)


def _required_evidence(spec: PaperImplementationSpec, missing: list[str]) -> list[str]:
    required = list(spec.paper_evidence_refs)
    if "paper_specific_mechanism_interpretation" in missing:
        required.append("paper_specific_mechanism_interpretation")
    if "paper_specific_config" in missing:
        required.append("paper_specific_config_review")
    if "unit_validation" in missing:
        required.append("non_mock_unit_validation")
    if "non_mock_smoke_validation" in missing:
        required.append("non_mock_forward_backward_smoke")
    if "yolo26_compatibility_validation" in missing:
        required.append("yolo26_one_to_one_dfl_free_imgsz_640_validation")
    return _unique(required)


def _strategy(spec: PaperImplementationSpec, missing: list[str]) -> str:
    if not spec.paper_specific_mechanisms:
        return (
            "Resolve the source paper mechanism from auditable evidence before "
            "binding any shared primitive."
        )
    if not missing:
        return (
            "Preserve this paper-specific configuration and provenance while "
            "running the fixed-protocol training validation."
        )
    return (
        "Implement and validate the listed paper-specific gaps, then rerun the "
        "non-mock readiness checks before promotion."
    )


def _test_plan(spec: PaperImplementationSpec) -> tuple[list[str], list[str], list[str]]:
    unit = list(spec.unit_test_refs) or [
        "add paper-specific unit coverage for "
        + (", ".join(spec.paper_specific_mechanisms) or "the unresolved mechanism")
    ]
    smoke = list(spec.smoke_test_refs) or [
        "add non-mock synthetic forward/backward smoke for the paper route"
    ]
    compatibility = list(spec.compatibility_test_refs) or [
        "validate YOLO26 one-to-one head, DFL-free regression, and imgsz=640"
    ]
    return _unique(unit), _unique(smoke), _unique(compatibility)


def _plan_entry(paper: Paper83Paper, spec: PaperImplementationSpec) -> Paper83EngineeringPlanEntry:
    primary = classify_implementation_domain(spec)
    missing = _missing_parts(spec)
    unit, smoke, compatibility = _test_plan(spec)
    blockers = list(spec.blockers)
    if spec.readiness != "implementation_ready":
        blockers.append(f"paper_readiness_not_confirmed:{spec.readiness}")
    blockers = _unique(blockers)
    evidence_status = (
        "sufficient_for_planning"
        if spec.paper_specific_mechanisms and spec.paper_evidence_refs
        else "blocked_missing_evidence"
    )
    status = "incompatible" if paper.current_disposition == "incompatible" else "planned"
    known_facts = [
        f"current_readiness:{spec.readiness}",
        f"implementation_evidence_class:{spec.implementation_evidence_class}",
    ]
    if spec.paper_specific_mechanisms:
        known_facts.append(
            "paper_specific_mechanisms:" + ",".join(spec.paper_specific_mechanisms)
        )
    else:
        known_facts.append("shared adapter mapping is not paper implementation")
    unknown_facts = [
        part for part in missing if part in {
            "paper_specific_mechanism_interpretation",
            "paper_specific_config",
            "paper_specific_evidence_provenance",
        }
    ]
    if not spec.runtime_implementation_verified:
        unknown_facts.append("runtime_implementation_verification")
    if not spec.non_mock_smoke_passed:
        unknown_facts.append("non_mock_smoke_verification")
    unknown_facts = _unique(unknown_facts)
    return Paper83EngineeringPlanEntry(
        paper_id=paper.paper_id,
        title=paper.title,
        method_profile_id=paper.method_profile_id,
        execution_fingerprint=spec.implementation_fingerprint,
        current_disposition=paper.current_disposition,
        implementation_domain=primary,
        secondary_domains=_secondary_domains(spec, primary),
        paper_mechanism_summary=spec.mechanism_summary,
        paper_specific_mechanism_ids=list(spec.paper_specific_mechanisms),
        shared_primitives_available=list(spec.shared_primitives),
        existing_component_ids=list(spec.component_ids),
        existing_adapter_ids=list(spec.adapter_ids),
        paper_specific_missing_parts=missing,
        runtime_insertion_points=list(spec.runtime_insertion_points),
        implementation_strategy=_strategy(spec, missing),
        unit_test_plan=unit,
        smoke_test_plan=smoke,
        compatibility_test_plan=compatibility,
        required_assets=list(spec.required_assets),
        evidence_status=evidence_status,
        status=status,
        known_facts=_unique(known_facts),
        unknown_facts=unknown_facts,
        required_evidence=_required_evidence(spec, missing),
        difficulty=(
            "high"
            if not spec.paper_specific_mechanisms or len(missing) >= 3
            else "medium"
            if missing
            else "low"
        ),
        estimated_files=max(1, len(missing) + len(spec.component_ids)),
        blockers=blockers,
        dependency_papers_or_primitives=list(spec.shared_primitives),
        source_locations=list(spec.paper_evidence_refs),
    )


def _balanced_chunks(items: list[str], max_size: int = 6) -> list[list[str]]:
    if not items:
        return []
    chunk_count = max(1, math.ceil(len(items) / max_size))
    base, remainder = divmod(len(items), chunk_count)
    sizes = [base + (1 if index < remainder else 0) for index in range(chunk_count)]
    chunks: list[list[str]] = []
    offset = 0
    for size in sizes:
        chunks.append(items[offset : offset + size])
        offset += size
    return chunks


def _build_batches(entries: list[Paper83EngineeringPlanEntry]) -> list[Paper83EngineeringBatch]:
    groups: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        focus = entry.implementation_domain
        if focus in {"inference", "evaluation"}:
            focus = "independent_components"
        groups[focus].append(entry.paper_id)

    small_groups = [name for name, ids in groups.items() if len(ids) < 4]
    if small_groups:
        mixed = groups.setdefault("independent_components", [])
        for name in small_groups:
            if name == "independent_components":
                continue
            mixed.extend(groups.pop(name))

    batches: list[Paper83EngineeringBatch] = []
    entries_by_id = {entry.paper_id: entry for entry in entries}
    for focus in sorted(groups):
        paper_ids = sorted(groups[focus])
        for index, chunk in enumerate(_balanced_chunks(paper_ids), start=1):
            primitive_ids = _unique(
                primitive
                for paper_id in chunk
                for primitive in entries_by_id[paper_id].shared_primitives_available
            )
            batches.append(
                Paper83EngineeringBatch(
                    batch_id=f"paper83-{_slug(focus)}-{index:02d}",
                    focus_domain=focus,
                    shared_primitives=primitive_ids,
                    paper_ids=chunk,
                    paper_count=len(chunk),
                    rationale=(
                        f"Process {len(chunk)} frozen papers through the {focus} "
                        "implementation boundary while preserving per-paper specs."
                    ),
                )
            )
    return batches


class Paper83EngineeringPlanBuilder:
    """Build the complete paper-83 plan from the immutable campaign manifest."""

    def __init__(
        self,
        *,
        manifest_path: Path | str = "configs/research/paper_83_manifest.yaml",
        implementation_registry_path: Path | str | None = None,
        method_coverage_path: Path | str = "research/production/paper_method_coverage.yaml",
        inventory_path: Path | str | None = "runs/coverage-audit/paper_execution_inventory.yaml",
        coverage_path: Path | str | None = "research/production/coverage_baseline.yaml",
        contracts_path: Path | str | None = "research/production/component_contracts.yaml",
        tests_root: Path | str | None = "tests",
        implementation_registry: PaperImplementationRegistry | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.implementation_registry_path = (
            Path(implementation_registry_path)
            if implementation_registry_path is not None
            else None
        )
        self.method_coverage_path = Path(method_coverage_path)
        self.inventory_path = Path(inventory_path) if inventory_path is not None else None
        self.coverage_path = Path(coverage_path) if coverage_path is not None else None
        self.contracts_path = Path(contracts_path) if contracts_path is not None else None
        self.tests_root = tests_root
        self._implementation_registry = implementation_registry

    def build(self) -> Paper83EngineeringPlan:
        manifest = Paper83Manifest.from_yaml(self.manifest_path)
        paper_ids = [paper.paper_id for paper in manifest.papers]
        if len(paper_ids) != 83 or len(set(paper_ids)) != 83:
            raise Paper83EngineeringPlanError(
                f"frozen paper manifest must contain 83 unique IDs, got {len(paper_ids)}"
            )
        registry = self._load_registry()
        by_id = {record.paper_id: record for record in registry.records}
        if sorted(by_id) != sorted(paper_ids):
            missing = sorted(set(paper_ids) - set(by_id))
            extra = sorted(set(by_id) - set(paper_ids))
            raise Paper83EngineeringPlanError(
                f"implementation registry membership mismatch: missing={missing} extra={extra}"
            )
        if registry.manifest_membership_hash != manifest.campaign.membership_hash:
            raise Paper83EngineeringPlanError(
                "implementation registry membership hash does not match frozen manifest"
            )

        entries = [
            _plan_entry(paper, by_id[paper.paper_id])
            for paper in manifest.papers
        ]
        batches = _build_batches(entries)
        batch_ids = [paper_id for batch in batches for paper_id in batch.paper_ids]
        if sorted(batch_ids) != sorted(paper_ids):
            raise Paper83EngineeringPlanError(
                "generated engineering batches do not cover the frozen campaign"
            )
        summary = {
            "paper_count": len(entries),
            "implementation_ready_count": registry.implementation_ready_count,
            "planned_count": sum(item.status == "planned" for item in entries),
            "blocked_count": sum(bool(item.blockers) for item in entries),
            "incompatible_count": sum(item.status == "incompatible" for item in entries),
            "distillation_count": sum(
                item.implementation_domain == "distillation" for item in entries
            ),
            "domain_adaptation_count": sum(
                item.implementation_domain == "domain_adaptation" for item in entries
            ),
            "independent_component_count": sum(
                item.implementation_domain not in {"distillation", "domain_adaptation"}
                for item in entries
            ),
        }
        return Paper83EngineeringPlan(
            manifest_path=str(self.manifest_path.resolve()),
            manifest_membership_hash=manifest.campaign.membership_hash,
            manifest_paper_ids=paper_ids,
            paper_count=len(entries),
            source_hashes=self._source_hashes(registry),
            papers=entries,
            batches=batches,
            batch_directory="embedded_in_plan",
            summary=summary,
        ).with_hash()

    def _load_registry(self) -> PaperImplementationRegistry:
        if self._implementation_registry is not None:
            return self._implementation_registry
        if self.implementation_registry_path is not None and self.implementation_registry_path.is_file():
            return PaperImplementationRegistry.from_yaml(self.implementation_registry_path)
        return PaperImplementationRegistryBuilder(
            manifest_path=self.manifest_path,
            method_coverage_path=self.method_coverage_path,
            inventory_path=self.inventory_path,
            coverage_path=self.coverage_path,
            contracts_path=self.contracts_path,
            tests_root=self.tests_root,
        ).build()

    def _source_hashes(self, registry: PaperImplementationRegistry) -> dict[str, str]:
        paths: Mapping[str, Path | None] = {
            "manifest": self.manifest_path,
            "method_coverage": self.method_coverage_path,
            "inventory": self.inventory_path,
            "coverage": self.coverage_path,
            "contracts": self.contracts_path,
        }
        values = {
            name: digest
            for name, path in paths.items()
            if path is not None
            and (digest := _sha256_file(path)) is not None
        }
        if registry.registry_hash:
            values["implementation_registry"] = registry.registry_hash
        return values


def build_paper_83_engineering_plan(**kwargs: object) -> Paper83EngineeringPlan:
    """Functional wrapper for callers and the CLI."""

    return Paper83EngineeringPlanBuilder(**kwargs).build()


def render_paper_83_engineering_plan_markdown(
    plan: Paper83EngineeringPlan,
) -> str:
    """Render a concise plan without claiming that planned papers are ready."""

    lines = [
        "# Paper-83 Engineering Plan",
        "",
        "This document is an implementation work plan, not a reproduction or "
        "training result.",
        "",
        f"- Frozen papers: {plan.paper_count}",
        f"- Membership hash: `{plan.manifest_membership_hash}`",
        f"- Implementation-ready at source audit: {plan.summary.get('implementation_ready_count', 0)}",
        f"- Batches: {len(plan.batches)}",
        "",
        "Shared primitives are dependencies only; each row remains a separate "
        "paper implementation contract.",
        "",
        "## Batches",
        "",
        "| Batch | Focus | Papers | Shared primitives |",
        "|---|---|---:|---|",
    ]
    for batch in plan.batches:
        primitives = "<br>".join(batch.shared_primitives) or "none"
        lines.append(
            f"| `{batch.batch_id}` | {batch.focus_domain} | "
            f"{batch.paper_count} | {primitives} |"
        )
    lines.extend(
        [
            "",
            "## Per-Paper Plan",
            "",
            "| Paper | Domain | Readiness at audit | Mechanisms | Missing parts | Disposition |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in plan.papers:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md_cell(item.paper_id)}`",
                    item.implementation_domain,
                    _readiness_from_facts(item.known_facts),
                    _md_items(item.paper_specific_mechanism_ids),
                    _md_items(item.paper_specific_missing_parts),
                    item.current_disposition or "unknown",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "`implementation_ready` is intentionally not inferred from a shared "
            "adapter, a recipe, a mock smoke, or a paper claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _readiness_from_facts(facts: list[str]) -> str:
    prefix = "current_readiness:"
    return next((item.removeprefix(prefix) for item in facts if item.startswith(prefix)), "unknown")


def _md_items(values: Iterable[str]) -> str:
    return _md_cell("<br>".join(values) if values else "none")


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def write_paper_83_engineering_plan(
    plan: Paper83EngineeringPlan,
    *,
    yaml_path: Path | str,
    markdown_path: Path | str,
) -> tuple[Path, Path]:
    """Write the machine plan and its human-readable companion."""

    yaml_output = plan.to_yaml(yaml_path, exclude_none=True, sort_keys=False)
    markdown_output = Path(markdown_path)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_paper_83_engineering_plan_markdown(plan),
        encoding="utf-8",
    )
    return yaml_output, markdown_output


__all__ = [
    "Paper83EngineeringPlanBuilder",
    "Paper83EngineeringPlanError",
    "build_paper_83_engineering_plan",
    "classify_implementation_domain",
    "render_paper_83_engineering_plan_markdown",
    "write_paper_83_engineering_plan",
]
