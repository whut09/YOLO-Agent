"""Build a paper-level execution inventory from frozen method coverage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from yolo_agent.recipes.schemas import RecipeSpec
from yolo_agent.research.executable_coverage_schemas import (
    ExecutablePaperCoverageBaseline,
    PaperExecutableCoverageEntry,
)
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)
from yolo_agent.research.schemas import PaperRecord


GENERIC_COMPONENT_IDS = frozenset(
    {
        "distillation.yolo26_teacher_student",
        "domain_adaptation.general",
        "quality_alignment.general",
    }
)


class PaperExecutionInventoryBuilder:
    """Create one execution record for every compatible paper ID."""

    def __init__(
        self,
        *,
        generic_component_ids: Iterable[str] = GENERIC_COMPONENT_IDS,
    ) -> None:
        self.generic_component_ids = frozenset(generic_component_ids)

    @staticmethod
    def compatible_method_pairs(
        method_coverage: PaperMethodCoverageReport,
        compatible_paper_ids: Iterable[str],
    ) -> list[tuple[PaperMethodProfile, PaperImplementationDecision]]:
        """Return sorted profile/decision pairs without dropping a paper ID."""
        profiles = {item.paper_id: item for item in method_coverage.profiles}
        decisions = {item.paper_id: item for item in method_coverage.decisions}
        requested = sorted(set(compatible_paper_ids))
        missing_profiles = sorted(set(requested) - set(profiles))
        missing_decisions = sorted(set(requested) - set(decisions))
        if missing_profiles or missing_decisions:
            failures = []
            if missing_profiles:
                failures.append("missing profiles: " + ", ".join(missing_profiles))
            if missing_decisions:
                failures.append("missing decisions: " + ", ".join(missing_decisions))
            raise ValueError("compatible paper coverage is incomplete; " + "; ".join(failures))
        return [(profiles[paper_id], decisions[paper_id]) for paper_id in requested]

    @staticmethod
    def paper_index(papers: Iterable[PaperRecord]) -> Mapping[str, PaperRecord]:
        """Index paper metadata and reject duplicate paper records."""
        indexed: dict[str, PaperRecord] = {}
        for paper in papers:
            if paper.paper_id in indexed:
                raise ValueError(f"duplicate paper metadata: {paper.paper_id}")
            indexed[paper.paper_id] = paper
        return indexed

    def build(
        self,
        method_coverage: PaperMethodCoverageReport,
        executable_coverage: ExecutablePaperCoverageBaseline,
        papers: Iterable[PaperRecord],
        recipes: Iterable[RecipeSpec] = (),
        *,
        expected_compatible_count: int | None = None,
    ) -> PaperExecutionInventory:
        """Build one inventory row for every compatible paper.

        The executable coverage report is used only to establish the frozen
        compatibility denominator and runtime evidence.  It never collapses
        paper records by canonical component.
        """
        compatible_ids = executable_coverage.denominators["yolo26_compatible_papers"].paper_ids
        if expected_compatible_count is not None and len(compatible_ids) != expected_compatible_count:
            raise ValueError(
                "compatible paper denominator changed: "
                f"expected {expected_compatible_count}, got {len(compatible_ids)}"
            )
        pairs = self.compatible_method_pairs(method_coverage, compatible_ids)
        paper_by_id = self.paper_index(papers)
        missing_papers = sorted(set(compatible_ids) - set(paper_by_id))
        if missing_papers:
            raise ValueError(
                "compatible paper metadata is incomplete: " + ", ".join(missing_papers)
            )
        coverage_by_id = {item.paper_id: item for item in executable_coverage.entries}
        recipe_list = list(recipes)
        records = [
            self._build_record(
                profile,
                decision,
                coverage_by_id[profile.paper_id],
                paper_by_id[profile.paper_id],
                recipe_list,
                source_method_coverage_hash=executable_coverage.source_method_coverage_hash,
            )
            for profile, decision in pairs
        ]
        generic_counts = {
            component_id: sum(component_id in item.generic_component_ids for item in records)
            for component_id in sorted(self.generic_component_ids)
            if any(component_id in item.generic_component_ids for item in records)
        }
        inventory = PaperExecutionInventory(
            source_method_coverage_hash=executable_coverage.source_method_coverage_hash,
            source_maturity_hash=executable_coverage.source_maturity_hash,
            all_paper_count=executable_coverage.denominators["all_papers"].paper_count,
            compatible_paper_count=len(records),
            exact_reproduction_candidates=executable_coverage.denominators[
                "exact_reproduction_candidates"
            ].paper_count,
            generic_mechanism_counts=generic_counts,
            records=records,
        )
        return inventory.with_hash()

    def _build_record(
        self,
        profile: PaperMethodProfile,
        decision: PaperImplementationDecision,
        coverage: PaperExecutableCoverageEntry,
        paper: PaperRecord,
        recipes: Sequence[RecipeSpec],
        *,
        source_method_coverage_hash: str,
    ) -> PaperExecutionSpec:
        canonical = sorted(set(coverage.canonical_mechanisms or decision.canonical_component_ids))
        generic = sorted(set(canonical) & self.generic_component_ids)
        paper_specific = sorted(set(canonical) - set(generic))
        recipe_ids = sorted({recipe.recipe_id for recipe in recipes if set(recipe.component_ids) & set(canonical)})
        required_evidence = self._required_evidence(
            profile,
            decision,
            coverage,
            paper_specific=paper_specific,
            recipe_ids=recipe_ids,
        )
        disposition, reason = self._disposition(
            profile,
            decision,
            coverage,
            paper_specific=paper_specific,
            recipe_ids=recipe_ids,
        )
        required_protocol = {
            "imgsz": 640,
            "datasets": sorted(set(paper.datasets)),
            "paper_protocol_constraints": profile.protocol_constraints,
            "source": "paper_profile_and_yolo26_fixed_protocol",
        }
        required_checkpoints = self._required_checkpoints(profile, recipes, canonical)
        fingerprint_payload = {
            "paper_id": profile.paper_id,
            "profile_id": profile.profile_id,
            "canonical_component_ids": canonical,
            "paper_specific_mechanism_ids": paper_specific,
            "recipe_ids": recipe_ids,
            "required_dataset_protocol": required_protocol,
            "required_checkpoints": required_checkpoints,
            "source_method_coverage_hash": source_method_coverage_hash,
        }
        return PaperExecutionSpec(
            paper_id=profile.paper_id,
            profile_id=profile.profile_id,
            title=paper.title,
            source_locations=sorted(set(profile.source_locations) | set(coverage.source_locations)),
            original_method_name=profile.method_name if profile.method_name != "unknown" else (profile.method_names[0] if profile.method_names else "unknown"),
            original_method_family=paper.detector_family or (paper.task_families[0] if paper.task_families else "unknown"),
            canonical_component_ids=canonical,
            paper_specific_mechanism_ids=paper_specific,
            generic_component_ids=generic,
            adaptation_mode=profile.adaptation_mode,
            exact_reproduction_possible=coverage.exact_reproduction_possible,
            required_dataset_protocol=required_protocol,
            required_checkpoints=required_checkpoints,
            required_evidence=required_evidence,
            recipe_ids=recipe_ids,
            execution_fingerprint=_fingerprint(fingerprint_payload),
            current_disposition=disposition,
            disposition_reason=reason,
            reusable_adapter_ids=sorted(set(coverage.reusable_adapter_candidates)),
            runtime_ready_adapters=sorted(set(coverage.runtime_ready_adapters)),
        )

    def _disposition(
        self,
        profile: PaperMethodProfile,
        decision: PaperImplementationDecision,
        coverage: PaperExecutableCoverageEntry,
        *,
        paper_specific: list[str],
        recipe_ids: list[str],
    ) -> tuple[str, str]:
        if not paper_specific:
            if coverage.blocking_fields or not coverage.canonical_mechanisms:
                return "evidence_recovery", "paper-specific mechanism evidence is incomplete"
            return "implementation_request", "canonical mechanism is generic; paper-specific implementation is unresolved"
        if coverage.compatibility_class in {"incompatible", "separate_detector_family"}:
            return "incompatible", coverage.exclusion_reason or "YOLO26 compatibility contract rejects this route"
        if not recipe_ids:
            return "implementation_request", "no local recipe is bound to the paper-specific mechanism"
        if coverage.runtime_ready_adapters and not coverage.blocking_fields:
            return "runtime_ready", "paper-specific mechanism and runtime evidence are available"
        if coverage.reusable_adapter_candidates:
            return "blocked_runtime", "; ".join(coverage.blocking_fields) or "runtime readiness evidence is incomplete"
        return "evidence_recovery", "; ".join(coverage.blocking_fields) or "adapter evidence is incomplete"

    @staticmethod
    def _required_evidence(
        profile: PaperMethodProfile,
        decision: PaperImplementationDecision,
        coverage: PaperExecutableCoverageEntry,
        *,
        paper_specific: list[str],
        recipe_ids: list[str],
    ) -> list[str]:
        evidence = [
            "paper_specific_mechanism_evidence",
            "yolo26_compatibility_evidence",
            "matched_baseline_protocol",
        ]
        if recipe_ids:
            evidence.append("recipe_contract_and_runtime_payload")
        if coverage.runtime_ready_adapters:
            evidence.append("runtime_adapter_maturity_artifact")
        if not paper_specific:
            evidence.append("paper_specific_method_description")
        if profile.protocol_constraints:
            evidence.append("paper_protocol_constraints")
        for gap in decision.adaptation_gaps:
            if gap.severity == "blocking":
                evidence.extend(gap.required_evidence)
        return sorted(set(evidence))

    @staticmethod
    def _required_checkpoints(
        profile: PaperMethodProfile,
        recipes: Sequence[RecipeSpec],
        canonical: list[str],
    ) -> list[str]:
        checkpoints: set[str] = set()
        for recipe in recipes:
            if set(recipe.component_ids) & set(canonical):
                for key, value in recipe.train_overrides.items():
                    if "checkpoint" in key or key in {"teacher", "student"}:
                        checkpoints.add(f"{key}:{value}")
        return sorted(checkpoints)


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_paper_execution_inventory_markdown(
    inventory: PaperExecutionInventory,
) -> str:
    """Render the paper-level denominator and every execution disposition."""
    lines = [
        "# Paper Execution Inventory",
        "",
        "This is a paper-level execution ledger. Shared adapters do not imply "
        "exact reproduction of each source paper.",
        "",
        "## Frozen Counts",
        "",
        f"- All papers: {inventory.all_paper_count}",
        f"- YOLO26-compatible papers: {inventory.compatible_paper_count}",
        f"- Exact reproduction candidates: {inventory.exact_reproduction_candidates}",
        f"- Source method coverage hash: `{inventory.source_method_coverage_hash}`",
        f"- Source maturity hash: `{inventory.source_maturity_hash or 'not_available'}`",
        f"- Inventory hash: `{inventory.inventory_hash}`",
        "",
        "## Generic Mechanisms",
        "",
        "| Canonical mechanism | Papers |",
        "|---|---:|",
    ]
    for component_id, count in sorted(inventory.generic_mechanism_counts.items()):
        lines.append(f"| `{_cell(component_id)}` | {count} |")
    lines.extend(
        [
            "",
            "## Per-Paper Execution",
            "",
            "| Paper | Title | Specific mechanisms | Generic mechanisms | "
            "Recipes | Disposition | Reason | Fingerprint |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in inventory.records:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_cell(item.paper_id)}`",
                    _cell(item.title),
                    _items(item.paper_specific_mechanism_ids),
                    _items(item.generic_component_ids),
                    _items(item.recipe_ids),
                    item.current_disposition,
                    _cell(item.disposition_reason),
                    f"`{item.execution_fingerprint}`",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_paper_execution_inventory_artifacts(
    inventory: PaperExecutionInventory,
    *,
    yaml_path: Path | str,
    markdown_path: Path | str,
) -> tuple[Path, Path]:
    """Write the machine-readable inventory and its audit rendering."""
    yaml_output = Path(yaml_path)
    markdown_output = Path(markdown_path)
    _atomic_text(
        yaml_output,
        yaml.safe_dump(
            inventory.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
        ),
    )
    _atomic_text(
        markdown_output,
        render_paper_execution_inventory_markdown(inventory),
    )
    return yaml_output, markdown_output


def _items(values: list[str]) -> str:
    return _cell("<br>".join(values) if values else "none")


def _cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "GENERIC_COMPONENT_IDS",
    "PaperExecutionInventoryBuilder",
    "render_paper_execution_inventory_markdown",
    "write_paper_execution_inventory_artifacts",
]
