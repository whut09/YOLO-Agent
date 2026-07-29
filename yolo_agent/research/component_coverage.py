"""Coverage accounting for paper component aliases and local adapters."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.components.maturity import MaturityName
from yolo_agent.research.component_aliases import ComponentAliasResolution, ComponentAliasResolver
from yolo_agent.research.schemas import PaperRecord


class ComponentCoverageReport(BaseModel, YAMLModelMixin):
    """Machine-readable coverage report for a frozen paper catalog."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_coverage.v2"
    paper_count: int = Field(default=0, ge=0)
    total_paper_components: int = Field(default=0, ge=0)
    resolved: int = Field(default=0, ge=0)
    unresolved: int = Field(default=0, ge=0)
    executable: int = Field(default=0, ge=0)
    adapter_required: int = Field(default=0, ge=0)
    incompatible: int = Field(default=0, ge=0)
    canonical_component_count: int = Field(default=0, ge=0)
    local_component_count: int = Field(default=0, ge=0)
    metadata_only: int = Field(default=0, ge=0)
    recipe_idea_only: int = Field(default=0, ge=0)
    adapter_implemented: int = Field(default=0, ge=0)
    runtime_integrated: int = Field(default=0, ge=0)
    unit_tested: int = Field(default=0, ge=0)
    smoke_passed: int = Field(default=0, ge=0)
    gpu_certified: int = Field(default=0, ge=0)
    pilot_reproduced: int = Field(default=0, ge=0)
    full_reproduced: int = Field(default=0, ge=0)
    confirmed_multi_seed: int = Field(default=0, ge=0)
    implemented_adapter_count: int = Field(default=0, ge=0)
    failed_artifact_count: int = Field(default=0, ge=0)
    mock_artifact_count: int = Field(default=0, ge=0)
    components_by_maturity: dict[MaturityName, list[str]] = Field(default_factory=dict)
    real_adapter_components: list[str] = Field(default_factory=list)
    paper_prior_only_components: list[str] = Field(default_factory=list)
    unresolved_components: list[str] = Field(default_factory=list)
    canonical_paper_sources: dict[str, list[str]] = Field(default_factory=dict)
    resolutions: list[ComponentAliasResolution] = Field(default_factory=list)


class ComponentCoverageAnalyzer:
    """Resolve paper components and summarize implementation coverage."""

    def __init__(self, resolver: ComponentAliasResolver) -> None:
        self.resolver = resolver

    def analyze_papers(self, papers: list[PaperRecord]) -> ComponentCoverageReport:
        resolutions = [
            self.resolver.resolve(component_id, source_paper_ids=[paper.paper_id])
            for paper in papers
            for component_id in paper.component_ids
        ]
        return self.analyze_resolutions(resolutions, paper_count=len(papers))

    def analyze_resolutions(
        self,
        resolutions: list[ComponentAliasResolution],
        *,
        paper_count: int = 0,
    ) -> ComponentCoverageReport:
        canonical_sources: dict[str, set[str]] = {}
        executable: set[str] = set()
        adapter_required: set[str] = set()
        incompatible: set[str] = set()
        real_adapters: set[str] = set()
        paper_prior_only: set[str] = set()
        unresolved_components: set[str] = set()

        for result in resolutions:
            if not result.resolved:
                unresolved_components.add(result.paper_component_id)
                continue
            for mapping in result.mappings:
                canonical_sources.setdefault(mapping.canonical_component_id, set()).update(result.source_paper_ids)
                if mapping.adapter_verified:
                    real_adapters.add(mapping.canonical_component_id)
                else:
                    paper_prior_only.add(mapping.canonical_component_id)
                if mapping.executable:
                    executable.add(mapping.canonical_component_id)
                elif mapping.yolo26_compatibility == "incompatible":
                    incompatible.add(mapping.canonical_component_id)
                else:
                    adapter_required.add(mapping.canonical_component_id)

        resolved_count = sum(1 for item in resolutions if item.resolved)
        local_contracts = list(self.resolver.contracts.values())
        by_maturity: dict[MaturityName, list[str]] = {}
        for contract in local_contracts:
            by_maturity.setdefault(contract.maturity, []).append(contract.component_id)
        by_maturity = {
            key: sorted(value)
            for key, value in sorted(by_maturity.items())
        }
        maturity_counts = {
            name: len(by_maturity.get(name, []))
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
        return ComponentCoverageReport(
            paper_count=paper_count,
            total_paper_components=len(resolutions),
            resolved=resolved_count,
            unresolved=len(resolutions) - resolved_count,
            executable=len(executable),
            adapter_required=len(adapter_required),
            incompatible=len(incompatible),
            canonical_component_count=len(canonical_sources),
            local_component_count=len(local_contracts),
            **maturity_counts,
            implemented_adapter_count=sum(
                1
                for contract in local_contracts
                if self.resolver.adapter_verified(contract.component_id)
            ),
            failed_artifact_count=sum(
                item.status == "failed"
                for contract in local_contracts
                for item in contract.maturity_artifacts
            ),
            mock_artifact_count=sum(
                item.mock
                for contract in local_contracts
                for item in contract.maturity_artifacts
            ),
            components_by_maturity=by_maturity,
            real_adapter_components=sorted(real_adapters),
            paper_prior_only_components=sorted(paper_prior_only),
            unresolved_components=sorted(unresolved_components),
            canonical_paper_sources={key: sorted(value) for key, value in sorted(canonical_sources.items())},
            resolutions=resolutions,
        )

    def write_report(
        self,
        path: Path | str,
        report: ComponentCoverageReport,
    ) -> Path:
        target = Path(path)
        report.to_yaml(target, exclude_none=True, sort_keys=False)
        return target


__all__ = ["ComponentCoverageAnalyzer", "ComponentCoverageReport"]
