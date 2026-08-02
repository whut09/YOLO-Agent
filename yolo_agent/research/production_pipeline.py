"""Offline Paper Intelligence production pipeline and resumable stage state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field

from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
    current_code_commit,
    installed_ultralytics_version,
)
from yolo_agent.components.maturity_registry_schemas import ComponentEvidenceOverlay
from yolo_agent.components.yolo26_compatibility import YOLO26CompatibilityChecker
from yolo_agent.research.component_aliases import ComponentAliasResolution, ComponentAliasResolver
from yolo_agent.research.component_coverage import ComponentCoverageAnalyzer
from yolo_agent.research.cached_code_metadata import CachedCodeMetadataLoader
from yolo_agent.research.component_extractor import (
    ComponentExtractionBundle,
    ComponentExtractionResult,
    ExtractedClaim,
    ExtractedComponent,
    SourceLocation,
)
from yolo_agent.research.note_parser import PaperEvidenceSummary, PaperNoteParser
from yolo_agent.research.method_profiles import PaperMethodProfileBuilder
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.mechanism_cluster_report import (
    build_mechanism_cluster_report,
)
from yolo_agent.research.mechanism_cluster_markdown import (
    write_mechanism_cluster_markdown,
)
from yolo_agent.research.paper_mechanism_clusterer import PaperMechanismClusterer
from yolo_agent.research.paper_method_evidence_report import (
    build_method_evidence_coverage_report,
    write_method_evidence_coverage_markdown,
)
from yolo_agent.research.executable_coverage import ExecutablePaperCoverageAuditor
from yolo_agent.research.executable_coverage_report import (
    write_executable_coverage_artifacts,
)
from yolo_agent.research.maturity_snapshot import (
    EffectiveComponentMaturityManifest,
    FrozenComponentMaturity,
    FrozenMaturityArtifact,
)
from yolo_agent.research.paper_classifier import PaperClassification, PaperClassifier
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.reproduction_state import ReproductionStatus
from yolo_agent.research.schemas import ComponentTaxonomy, PaperRecord
from yolo_agent.research.snapshot import ResearchMaturitySummary, freeze_research_snapshot
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.recipes.schemas import AtomicRecipe, RecipeSpec
from yolo_agent.resources import ResourcePaths


class ResearchStageState(BaseModel):
    status: str = "pending"
    input_hash: str | None = None
    output_hash: str | None = None
    output_path: str | None = None
    message: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchProductionState(BaseModel):
    schema_version: str = "research_production_state.v1"
    status: str = "initialized"
    stages: dict[str, ResearchStageState] = Field(default_factory=dict)
    snapshot_hash: str | None = None
    last_error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StoredExtraction(BaseModel):
    paper_id: str
    input_sha256: str
    result: ComponentExtractionResult


class ResearchProductionResult(BaseModel):
    status: str
    snapshot_hash: str | None = None
    snapshot_path: str | None = None
    stage_status: dict[str, str] = Field(default_factory=dict)
    paper_count: int = 0
    component_count: int = 0
    recipe_count: int = 0
    paper_intelligence: str = "unavailable"
    unavailable_reason: str | None = None
    maturity_summary: ResearchMaturitySummary = Field(default_factory=ResearchMaturitySummary)
    errors: list[str] = Field(default_factory=list)


class ResearchProductionPipeline:
    """Build a frozen, offline research snapshot without training side effects."""

    STAGES = (
        "sync",
        "deduplicate",
        "classify",
        "parse_paper_evidence",
        "extract_components",
        "resolve_aliases",
        "component_coverage",
        "map_method_profiles",
        "executable_coverage_baseline",
        "contract_draft",
        "compatibility_review",
        "recipe_generation",
        "reproduction_queue",
        "snapshot",
    )

    def __init__(
        self,
        research_root: Path | str = "research",
        *,
        taxonomy_path: Path | str | None = None,
        component_compatibility_path: Path | str | None = None,
        analyzer: Any | None = None,
        registry_factory: Callable[[Path], PaperRegistry] = PaperRegistry,
        alias_resolver: ComponentAliasResolver | None = None,
        note_parser: PaperNoteParser | None = None,
        maturity_registry: ComponentMaturityRegistry | Path | str | None = None,
        maturity_protocol_hash: str | None = None,
        maturity_ultralytics_version: str | None = None,
        cached_code_root: Path | str | None = None,
    ) -> None:
        self.root = Path(research_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.root / "production"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.artifacts_dir / "production_state.yaml"
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else ResourcePaths.COMPONENT_TAXONOMY
        self.component_compatibility_path = (
            Path(component_compatibility_path)
            if component_compatibility_path
            else ResourcePaths.COMPONENT_COMPATIBILITY
        )
        self.analyzer = analyzer
        self.registry_factory = registry_factory
        self.alias_resolver = alias_resolver or ComponentAliasResolver.from_yaml()
        self.note_parser = note_parser or PaperNoteParser(
            ledger_path=self.artifacts_dir / "research_decision_ledger.jsonl"
        )
        self.maturity_registry = (
            maturity_registry
            if isinstance(maturity_registry, ComponentMaturityRegistry)
            else ComponentMaturityRegistry(maturity_registry)
            if maturity_registry is not None
            else None
        )
        self.maturity_protocol_hash = maturity_protocol_hash
        self.maturity_ultralytics_version = maturity_ultralytics_version
        self.cached_code_root = (
            Path(cached_code_root) if cached_code_root is not None else None
        )

    def run(
        self,
        *,
        sync: bool = False,
        scout: Any | None = None,
        since: datetime | None = None,
        year_from: int | None = None,
        force: bool = False,
        snapshot_source: dict[str, str | None] | None = None,
        unavailable_reason_override: str | None = None,
        include_local_implementations: bool = False,
    ) -> ResearchProductionResult:
        state = self.load_state()
        result = ResearchProductionResult(status="running")
        previous_method_coverage, baseline_snapshot_hash = (
            _load_latest_frozen_method_coverage(self.root)
        )
        try:
            if sync:
                if scout is None:
                    raise ValueError("sync requested but PaperScout was not supplied")
                scout.sync(since=since, year_from=year_from)
                self._complete(state, "sync", self.root / "paper_scout_state.json", "Metadata sources synchronized.")
            else:
                self._skip(state, "sync", "Offline mode: using existing local registry.")

            registry = self.registry_factory(self.root)
            registry.deduplicate()
            self._complete(state, "deduplicate", registry.papers_path, "Local registry deduplicated.")

            papers = registry.list()
            taxonomy = ComponentTaxonomy.model_validate(
                yaml.safe_load(self.taxonomy_path.read_text(encoding="utf-8-sig")) or {}
            )
            classifications = self._classify(papers)
            classifications_path = self.artifacts_dir / "classifications.jsonl"
            _write_jsonl(classifications_path, [item.model_dump(mode="json") for item in classifications])
            self._complete(state, "classify", classifications_path, f"Classified {len(classifications)} papers.")

            note_root = _note_root((snapshot_source or {}).get("catalog_path"))
            paper_evidence = [
                _parse_paper_evidence_non_blocking(
                    self.note_parser,
                    paper=paper,
                    taxonomy=taxonomy,
                    note_path=_resolve_note_path(paper, note_root),
                )
                for paper in papers
            ]
            paper_evidence_path = self.artifacts_dir / "paper_evidence_summaries.jsonl"
            _write_jsonl(paper_evidence_path, [item.model_dump(mode="json") for item in paper_evidence])
            self._complete(
                state,
                "parse_paper_evidence",
                paper_evidence_path,
                f"Parsed paper evidence for {len(paper_evidence)} papers.",
            )

            extractions = self._extract(papers, taxonomy, force=force)
            extractions_path = self.artifacts_dir / "component_extractions.jsonl"
            _write_jsonl(extractions_path, [item.model_dump(mode="json") for item in extractions])
            self._complete(state, "extract_components", extractions_path, "Component extraction completed offline.")

            extracted_components = [component for item in extractions for component in item.result.extracted_components]
            alias_resolutions = [
                self.alias_resolver.resolve(
                    component.component_id,
                    source_paper_ids=sorted({claim.paper_id for claim in component.claimed_effects}),
                )
                for component in extracted_components
            ]
            alias_path = self.artifacts_dir / "component_alias_resolutions.yaml"
            _write_yaml(alias_path, {
                "schema_version": "component_alias_resolution.v1",
                "resolutions": [item.model_dump(mode="json") for item in alias_resolutions],
            })
            self._complete(
                state,
                "resolve_aliases",
                alias_path,
                f"Resolved {len(alias_resolutions)} paper component identifiers.",
            )

            coverage = ComponentCoverageAnalyzer(self.alias_resolver).analyze_resolutions(
                alias_resolutions,
                paper_count=len(papers),
            )
            coverage_path = self.artifacts_dir / "component_coverage_report.yaml"
            _write_yaml(coverage_path, coverage.model_dump(mode="json"))
            self._complete(
                state,
                "component_coverage",
                coverage_path,
                f"Audited {coverage.canonical_component_count} canonical components.",
            )

            canonical_components = _canonicalize_components(extracted_components, alias_resolutions)
            contracts = _contract_drafts(canonical_components)
            selected_overlays: dict[str, ComponentEvidenceOverlay] = {}
            if include_local_implementations:
                contracts, selected_overlays = _merge_local_component_contracts(
                    contracts,
                    maturity_registry=self.maturity_registry,
                    protocol_hash=self.maturity_protocol_hash,
                    ultralytics_version=self.maturity_ultralytics_version,
                )
            contracts, maturity_evidence = _materialize_maturity_evidence(
                contracts,
                self.artifacts_dir / "component_maturity_evidence",
            )
            effective_alias_resolver = ComponentAliasResolver(
                self.alias_resolver.config,
                contracts=contracts,
            )
            cached_code = (
                {
                    paper.paper_id: CachedCodeMetadataLoader(
                        self.cached_code_root
                    ).load(paper)
                    for paper in papers
                }
                if self.cached_code_root is not None
                else {}
            )
            method_coverage = PaperMethodProfileBuilder(
                effective_alias_resolver
            ).build(
                papers,
                evidence_summaries={item.paper_id: item for item in paper_evidence},
                cached_metadata={
                    paper_id: bundle.extractor_sources()
                    for paper_id, bundle in cached_code.items()
                },
            )
            method_evidence_path = (
                self.artifacts_dir / "paper_method_evidence.jsonl"
            )
            _write_jsonl(
                method_evidence_path,
                [
                    profile.structured_method_evidence.model_dump(mode="json")
                    for profile in method_coverage.profiles
                    if profile.structured_method_evidence is not None
                ],
            )
            cached_code_path = self.artifacts_dir / "cached_code_metadata.yaml"
            _write_yaml(
                cached_code_path,
                {
                    "schema_version": "cached_code_metadata.v1",
                    "offline_only": True,
                    "papers": {
                        paper_id: bundle.model_dump(mode="json")
                        for paper_id, bundle in sorted(cached_code.items())
                    },
                },
            )
            method_coverage_path = self.artifacts_dir / "paper_method_coverage.yaml"
            _write_yaml(
                method_coverage_path,
                method_coverage.model_dump(mode="json", exclude_none=True),
            )
            mechanism_clusterer = PaperMechanismClusterer()
            mechanism_matches, mechanism_conflicts = mechanism_clusterer.cluster(
                method_coverage
            )
            mechanism_report = build_mechanism_cluster_report(
                method_coverage,
                config=mechanism_clusterer.config,
                matches=mechanism_matches,
                conflicts=mechanism_conflicts,
            )
            mechanism_report_path = (
                self.artifacts_dir / "paper_mechanism_clusters.yaml"
            )
            mechanism_report.to_yaml(
                mechanism_report_path,
                exclude_none=True,
                sort_keys=False,
            )
            mechanism_markdown_path = (
                self.artifacts_dir / "paper_mechanism_clusters.md"
            )
            write_mechanism_cluster_markdown(
                mechanism_report,
                mechanism_markdown_path,
            )
            method_evidence_coverage = build_method_evidence_coverage_report(
                method_coverage,
            )
            method_evidence_coverage_path = (
                self.artifacts_dir / "paper_method_evidence_coverage.yaml"
            )
            method_evidence_coverage.to_yaml(
                method_evidence_coverage_path,
                exclude_none=True,
                sort_keys=False,
            )
            method_evidence_delta = build_method_evidence_coverage_report(
                method_coverage,
                previous=previous_method_coverage,
                baseline_snapshot_hash=baseline_snapshot_hash,
            )
            method_evidence_delta_path = (
                self.artifacts_dir / "paper_method_evidence_delta.yaml"
            )
            method_evidence_delta.to_yaml(
                method_evidence_delta_path,
                exclude_none=True,
                sort_keys=False,
            )
            method_evidence_coverage_markdown_path = (
                self.artifacts_dir / "paper_method_evidence_delta.md"
            )
            write_method_evidence_coverage_markdown(
                method_evidence_delta,
                method_evidence_coverage_markdown_path,
            )
            self._complete(
                state,
                "map_method_profiles",
                method_coverage_path,
                f"Mapped {method_coverage.profile_count} paper methods against effective component maturity.",
            )
            maturity_summary = _maturity_summary(contracts)
            effective_maturity_path = self.artifacts_dir / "effective_component_maturity.yaml"
            effective_maturity = _effective_maturity_manifest(
                contracts,
                selected_overlays=selected_overlays,
                maturity_evidence=maturity_evidence,
                ultralytics_version=(
                    self.maturity_ultralytics_version
                    or installed_ultralytics_version()
                ),
            )
            effective_maturity.to_yaml(
                effective_maturity_path,
                exclude_none=True,
                sort_keys=False,
            )
            executable_coverage = ExecutablePaperCoverageAuditor(
                contracts={item.component_id: item for item in contracts},
                maturity=effective_maturity,
            ).build(
                method_coverage,
                source_method_coverage_hash=_file_or_dir_hash(
                    method_coverage_path
                ),
                source_taxonomy_hash=_file_or_dir_hash(self.taxonomy_path),
            )
            executable_coverage_path = (
                self.artifacts_dir / "coverage_baseline.yaml"
            )
            executable_coverage_markdown_path = (
                self.artifacts_dir / "coverage_baseline.md"
            )
            write_executable_coverage_artifacts(
                executable_coverage,
                yaml_path=executable_coverage_path,
                markdown_path=executable_coverage_markdown_path,
            )
            self._complete(
                state,
                "executable_coverage_baseline",
                executable_coverage_path,
                "Audited four executable paper coverage denominators.",
            )
            contracts_path = self.artifacts_dir / "component_contracts.yaml"
            _write_yaml(contracts_path, {"schema_version": "component_contract_registry.v1", "components": {item.component_id: item.model_dump(mode="json") for item in contracts}})
            self._complete(state, "contract_draft", contracts_path, f"Drafted {len(contracts)} component contracts.")

            compatibility = _compatibility_reviews(contracts)
            compatibility_path = self.artifacts_dir / "compatibility_reviews.yaml"
            _write_yaml(compatibility_path, {"schema_version": "compatibility_review.v1", "imgsz": 640, "components": compatibility})
            self._complete(state, "compatibility_review", compatibility_path, "YOLO26 compatibility reviewed.")

            recipes = _recipe_drafts(canonical_components, contracts)
            if include_local_implementations:
                recipes = _merge_local_recipes(recipes, contracts)
            recipes_path = self.artifacts_dir / "recipes.yaml"
            _write_yaml(recipes_path, {"schema_version": "recipe_registry.v1", "recipes": [item.model_dump(mode="json") for item in recipes]})
            self._complete(state, "recipe_generation", recipes_path, f"Generated {len(recipes)} metadata-only recipes.")

            reproduction_queue = _reproduction_queue(canonical_components, contracts, recipes)
            queue_path = self.artifacts_dir / "reproduction_queue.yaml"
            _write_yaml(queue_path, {"schema_version": "reproduction_queue.v1", "items": reproduction_queue})
            self._complete(state, "reproduction_queue", queue_path, "Reproduction queue drafted; no training enqueued.")

            snapshot_artifacts = {
                "papers": registry.papers_path,
                "classifications": classifications_path,
                "paper_evidence": paper_evidence_path,
                "paper_method_evidence": method_evidence_path,
                "paper_method_evidence_coverage": method_evidence_coverage_path,
                "cached_code_metadata": cached_code_path,
                "component_extractions": extractions_path,
                "component_alias_resolutions": alias_path,
                "component_taxonomy": self.taxonomy_path,
                "component_coverage": coverage_path,
                "paper_method_coverage": method_coverage_path,
                "paper_mechanism_clusters": mechanism_report_path,
                "paper_mechanism_cluster_taxonomy": (
                    ResourcePaths.PAPER_MECHANISM_CLUSTERS
                ),
                "executable_coverage_baseline": executable_coverage_path,
                "executable_coverage_report": executable_coverage_markdown_path,
                "effective_component_maturity": effective_maturity_path,
                "component_contracts": contracts_path,
                "compatibility_reviews": compatibility_path,
                "recipes": recipes_path,
                "reproduction_queue": queue_path,
                **maturity_evidence,
            }
            snapshot, snapshot_dir = freeze_research_snapshot(
                self.root,
                snapshot_artifacts,
                paper_count=len(papers),
                component_count=len(contracts),
                recipe_count=len(recipes),
                papers_version=_papers_version(papers),
                alias_resolution_version=_file_or_dir_hash(alias_path),
                coverage_version=_combined_hash(
                    coverage_path,
                    method_coverage_path,
                    mechanism_report_path,
                    ResourcePaths.PAPER_MECHANISM_CLUSTERS,
                    executable_coverage_path,
                    executable_coverage_markdown_path,
                ),
                paper_evidence_version=_combined_hash(
                    paper_evidence_path,
                    method_evidence_path,
                    cached_code_path,
                ),
                paper_method_coverage_version=_file_or_dir_hash(method_coverage_path),
                effective_maturity_version=_file_or_dir_hash(effective_maturity_path),
                maturity_summary=maturity_summary,
                source_repository=(snapshot_source or {}).get("source_repository"),
                source_commit=(snapshot_source or {}).get("source_commit"),
                source_catalog_hash=(snapshot_source or {}).get("source_catalog_hash"),
                importer_version=(snapshot_source or {}).get("importer_version"),
                unavailable_reason_override=unavailable_reason_override,
            )
            state.snapshot_hash = snapshot.snapshot_hash
            self._complete(state, "snapshot", snapshot_dir / "snapshot.yaml", f"Frozen snapshot {snapshot.snapshot_hash}.")
            state.status = "completed"
            state.last_error = None
            self.save_state(state)
            result.status = "completed"
            result.snapshot_hash = snapshot.snapshot_hash
            result.snapshot_path = snapshot_dir.as_posix()
            result.paper_count = len(papers)
            result.component_count = len(contracts)
            result.recipe_count = len(recipes)
            result.paper_intelligence = snapshot.paper_intelligence
            result.unavailable_reason = snapshot.unavailable_reason
            result.maturity_summary = snapshot.maturity_summary
        except Exception as exc:
            state.status = "failed"
            state.last_error = str(exc)
            self.save_state(state)
            result.status = "failed"
            result.errors.append(str(exc))
        result.stage_status = {name: item.status for name, item in state.stages.items()}
        return result

    def load_state(self) -> ResearchProductionState:
        if self.state_path.is_file():
            return ResearchProductionState.model_validate(yaml.safe_load(self.state_path.read_text(encoding="utf-8-sig")) or {})
        return ResearchProductionState(stages={name: ResearchStageState() for name in self.STAGES})

    def save_state(self, state: ResearchProductionState) -> Path:
        state.updated_at = datetime.now(timezone.utc)
        _write_yaml(self.state_path, state.model_dump(mode="json"))
        return self.state_path

    def _classify(self, papers: list[PaperRecord]) -> list[PaperClassification]:
        classifier = PaperClassifier()
        return [classifier.classify(paper) for paper in papers]

    def _extract(self, papers: list[PaperRecord], taxonomy: ComponentTaxonomy, *, force: bool) -> list[StoredExtraction]:
        existing: dict[str, StoredExtraction] = {}
        path = self.artifacts_dir / "component_extractions.jsonl"
        if path.is_file() and not force:
            for raw in _read_jsonl(path):
                item = StoredExtraction.model_validate(raw)
                existing[item.paper_id] = item
        outputs: list[StoredExtraction] = []
        for paper in papers:
            input_hash = _paper_hash(paper)
            if paper.paper_id in existing and existing[paper.paper_id].input_sha256 == input_hash:
                outputs.append(existing[paper.paper_id])
                continue
            if self.analyzer is None:
                result = _curated_component_extraction(paper, taxonomy)
            else:
                result = self.analyzer.analyze(paper=paper, taxonomy=taxonomy)
            outputs.append(StoredExtraction(paper_id=paper.paper_id, input_sha256=input_hash, result=result))
        return outputs

    def _complete(self, state: ResearchProductionState, stage: str, output: Path, message: str) -> None:
        state.stages.setdefault(stage, ResearchStageState())
        state.stages[stage] = ResearchStageState(
            status="completed",
            output_hash=_file_or_dir_hash(output),
            output_path=output.resolve().as_posix(),
            message=message,
        )
        self.save_state(state)

    def _skip(self, state: ResearchProductionState, stage: str, message: str) -> None:
        state.stages.setdefault(stage, ResearchStageState())
        state.stages[stage] = ResearchStageState(status="skipped", message=message)
        self.save_state(state)


def _contract_drafts(components: list[ExtractedComponent]) -> list[ComponentContract]:
    contracts: dict[str, ComponentContract] = {}
    for component in components:
        paper_ids = sorted({claim.paper_id for claim in component.claimed_effects})
        constraints = {
            "source_component_category": component.component_category,
            "training_only": component.training_only,
            "inference_only": component.inference_only,
        }
        existing = contracts.get(component.component_id)
        contract = ComponentContract(
            component_id=component.component_id,
            display_name=component.name,
            category=component.component_category,
            source_papers=paper_ids,
            insertion_point=component.insertion_point,
            supported_detector_families=["yolo26"],
            tensor_input_contract={"inputs": component.required_inputs, "compatibility_constraints": constraints},
            tensor_output_contract={"outputs": component.produced_outputs},
            training_only=component.training_only,
            inference_only=component.inference_only,
            changes_model_graph=component.component_category in {"backbone", "neck", "detection_head", "feature_pyramid"},
            fixed_imgsz_compatible="unknown",
            maturity="metadata_only",
            tests_required=["adapter_validation", "shape", "backward", "cpu_smoke"],
            known_risks=[*component.uncertainties, *component.implementation_notes],
        )
        if existing is not None:
            contract = contract.model_copy(update={
                "source_papers": sorted(set([*existing.source_papers, *contract.source_papers])),
                "known_risks": list(dict.fromkeys([*existing.known_risks, *contract.known_risks])),
            })
        contracts[component.component_id] = contract
    return sorted(contracts.values(), key=lambda item: item.component_id)


def _merge_local_component_contracts(
    paper_contracts: list[ComponentContract],
    *,
    maturity_registry: ComponentMaturityRegistry | None = None,
    protocol_hash: str | None = None,
    ultralytics_version: str | None = None,
) -> tuple[list[ComponentContract], dict[str, ComponentEvidenceOverlay]]:
    """Merge code-owned component maturity into the frozen research snapshot."""
    contracts = {item.component_id: item for item in paper_contracts}
    paths = [
        ResourcePaths.COMPONENT_COMPATIBILITY,
        *sorted(ResourcePaths.COMPONENTS_DIR.rglob("*.yaml")),
    ]
    for path in paths:
        if not path.is_file():
            continue
        try:
            local_contracts = load_contracts(path)
        except (KeyError, TypeError, ValueError):
            continue
        for local in local_contracts:
            paper = contracts.get(local.component_id)
            if paper is not None:
                local = local.model_copy(
                    update={
                        "source_papers": sorted(set(paper.source_papers) | set(local.source_papers)),
                        "known_risks": list(dict.fromkeys([*paper.known_risks, *local.known_risks])),
                    }
                )
            contracts[local.component_id] = local
    selected_overlays: dict[str, ComponentEvidenceOverlay] = {}
    if maturity_registry is not None:
        runtime_version = ultralytics_version or installed_ultralytics_version()
        for component_id, contract in list(contracts.items()):
            try:
                adapter_hash = adapter_source_hash(contract)
            except (AttributeError, ImportError, TypeError, ValueError):
                continue
            effective, _, overlay = maturity_registry.resolve(
                contract,
                adapter_hash=adapter_hash,
                ultralytics_version=runtime_version,
                protocol_hash=protocol_hash,
            )
            contracts[component_id] = effective
            if overlay is not None:
                selected_overlays[component_id] = overlay
    return (
        sorted(contracts.values(), key=lambda item: item.component_id),
        selected_overlays,
    )


def _effective_maturity_manifest(
    contracts: list[ComponentContract],
    *,
    selected_overlays: dict[str, ComponentEvidenceOverlay],
    maturity_evidence: dict[str, Path],
    ultralytics_version: str,
) -> EffectiveComponentMaturityManifest:
    artifact_names = {
        path.resolve(): name for name, path in maturity_evidence.items()
    }
    entries: list[FrozenComponentMaturity] = []
    for contract in contracts:
        try:
            adapter_hash = adapter_source_hash(contract)
        except (AttributeError, ImportError, TypeError, ValueError):
            continue
        overlay = selected_overlays.get(contract.component_id)
        artifacts = [
            FrozenMaturityArtifact(
                snapshot_artifact_name=artifact_names[artifact.artifact_path.resolve()],
                target_maturity=artifact.target_maturity,
                artifact_type=artifact.artifact_type,
                artifact_sha256=artifact.artifact_sha256,
                protocol_hash=artifact.protocol_hash,
                status=artifact.status,
                mock=artifact.mock,
            )
            for artifact in contract.maturity_artifacts
            if artifact.artifact_path.resolve() in artifact_names
        ]
        entries.append(FrozenComponentMaturity(
            component_id=contract.component_id,
            adapter_hash=adapter_hash,
            code_commit=(overlay.code_commit if overlay else current_code_commit()),
            ultralytics_version=(
                overlay.ultralytics_version if overlay else ultralytics_version
            ),
            protocol_hash=overlay.protocol_hash if overlay else "unavailable",
            overlay_identity_key=overlay.identity_key if overlay else None,
            overlay_evidence_hash=overlay.evidence_hash if overlay else None,
            effective_maturity=contract.maturity,
            runtime_execution_ready=contract.can_execute,
            artifacts=artifacts,
        ))
    return EffectiveComponentMaturityManifest(
        entries=sorted(entries, key=lambda item: item.component_id)
    )


def _materialize_maturity_evidence(
    contracts: list[ComponentContract],
    output_dir: Path,
) -> tuple[list[ComponentContract], dict[str, Path]]:
    """Copy valid machine evidence before freezing effective contracts."""
    updated_contracts: list[ComponentContract] = []
    snapshot_artifacts: dict[str, Path] = {}
    for contract in contracts:
        artifacts = []
        for index, artifact in enumerate(contract.maturity_artifacts):
            try:
                artifact.verify()
            except ValueError:
                continue
            suffix = "".join(artifact.artifact_path.suffixes) or ".dat"
            component = re.sub(r"[^A-Za-z0-9_.-]+", "_", contract.component_id)
            target = (
                output_dir
                / component
                / f"{artifact.target_maturity}.{artifact.artifact_sha256[:16]}{suffix}"
            )
            if not target.is_file() or _file_or_dir_hash(target) != artifact.artifact_sha256:
                _atomic_copy_file(artifact.artifact_path, target)
            copied = artifact.model_copy(update={"artifact_path": target.resolve()})
            artifacts.append(copied)
            snapshot_artifacts[
                f"component_maturity_{component}_{artifact.target_maturity}_{index}"
            ] = target
        updated_contracts.append(
            contract.model_copy(update={"maturity_artifacts": artifacts})
        )
    return updated_contracts, snapshot_artifacts


def _merge_local_recipes(
    paper_recipes: list[RecipeSpec],
    contracts: list[ComponentContract],
) -> list[RecipeSpec]:
    """Add reviewed local recipe bundles without consulting a live paper registry."""
    recipes = {(item.recipe_id, item.version): item for item in paper_recipes}
    paths = [
        ResourcePaths.RECIPE_BUNDLES,
        *sorted(ResourcePaths.RECIPES_DIR.glob("*.yaml")),
    ]
    for path in paths:
        if not path.is_file():
            continue
        try:
            local_registry = RecipeRegistry.from_path(path, component_contracts=contracts)
        except (KeyError, TypeError, ValueError):
            continue
        for recipe in local_registry.list():
            recipes[(recipe.recipe_id, recipe.version)] = recipe
    return sorted(recipes.values(), key=lambda item: (item.recipe_id, item.version))


def _canonicalize_components(
    components: list[ExtractedComponent],
    resolutions: list[ComponentAliasResolution],
) -> list[ExtractedComponent]:
    """Apply explicit mappings while keeping unresolved concepts metadata-only."""
    output: list[ExtractedComponent] = []
    for component, resolution in zip(components, resolutions, strict=True):
        if not resolution.resolved:
            output.append(component.model_copy(update={
                "implementation_notes": [
                    *component.implementation_notes,
                    "unresolved_component_alias: metadata_only; adapter status was not inferred",
                ],
            }))
            continue
        for mapping in resolution.mappings:
            output.append(component.model_copy(update={
                "component_id": mapping.canonical_component_id,
                "component_category": mapping.category,
                "insertion_point": mapping.insertion_point,
                "target_error_types": sorted(set([*component.target_error_types, *mapping.target_error_types])),
                "implementation_notes": [*component.implementation_notes, mapping.mapping_reason],
            }))
    return output


def _maturity_summary(contracts: list[ComponentContract]) -> ResearchMaturitySummary:
    counts = {
        name: 0
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
    for contract in contracts:
        if contract.maturity in counts:
            counts[contract.maturity] += 1
    return ResearchMaturitySummary.model_validate(counts)


def _compatibility_reviews(contracts: list[ComponentContract]) -> dict[str, Any]:
    checker = YOLO26CompatibilityChecker()
    result: dict[str, Any] = {}
    for contract in contracts:
        review = checker.check(
            components=[contract],
            train_overrides={"imgsz": 640},
            changed_variables=[contract.category],
            single_variable=True,
            execution_requested=False,
        )
        payload = review.model_dump(mode="json")
        if contract.maturity == "metadata_only":
            payload["metadata_only"] = sorted(set([*payload.get("metadata_only", []), contract.component_id]))
        result[contract.component_id] = payload
    return result


def _recipe_drafts(components: list[ExtractedComponent], contracts: list[ComponentContract]) -> list[RecipeSpec]:
    contract_ids = {item.component_id for item in contracts}
    recipes: dict[str, RecipeSpec] = {}
    for component in components:
        if component.component_id not in contract_ids:
            continue
        categories = component.component_category if component.component_category != "unknown" else "component"
        target_metrics = sorted({metric for claim in component.claimed_effects for metric in _claim_metrics(claim.claim)})
        target_metrics = target_metrics or ["map50_95"]
        facts = [{"fact_type": item} for item in component.target_error_types if item != "unknown"] or [{"component": component.component_id}]
        recipe = AtomicRecipe(
                recipe_id=f"paper.{_slug(component.component_id)}",
                version="0.1.0",
                target_error_facts=facts,
                target_metrics=target_metrics,
                component_ids=[component.component_id],
                fixed_variables={"imgsz": 640},
                train_overrides={"imgsz": 640},
                primary_changed_variable=categories,
                compatibility_requirements=["fixed_imgsz_640", "component_adapter", "pilot_evidence"],
                expected_effects={"source": "paper_claim", "target_metrics": target_metrics},
                evidence_prior=[claim.model_dump(mode="json") for claim in component.claimed_effects],
                implementation_risk="high",
                training_cost={"profile": "pilot"},
                inference_cost={"latency_guard": "required", "model_size_guard": "required"},
                stop_conditions=["pilot target error fact does not improve", "latency guard regresses", "model_size guard regresses"],
                promotion_requirements=["adapter_implemented", "smoke_passed", "pilot_reproduced", "three_seed_confirmation"],
                maturity="metadata_only",
            )
        recipes.setdefault(recipe.recipe_id, recipe)
    return [recipes[key] for key in sorted(recipes)]


def _curated_component_extraction(
    paper: PaperRecord,
    taxonomy: ComponentTaxonomy,
) -> ComponentExtractionResult:
    """Convert curated component ids into metadata-only paper claims without LLM inference."""
    if not paper.component_ids:
        return ComponentExtractionResult(
            status="skipped",
            paper_id=paper.paper_id,
            provider="offline",
            model="none",
            warnings=["offline_extraction_not_configured"],
        )
    location = paper.provenance.source_path if paper.provenance else "catalog:component_ids"
    components = []
    for component_id in paper.component_ids:
        category = _curated_component_category(component_id, paper, taxonomy)
        components.append(ExtractedComponent(
            component_id=component_id,
            name=component_id.replace("_", " ").replace(".", " ").strip().title(),
            component_category=category,
            insertion_point="unknown",
            claimed_effects=[ExtractedClaim(
                claim="Curated catalog identifies this component as a paper-level research prior.",
                paper_id=paper.paper_id,
                source_location=location,
                evidence_level="paper_claim",
            )],
            target_error_types=["unknown"],
            training_only="unknown",
            inference_only="unknown",
            implementation_notes=["Catalog metadata only; adapter implementation and execution are not verified."],
            evidence_level="paper_claim",
            uncertainties=["Exact tensor contract, compatibility, and local effect remain unknown."],
            source_locations=[SourceLocation(paper_id=paper.paper_id, location=location)],
        ))
    return ComponentExtractionResult(
        status="used",
        paper_id=paper.paper_id,
        provider="curated_catalog",
        model="none",
        bundle=ComponentExtractionBundle(extracted_components=components),
        warnings=["curated_component_ids_metadata_only"],
    )


def _curated_component_category(
    component_id: str,
    paper: PaperRecord,
    taxonomy: ComponentTaxonomy,
) -> str:
    normalized = component_id.casefold().replace("-", "_").replace(".", "_")
    keyword_categories = (
        (("sahi", "slice"), "slicing"),
        (("sample", "oversampling"), "sampling"),
        (("distill", "teacher_student"), "distillation"),
        (("augment", "mosaic", "mixup", "copy_paste"), "augmentation"),
        (("assign",), "assigner"),
        (("match", "hungarian"), "matching"),
        (("classification_loss", "focal", "quality_focal"), "classification_loss"),
        (("bbox", "iou", "dfl", "regression_loss"), "bbox_regression_loss"),
        (("attention",), "attention"),
        (("fpn", "pyramid", "multi_scale"), "feature_pyramid"),
        (("neck",), "neck"),
        (("head", "query"), "detection_head"),
        (("backbone",), "backbone"),
        (("conv",), "convolution_block"),
        (("nms",), "nms"),
        (("domain",), "domain_adaptation"),
        (("pretrain",), "pretraining"),
    )
    for keywords, category in keyword_categories:
        if any(keyword in normalized for keyword in keywords) and category in taxonomy.categories:
            return category
    if len(paper.component_categories) == 1:
        return paper.component_categories[0]
    return "unknown"


def _reproduction_queue(
    components: list[ExtractedComponent],
    contracts: list[ComponentContract],
    recipes: list[RecipeSpec],
) -> list[dict[str, Any]]:
    contract_by_id = {item.component_id: item for item in contracts}
    recipe_by_component = {component_id: recipe.recipe_id for recipe in recipes for component_id in recipe.component_ids}
    items: list[dict[str, Any]] = []
    for component in components:
        contract = contract_by_id.get(component.component_id)
        if contract is None:
            continue
        for paper_id in sorted({claim.paper_id for claim in component.claimed_effects}):
            status: ReproductionStatus = "registered"
            if not contract.can_execute:
                status = "adapter_required"
            items.append({
                "paper_id": paper_id,
                "component_id": component.component_id,
                "recipe_id": recipe_by_component.get(component.component_id),
                "status": status,
                "maturity": contract.maturity,
                "required_next": "adapter_implemented" if status == "adapter_required" else "unit_tested",
                "queued_for_training": False,
            })
    return items


def _claim_metrics(text: str) -> list[str]:
    lowered = text.lower()
    return [metric for metric in ("map50_95", "ap_small", "ap_medium", "ap_large", "latency_ms", "recall", "precision") if metric.replace("_", "") in lowered.replace("-", "").replace("_", "")]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "component"


def _paper_hash(paper: PaperRecord) -> str:
    payload = json.dumps(_semantic_paper_payload(paper), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _papers_version(papers: list[PaperRecord]) -> str:
    payload = [
        _semantic_paper_payload(paper)
        for paper in sorted(papers, key=lambda item: item.paper_id)
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _semantic_paper_payload(paper: PaperRecord) -> dict[str, Any]:
    """Return replay-relevant paper content without import-time bookkeeping."""
    payload = paper.model_dump(mode="json", exclude={"created_at"})
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("imported_at", None)
        provenance.pop("history", None)
    return payload


def _file_or_dir_hash(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        values = []
        for item in sorted(path.rglob("*")):
            if item.is_file():
                values.append((item.relative_to(path).as_posix(), hashlib.sha256(item.read_bytes()).hexdigest()))
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()
    return "missing"


def _combined_hash(*paths: Path) -> str:
    values = [
        (path.name, _file_or_dir_hash(path))
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def _note_root(catalog_path: str | None) -> Path | None:
    if not catalog_path:
        return None
    path = Path(catalog_path)
    return path.parent.parent if path.parent.name == "data" else path.parent


def _resolve_note_path(paper: PaperRecord, note_root: Path | None) -> Path | None:
    note_path = paper.provenance.original_note_path if paper.provenance else None
    if not note_path:
        return None
    candidate = Path(note_path)
    if candidate.is_absolute() or note_root is None:
        return candidate
    return note_root / candidate


def _parse_paper_evidence_non_blocking(
    parser: PaperNoteParser,
    *,
    paper: PaperRecord,
    taxonomy: ComponentTaxonomy,
    note_path: Path | None,
) -> PaperEvidenceSummary:
    """Turn any parser failure into an auditable artifact instead of a pipeline stop."""
    try:
        return parser.parse(paper=paper, taxonomy=taxonomy, note_path=note_path)
    except Exception as exc:
        return PaperEvidenceSummary(
            paper_id=paper.paper_id,
            status="failed",
            warnings=[f"parser_failed_non_blocking:{type(exc).__name__}:{exc}"],
        )


def _load_latest_frozen_method_coverage(
    root: Path,
) -> tuple[PaperMethodCoverageReport | None, str | None]:
    pointer = root / "latest_snapshot.yaml"
    if not pointer.is_file():
        return None, None
    try:
        raw = yaml.safe_load(pointer.read_text(encoding="utf-8-sig")) or {}
        snapshot_hash = str(raw.get("snapshot_hash") or "").strip() or None
        snapshot_path = Path(str(raw.get("snapshot_path") or ""))
        if not snapshot_path.is_absolute():
            snapshot_path = root / snapshot_path
        path = snapshot_path / "paper_method_coverage.yaml"
        return PaperMethodCoverageReport.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        ), snapshot_hash
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return None, None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "ResearchProductionPipeline",
    "ResearchProductionResult",
    "ResearchProductionState",
    "ResearchStageState",
    "StoredExtraction",
]
