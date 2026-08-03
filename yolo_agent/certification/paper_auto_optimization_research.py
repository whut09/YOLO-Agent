"""Offline research preflight for paper-driven optimization acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.maturity import maturity_rank
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
    installed_ultralytics_version,
)
from yolo_agent.research.awesome_snapshot_builder import AwesomeSnapshotBuilder
from yolo_agent.research.maturity_snapshot import EffectiveComponentMaturityManifest
from yolo_agent.research.maturity_snapshot import FrozenComponentMaturity
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.snapshot import ResearchSnapshot
from yolo_agent.resources import ResourcePaths
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.certification.paper_auto_optimization_tracks import (
    PAPER_ACCEPTANCE_RECIPES,
    PaperAcceptanceRecipe,
    PaperAcceptanceTrackId,
)


SAMPLING_COMPONENT_ID = "sampling.small_object"


class PaperAcceptanceTrackContext(BaseModel):
    """Snapshot-bound method and runtime identity for one mechanism family."""

    model_config = ConfigDict(extra="forbid")

    track_id: PaperAcceptanceTrackId
    component_id: str
    component_family: str
    paper_ids: list[str] = Field(min_length=1)
    method_profile_ids: list[str] = Field(min_length=1)
    implementation_decision_hashes: list[str] = Field(min_length=1)
    adapter_hash: str
    maturity: str
    maturity_protocol_hash: str
    ultralytics_version: str


class PaperAcceptanceResearchContext(BaseModel, YAMLModelMixin):
    """Frozen paper, method, and certified adapter identity for one acceptance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_acceptance_research_context.v1"
    snapshot_hash: str
    snapshot_path: Path
    source_commit: str
    paper_ids: list[str] = Field(min_length=1)
    method_profile_ids: list[str] = Field(min_length=1)
    implementation_decision_hashes: list[str] = Field(min_length=1)
    component_id: str = SAMPLING_COMPONENT_ID
    adapter_hash: str
    maturity: str
    maturity_protocol_hash: str
    ultralytics_version: str
    tracks: list[PaperAcceptanceTrackContext] = Field(default_factory=list)
    context_hash: str = ""

    @model_validator(mode="after")
    def validate_context_hash(self) -> "PaperAcceptanceResearchContext":
        expected = self.calculate_hash()
        if self.context_hash and self.context_hash != expected:
            raise ValueError("paper acceptance research context hash mismatch")
        self.context_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"context_hash"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def effective_tracks(self) -> list[PaperAcceptanceTrackContext]:
        """Return v2 tracks, or a sampling-only projection for old artifacts."""
        if self.tracks:
            return list(self.tracks)
        return [
            PaperAcceptanceTrackContext(
                track_id="sampling",
                component_id=self.component_id,
                component_family="sampling",
                paper_ids=self.paper_ids,
                method_profile_ids=self.method_profile_ids,
                implementation_decision_hashes=self.implementation_decision_hashes,
                adapter_hash=self.adapter_hash,
                maturity=self.maturity,
                maturity_protocol_hash=self.maturity_protocol_hash,
                ultralytics_version=self.ultralytics_version,
            )
        ]


class PaperAcceptanceResearchPreparer:
    """Build and validate a new local ResearchSnapshot without network access."""

    def __init__(
        self,
        *,
        research_root: Path | str,
        source: Path | str | None,
        maturity_registry: Path | str,
        source_commit: str | None = None,
    ) -> None:
        self.research_root = Path(research_root).resolve()
        self.source = Path(source).resolve() if source is not None else None
        self.registry = ComponentMaturityRegistry(maturity_registry)
        self.source_commit = source_commit

    def prepare(self, output_path: Path | str) -> PaperAcceptanceResearchContext:
        """Produce a fresh snapshot and select four certified mechanism profiles."""
        builder = AwesomeSnapshotBuilder(
            self.research_root,
            maturity_registry=self.registry,
        )
        result = builder.build(
            source=self.source,
            source_commit=self.source_commit,
            force=True,
        )
        if result.status != "completed" or not result.snapshot_path:
            detail = "; ".join(result.errors) or result.unavailable_reason or result.status
            raise RuntimeError(f"fresh ResearchSnapshot failed: {detail}")

        snapshot_dir = Path(result.snapshot_path).resolve()
        snapshot = ResearchSnapshot.from_snapshot_dir(snapshot_dir)
        failures = snapshot.verify(snapshot_dir)
        if failures:
            raise RuntimeError("fresh ResearchSnapshot integrity failed: " + "; ".join(failures))
        if snapshot.snapshot_status != "current":
            raise RuntimeError(
                "fresh ResearchSnapshot is stale: " + "; ".join(snapshot.stale_reasons)
            )
        if snapshot.paper_intelligence != "available":
            raise RuntimeError(
                "paper intelligence unavailable: "
                + str(snapshot.unavailable_reason or "unknown")
            )

        coverage = PaperMethodCoverageReport.from_yaml(
            _snapshot_artifact(snapshot, snapshot_dir, "paper_method_coverage")
        )
        maturity_by_component = EffectiveComponentMaturityManifest.from_yaml(
            _snapshot_artifact(snapshot, snapshot_dir, "effective_component_maturity")
        ).by_component()
        runtime_version = installed_ultralytics_version()
        tracks = [
            _track_context(
                recipe=recipe,
                coverage=coverage,
                maturity_by_component=maturity_by_component,
                runtime_version=runtime_version,
            )
            for recipe in PAPER_ACCEPTANCE_RECIPES
        ]
        primary = next(
            item for item in tracks if item.component_id == SAMPLING_COMPONENT_ID
        )

        context = PaperAcceptanceResearchContext(
            snapshot_hash=snapshot.snapshot_hash,
            snapshot_path=snapshot_dir,
            source_commit=str(snapshot.source_commit or "unknown"),
            paper_ids=primary.paper_ids,
            method_profile_ids=primary.method_profile_ids,
            implementation_decision_hashes=primary.implementation_decision_hashes,
            adapter_hash=primary.adapter_hash,
            maturity=primary.maturity,
            maturity_protocol_hash=primary.maturity_protocol_hash,
            ultralytics_version=runtime_version,
            tracks=tracks,
        )
        context.to_yaml(output_path, exclude_none=True, sort_keys=False)
        return context


def _snapshot_artifact(
    snapshot: ResearchSnapshot,
    snapshot_dir: Path,
    name: str,
) -> Path:
    artifact = snapshot.artifacts.get(name)
    if artifact is None:
        raise RuntimeError(f"fresh snapshot artifact missing: {name}")
    path = snapshot_dir / artifact.path
    if not path.is_file():
        raise RuntimeError(f"fresh snapshot artifact path missing: {path}")
    return path


def load_sampling_contract() -> ComponentContract:
    return load_component_contract(SAMPLING_COMPONENT_ID)


def load_component_contract(component_id: str) -> ComponentContract:
    for path in sorted(ResourcePaths.COMPONENTS_DIR.rglob("*.yaml")):
        try:
            contracts = load_contracts(path)
        except (KeyError, TypeError, ValueError):
            continue
        for contract in contracts:
            if contract.component_id == component_id:
                return contract
    raise RuntimeError(f"{component_id} source contract is missing")


def _track_context(
    *,
    recipe: PaperAcceptanceRecipe,
    coverage: PaperMethodCoverageReport,
    maturity_by_component: dict[str, FrozenComponentMaturity],
    runtime_version: str,
) -> PaperAcceptanceTrackContext:
    component_id = recipe.component_id
    profiles = [
        item
        for item in coverage.profiles
        if component_id in item.canonical_component_ids and item.component_adaptation
    ]
    profile_ids = {item.profile_id for item in profiles}
    decisions = [
        item
        for item in coverage.decisions
        if item.profile_id in profile_ids
        and item.decision == "reuse_existing_adapter"
        and component_id in item.reusable_adapter_ids
    ]
    if not profiles or not decisions:
        raise RuntimeError(
            f"fresh snapshot has no reusable {component_id} MethodProfile"
        )
    maturity = maturity_by_component.get(component_id)
    if maturity is None:
        raise RuntimeError(f"fresh snapshot has no {component_id} maturity identity")
    if not maturity.runtime_execution_ready:
        raise RuntimeError(f"{component_id} runtime is not execution ready")
    effective_maturity = maturity.effective_maturity
    if maturity_rank(effective_maturity) < maturity_rank("gpu_certified"):
        raise RuntimeError(
            f"{component_id} requires gpu_certified maturity; got {effective_maturity}"
        )
    contract = load_component_contract(component_id)
    current_hash = adapter_source_hash(contract)
    if current_hash != maturity.adapter_hash:
        raise RuntimeError(f"{component_id} adapter hash changed after snapshot creation")
    maturity_version = maturity.ultralytics_version
    if maturity_version != runtime_version:
        raise RuntimeError(
            f"{component_id} Ultralytics version mismatch: "
            f"snapshot={maturity_version} runtime={runtime_version}"
        )
    return PaperAcceptanceTrackContext(
        track_id=recipe.track_id,
        component_id=component_id,
        component_family=recipe.component_family,
        paper_ids=sorted({item.paper_id for item in profiles}),
        method_profile_ids=sorted({item.profile_id for item in decisions}),
        implementation_decision_hashes=sorted(
            {item.decision_hash or item.with_hash().decision_hash for item in decisions}
        ),
        adapter_hash=current_hash,
        maturity=effective_maturity,
        maturity_protocol_hash=maturity.protocol_hash,
        ultralytics_version=runtime_version,
    )


__all__ = [
    "PaperAcceptanceResearchContext",
    "PaperAcceptanceResearchPreparer",
    "PaperAcceptanceTrackContext",
    "SAMPLING_COMPONENT_ID",
    "load_sampling_contract",
    "load_component_contract",
]
