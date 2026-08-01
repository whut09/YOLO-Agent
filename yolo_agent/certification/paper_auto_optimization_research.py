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
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.snapshot import ResearchSnapshot
from yolo_agent.resources import ResourcePaths
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.core.yaml_io import YAMLModelMixin


SAMPLING_COMPONENT_ID = "sampling.small_object"


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
        """Produce a fresh snapshot and select a certified sampling profile."""
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
        profiles = [
            item
            for item in coverage.profiles
            if SAMPLING_COMPONENT_ID in item.canonical_component_ids
            and item.component_adaptation
        ]
        profile_ids = {item.profile_id for item in profiles}
        decisions = [
            item
            for item in coverage.decisions
            if item.profile_id in profile_ids
            and item.decision == "reuse_existing_adapter"
            and SAMPLING_COMPONENT_ID in item.reusable_adapter_ids
        ]
        if not profiles or not decisions:
            raise RuntimeError(
                "fresh snapshot has no reusable sampling.small_object MethodProfile"
            )

        maturity = EffectiveComponentMaturityManifest.from_yaml(
            _snapshot_artifact(snapshot, snapshot_dir, "effective_component_maturity")
        ).by_component().get(SAMPLING_COMPONENT_ID)
        if maturity is None:
            raise RuntimeError("fresh snapshot has no sampling.small_object maturity identity")
        if not maturity.runtime_execution_ready:
            raise RuntimeError("sampling.small_object runtime is not execution ready")
        if maturity_rank(maturity.effective_maturity) < maturity_rank("gpu_certified"):
            raise RuntimeError(
                "sampling.small_object requires gpu_certified maturity; got "
                + maturity.effective_maturity
            )

        contract = _sampling_contract()
        current_hash = adapter_source_hash(contract)
        if current_hash != maturity.adapter_hash:
            raise RuntimeError(
                "sampling.small_object adapter hash changed after snapshot creation"
            )
        runtime_version = installed_ultralytics_version()
        if maturity.ultralytics_version != runtime_version:
            raise RuntimeError(
                "sampling.small_object Ultralytics version mismatch: "
                f"snapshot={maturity.ultralytics_version} runtime={runtime_version}"
            )

        context = PaperAcceptanceResearchContext(
            snapshot_hash=snapshot.snapshot_hash,
            snapshot_path=snapshot_dir,
            source_commit=str(snapshot.source_commit or "unknown"),
            paper_ids=sorted({item.paper_id for item in profiles}),
            method_profile_ids=sorted({item.profile_id for item in decisions}),
            implementation_decision_hashes=sorted(
                {item.decision_hash or item.with_hash().decision_hash for item in decisions}
            ),
            adapter_hash=current_hash,
            maturity=maturity.effective_maturity,
            maturity_protocol_hash=maturity.protocol_hash,
            ultralytics_version=runtime_version,
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


def _sampling_contract() -> ComponentContract:
    for path in sorted(ResourcePaths.COMPONENTS_DIR.rglob("*.yaml")):
        for contract in load_contracts(path):
            if contract.component_id == SAMPLING_COMPONENT_ID:
                return contract
    raise RuntimeError("sampling.small_object source contract is missing")


__all__ = [
    "PaperAcceptanceResearchContext",
    "PaperAcceptanceResearchPreparer",
    "SAMPLING_COMPONENT_ID",
]
