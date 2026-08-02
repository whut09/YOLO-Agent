"""Typed contracts for clustering paper methods into reusable mechanisms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.resources import ResourcePaths


ClusterMatchType = Literal["exact_match", "semantic_match", "unresolved"]
ClusterConfidence = Literal["low", "medium", "high"]
ClusterCompatibility = Literal[
    "compatible",
    "adapter_required",
    "incompatible",
    "separate_detector_family",
]


class MechanismClusterDefinition(BaseModel):
    """One reusable runtime mechanism with a single training semantic."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    display_name: str
    training_semantic: str
    adapter_family: str
    canonical_component_ids: list[str] = Field(default_factory=list)
    method_families: list[str] = Field(default_factory=list)
    semantic_aliases: list[str] = Field(default_factory=list)
    insertion_points: list[str] = Field(default_factory=list)
    required_runtime_hooks: list[str] = Field(default_factory=list)
    parameter_keys: list[str] = Field(default_factory=list)
    training_only: bool | None = None
    inference_changed: bool | None = None
    yolo26_compatibility: ClusterCompatibility = "adapter_required"
    constraints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> "MechanismClusterDefinition":
        if not self.cluster_id.strip() or not self.training_semantic.strip():
            raise ValueError("mechanism cluster requires identity and training semantic")
        if not self.adapter_family.strip():
            raise ValueError("mechanism cluster requires adapter_family")
        if not (
            self.canonical_component_ids
            or self.method_families
            or self.semantic_aliases
        ):
            raise ValueError("mechanism cluster requires at least one matching term")
        return self


class MechanismClusterConfig(BaseModel):
    """Validated cluster taxonomy loaded from the bundled offline config."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_mechanism_clusters.v1"
    clusters: list[MechanismClusterDefinition]

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = ResourcePaths.PAPER_MECHANISM_CLUSTERS,
    ) -> "MechanismClusterConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_clusters(self) -> "MechanismClusterConfig":
        cluster_ids = [item.cluster_id for item in self.clusters]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("duplicate mechanism cluster_id")
        semantic_keys = [
            (item.adapter_family, item.training_semantic) for item in self.clusters
        ]
        if len(semantic_keys) != len(set(semantic_keys)):
            raise ValueError("duplicate adapter-family training semantic")
        return self


class ClusterEvidence(BaseModel):
    """One source-grounded reason for assigning a paper to a cluster."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    value: str | bool
    source: str
    source_location: str
    confidence: ClusterConfidence


class PaperMechanismClusterMatch(BaseModel):
    """Auditable paper -> MethodProfile -> cluster -> adapter-family link."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    profile_id: str
    cluster_id: str | None = None
    adapter_family: str | None = None
    training_semantic: str | None = None
    match_type: ClusterMatchType
    confidence: ClusterConfidence
    confidence_score: float = Field(ge=0.0, le=1.0)
    evidence: list[ClusterEvidence] = Field(default_factory=list)
    match_reason: str
    adapter_available: bool = False
    runtime_ready: bool = False
    conflicts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_match(self) -> "PaperMechanismClusterMatch":
        if self.match_type == "semantic_match" and not self.evidence:
            raise ValueError("semantic_match requires source evidence")
        if self.match_type == "unresolved" and self.cluster_id is not None:
            raise ValueError("unresolved match cannot identify a cluster")
        if self.match_type != "unresolved" and not self.cluster_id:
            raise ValueError("resolved match requires cluster_id")
        return self


class MechanismClusterSummary(BaseModel):
    """Aggregated parameter and provenance surface for one cluster."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    adapter_family: str
    training_semantic: str
    paper_ids: list[str] = Field(default_factory=list)
    paper_count: int = Field(default=0, ge=0)
    parameter_differences: dict[str, list[str]] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    source_locations: list[str] = Field(default_factory=list)
    adapter_available: bool = False
    runtime_ready: bool = False


class MechanismClusterConflict(BaseModel):
    """Rejected merge where names overlap but runtime semantics do not."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    candidate_cluster_ids: list[str]
    reason: str
    evidence_locations: list[str] = Field(default_factory=list)


class AdapterCoverageOpportunity(BaseModel):
    """One adapter-family implementation opportunity ranked by paper coverage."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    cluster_id: str
    adapter_family: str
    paper_ids: list[str]
    paper_count: int = Field(ge=1)
    runtime_hooks: list[str] = Field(default_factory=list)
    implementation_status: Literal[
        "adapter_available",
        "runtime_ready",
        "adapter_required",
        "separate_detector_family",
        "incompatible",
    ]
    score: float
    reasons: list[str] = Field(default_factory=list)


class PaperMechanismClusterReport(BaseModel, YAMLModelMixin):
    """Complete reusable-mechanism clustering and implementation opportunity report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_mechanism_cluster_report.v1"
    paper_count: int = Field(default=0, ge=0)
    matched_paper_count: int = Field(default=0, ge=0)
    unresolved_paper_count: int = Field(default=0, ge=0)
    matches: list[PaperMechanismClusterMatch] = Field(default_factory=list)
    clusters: list[MechanismClusterSummary] = Field(default_factory=list)
    conflicts: list[MechanismClusterConflict] = Field(default_factory=list)
    implementation_opportunities: list[AdapterCoverageOpportunity] = Field(
        default_factory=list
    )
    report_hash: str = ""

    def with_hash(self) -> "PaperMechanismClusterReport":
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self.model_copy(update={"report_hash": digest})


__all__ = [
    "AdapterCoverageOpportunity",
    "ClusterCompatibility",
    "ClusterConfidence",
    "ClusterEvidence",
    "ClusterMatchType",
    "MechanismClusterConfig",
    "MechanismClusterConflict",
    "MechanismClusterDefinition",
    "MechanismClusterSummary",
    "PaperMechanismClusterMatch",
    "PaperMechanismClusterReport",
]
