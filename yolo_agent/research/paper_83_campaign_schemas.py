"""Schemas for the historical README paper-83 implementation campaign.

The campaign membership is deliberately immutable and separate from the
current adapter audit.  A later maturity change may alter a paper's current
status, but it must never silently alter the frozen paper-id set.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.paper_execution_schemas import PaperExecutionDisposition


PAPER_83_COUNT = 83
PAPER_85_COUNT = 85
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ReadmeCoverageDeclaration(BaseModel):
    """Coverage values and hashes parsed from the current README."""

    model_config = ConfigDict(extra="forbid")

    method_profile_numerator: int = Field(ge=0)
    method_profile_denominator: int = Field(ge=0)
    certified_adapter_numerator: int = Field(ge=0)
    certified_adapter_denominator: int = Field(ge=0)
    audit_snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    acceptance_hash: str = Field(pattern=SHA256_PATTERN)


class Paper83MembershipSource(BaseModel):
    """The exact acceptance metric that supplies campaign membership."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["coverage_acceptance_metric_numerator"]
    metric_id: Literal["compatible_papers_certified_adapter"]
    acceptance_hash: str = Field(pattern=SHA256_PATTERN)
    source_method_coverage_hash: str = Field(pattern=SHA256_PATTERN)
    source_executable_coverage_hash: str = Field(pattern=SHA256_PATTERN)
    source_registry_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class Paper83RepositoryLineage(BaseModel):
    """Repository and snapshot lineage recorded beside the frozen set."""

    model_config = ConfigDict(extra="forbid")

    git_commit: str = Field(min_length=1)
    readme_audit_snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    current_research_snapshot_hash: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    acceptance_lineage_status: Literal["exact", "historical"]


class Paper83Campaign(BaseModel):
    """Immutable campaign identity and its acceptance provenance."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["paper-83"]
    frozen: Literal[True] = True
    expected_paper_count: Literal[83] = PAPER_83_COUNT
    membership_source: Paper83MembershipSource
    repository: Paper83RepositoryLineage
    membership_hash: str = Field(pattern=SHA256_PATTERN)


class Paper83Paper(BaseModel):
    """One paper in the frozen historical campaign."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str
    year: int = Field(ge=1900, le=2100)
    source: str
    method_profile_id: str
    frozen_certified_adapter_ids: list[str] = Field(min_length=1)
    current_component_ids: list[str] = Field(default_factory=list)
    current_adapter_ids: list[str] = Field(default_factory=list)
    current_disposition: PaperExecutionDisposition


class Paper83Manifest(BaseModel, YAMLModelMixin):
    """Machine-readable membership and current status audit."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["paper_implementation_campaign.v1"] = (
        "paper_implementation_campaign.v1"
    )
    campaign: Paper83Campaign
    paper_count: Literal[83] = PAPER_83_COUNT
    papers: list[Paper83Paper] = Field(min_length=PAPER_83_COUNT, max_length=PAPER_83_COUNT)

    @model_validator(mode="after")
    def validate_manifest(self) -> "Paper83Manifest":
        paper_ids = [item.paper_id for item in self.papers]
        if paper_ids != sorted(set(paper_ids)):
            raise ValueError("paper-83 manifest paper IDs must be sorted and unique")
        if self.paper_count != len(paper_ids):
            raise ValueError("paper-83 manifest paper_count does not match papers")
        if self.campaign.expected_paper_count != len(paper_ids):
            raise ValueError("paper-83 campaign count does not match papers")
        if self.campaign.membership_hash != calculate_membership_hash(paper_ids):
            raise ValueError("paper-83 membership_hash does not match paper IDs")
        return self


def calculate_membership_hash(paper_ids: list[str] | tuple[str, ...]) -> str:
    """Hash only the sorted paper-id membership, never mutable paper status."""

    normalized = list(paper_ids)
    if len(normalized) != len(set(normalized)):
        raise ValueError("membership hash input contains duplicate paper IDs")
    payload = {"paper_ids": sorted(normalized)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PAPER_83_COUNT",
    "PAPER_85_COUNT",
    "Paper83Campaign",
    "Paper83Manifest",
    "Paper83MembershipSource",
    "Paper83Paper",
    "Paper83RepositoryLineage",
    "ReadmeCoverageDeclaration",
    "calculate_membership_hash",
]
