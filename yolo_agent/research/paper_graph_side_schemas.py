"""Paper model-graph audit schemas.

Mirrors the loss-side record contract: a per-paper status that never collapses
into shared component coverage, plus per-paper graph-diff payloads and
composition fingerprints that distinguish two papers sharing one plugin class.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin

GraphSidePaperStatus = Literal["ready", "blocked_missing_evidence", "out_of_scope"]


class PaperGraphSideRecord(BaseModel):
    """Per-paper graph-side status that never collapses into shared coverage."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_graph_side_record.v1"
    paper_id: str
    title: str
    year: int = Field(ge=1900, le=2100)
    method_profile_id: str
    current_disposition: str
    primary_domain: str
    secondary_domains: list[str] = Field(default_factory=list)
    in_scope: bool
    paper_specific_mechanism_ids: list[str] = Field(default_factory=list)
    resolved_route_ids: list[str] = Field(default_factory=list)
    component_ids: list[str] = Field(default_factory=list)
    adapter_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    remaining_dependencies: list[str] = Field(default_factory=list)
    paper_specific_config: dict[str, Any] = Field(default_factory=dict)
    behavior: "LossSideBehaviorEvidence"
    status: GraphSidePaperStatus
    blockers: list[str] = Field(default_factory=list)
    graph_diff: dict[str, Any] = Field(default_factory=dict)
    composition_fingerprint: str = ""
    implementation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_status(self) -> "PaperGraphSideRecord":
        if not self.paper_id.strip() or not self.method_profile_id.strip():
            raise ValueError("graph-side record requires paper and profile identity")
        if not self.title.strip() or not self.primary_domain.strip():
            raise ValueError("graph-side record requires paper metadata")
        if self.status == "ready" and self.blockers:
            raise ValueError("ready graph-side record must not carry blockers")
        if self.status == "blocked_missing_evidence" and not self.blockers:
            raise ValueError("blocked record must list at least one blocker")
        return self


class PaperGraphSideAudit(BaseModel, YAMLModelMixin):
    """Complete model-graph audit over the frozen campaign."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_graph_side_audit.v1"
    manifest_path: str
    plan_path: str
    plan_source_path: str
    manifest_membership_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    paper_count: int = Field(ge=0)
    graph_side_paper_count: int = Field(ge=0)
    records: list[PaperGraphSideRecord] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    audit_hash: str = ""

    @model_validator(mode="after")
    def validate_audit(self) -> "PaperGraphSideAudit":
        ids = [item.paper_id for item in self.records]
        if ids != sorted(set(ids)):
            raise ValueError("graph-side audit records must be sorted and unique")
        if self.paper_count != len(ids):
            raise ValueError("graph-side audit paper_count must equal record count")
        actual_count = sum(item.in_scope for item in self.records)
        if self.graph_side_paper_count != actual_count:
            raise ValueError("graph_side_paper_count does not match record scope")
        if self.audit_hash and self.audit_hash != self.calculate_hash():
            raise ValueError("graph-side audit hash mismatch")
        return self

    def calculate_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json", exclude={"audit_hash"}),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_hash(self) -> "PaperGraphSideAudit":
        return self.model_copy(update={"audit_hash": self.calculate_hash()})


from yolo_agent.research.paper_loss_side_schemas import LossSideBehaviorEvidence  # noqa: E402

PaperGraphSideRecord.model_rebuild()

__all__ = [
    "GraphSidePaperStatus",
    "PaperGraphSideAudit",
    "PaperGraphSideRecord",
]
