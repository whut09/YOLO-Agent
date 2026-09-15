"""Paper-level loss/assignment audit schemas.

Mirrors the data-side record contract: a per-paper status that never collapses
into shared component coverage, plus per-route evidence produced by real CPU
behavior probes (math matrix, assignment comparison) rather than metadata.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin

LossSidePaperStatus = Literal["ready", "blocked_missing_evidence", "out_of_scope"]

LossSideRouteKind = Literal["auxiliary_loss", "assignment"]


class LossSideRouteSpec(BaseModel):
    """One explicit mechanism-to-runtime route on the loss/assignment side."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_id: str
    route_kind: LossSideRouteKind
    component_id: str | None = None
    adapter_id: str | None = None
    implementation_path: str
    plugin_name: str
    runtime_hooks: list[str] = Field(min_length=1)
    required_evidence: list[str] = Field(default_factory=list)
    test_refs: list[str] = Field(default_factory=list)
    config_schema: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_route(self) -> "LossSideRouteSpec":
        if not self.mechanism_id.strip():
            raise ValueError("loss-side route requires a mechanism_id")
        if not self.implementation_path.strip():
            raise ValueError("loss-side route requires an implementation_path")
        if not self.plugin_name.strip():
            raise ValueError("loss-side route requires a plugin_name")
        if any(not value.strip() for value in self.runtime_hooks):
            raise ValueError("loss-side route runtime hooks must be non-empty")
        return self


class LossSideBehaviorEvidence(BaseModel):
    """Observed CPU behavior from math probes, separate from mAP or training."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    observed_changes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PaperLossSideRecord(BaseModel):
    """Per-paper loss/assignment status that never collapses into shared coverage."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_loss_side_record.v1"
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
    behavior: LossSideBehaviorEvidence
    status: LossSidePaperStatus
    blockers: list[str] = Field(default_factory=list)
    implementation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_status(self) -> "PaperLossSideRecord":
        if not self.paper_id.strip() or not self.method_profile_id.strip():
            raise ValueError("loss-side record requires paper and profile identity")
        if not self.title.strip() or not self.primary_domain.strip():
            raise ValueError("loss-side record requires paper metadata")
        if self.status == "ready" and self.blockers:
            raise ValueError("ready loss-side record must not carry blockers")
        if self.status == "blocked_missing_evidence" and not self.blockers:
            raise ValueError("blocked record must list at least one blocker")
        return self


class PaperLossSideAudit(BaseModel, YAMLModelMixin):
    """Complete loss/assignment audit over the frozen campaign."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_loss_side_audit.v1"
    manifest_path: str
    plan_path: str
    plan_source_path: str
    manifest_membership_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    paper_count: int = Field(ge=0)
    loss_side_paper_count: int = Field(ge=0)
    records: list[PaperLossSideRecord] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    audit_hash: str = ""

    @model_validator(mode="after")
    def validate_audit(self) -> "PaperLossSideAudit":
        ids = [item.paper_id for item in self.records]
        if ids != sorted(set(ids)):
            raise ValueError("loss-side audit records must be sorted and unique")
        if self.paper_count != len(ids):
            raise ValueError("loss-side audit paper_count must equal record count")
        actual_count = sum(item.in_scope for item in self.records)
        if self.loss_side_paper_count != actual_count:
            raise ValueError("loss_side_paper_count does not match record scope")
        if self.audit_hash and self.audit_hash != self.calculate_hash():
            raise ValueError("loss-side audit hash mismatch")
        return self

    def calculate_hash(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            self.model_dump(mode="json", exclude={"audit_hash"}),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_hash(self) -> "PaperLossSideAudit":
        return self.model_copy(update={"audit_hash": self.calculate_hash()})
