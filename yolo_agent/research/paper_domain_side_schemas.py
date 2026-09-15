"""Paper-level distillation/domain-adaptation audit schemas.

Mirrors the loss-side record contract: a per-paper status that never collapses
into shared branch coverage.  Generic teacher-student and domain-alignment
branches are recorded as *primitives* on the route; only the paper-specific
route (its own mechanism id, config, and behavior evidence) can mark a paper
ready.  Required-asset records carry ``blocked_for_real_reproduction`` flags
that never block code-level implementation readiness.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin

DomainSidePaperStatus = Literal["ready", "blocked_missing_evidence", "out_of_scope"]

DomainSideRouteKind = Literal["distillation_route", "domain_adaptation_route"]


class DomainSideRouteSpec(BaseModel):
    """One explicit paper-to-runtime route on the teacher/domain side."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_id: str
    route_kind: DomainSideRouteKind
    paper_id: str
    branch_id: str | None = None
    method_identity_status: str = "branch_bound"
    component_id: str
    adapter_class: str
    runtime_strategy: str | None = None
    runtime_hooks: list[str] = Field(min_length=1)
    required_assets: list[str] = Field(default_factory=list)
    config_schema: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_route(self) -> "DomainSideRouteSpec":
        if not self.mechanism_id.strip():
            raise ValueError("domain-side route requires a mechanism_id")
        if not self.paper_id.strip():
            raise ValueError("domain-side route requires the owning paper_id")
        if not self.component_id.strip():
            raise ValueError("domain-side route requires a component_id")
        if not self.adapter_class.strip():
            raise ValueError("domain-side route requires an adapter_class")
        if any(not value.strip() for value in self.runtime_hooks):
            raise ValueError("domain-side route runtime hooks must be non-empty")
        if self.method_identity_status not in {"branch_bound", "identity_recovery"}:
            raise ValueError(
                f"unknown method identity status: {self.method_identity_status}"
            )
        return self


class DomainSideBehaviorEvidence(BaseModel):
    """Observed CPU behavior from synthetic probes, separate from mAP."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    observed_changes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RequiredAssetRecord(BaseModel):
    """One required asset and whether its absence blocks real reproduction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    asset_kind: Literal[
        "teacher_checkpoint",
        "student_checkpoint",
        "source_checkpoint",
        "unlabeled_data",
        "target_domain_data",
        "source_domain_data",
        "pseudo_label_manifest",
        "query_manifest",
        "pair_manifest",
        "protocol_evidence",
    ]
    available_at_runtime: bool
    blocked_for_real_reproduction: bool
    recovery_action: str

    @model_validator(mode="after")
    def validate_record(self) -> "RequiredAssetRecord":
        if not self.asset_id.strip():
            raise ValueError("required-asset record requires an asset_id")
        if not self.available_at_runtime and not self.blocked_for_real_reproduction:
            raise ValueError(
                f"unavailable asset must set blocked_for_real_reproduction: {self.asset_id}"
            )
        if not self.recovery_action.strip():
            raise ValueError("required-asset record requires a recovery_action")
        return self


class PaperDomainSideRecord(BaseModel):
    """Per-paper teacher/domain status that never collapses into shared coverage."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_domain_side_record.v1"
    paper_id: str
    title: str
    year: int = Field(ge=1900, le=2100)
    method_profile_id: str
    current_disposition: str
    primary_domain: str
    in_scope: bool
    paper_specific_mechanism_ids: list[str] = Field(default_factory=list)
    resolved_route_ids: list[str] = Field(default_factory=list)
    component_ids: list[str] = Field(default_factory=list)
    adapter_classes: list[str] = Field(default_factory=list)
    branch_ids: list[str] = Field(default_factory=list)
    method_identity_statuses: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    remaining_dependencies: list[str] = Field(default_factory=list)
    paper_specific_config: dict[str, Any] = Field(default_factory=dict)
    required_assets: list[RequiredAssetRecord] = Field(default_factory=list)
    behavior: DomainSideBehaviorEvidence
    status: DomainSidePaperStatus
    blockers: list[str] = Field(default_factory=list)
    implementation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_status(self) -> "PaperDomainSideRecord":
        if not self.paper_id.strip() or not self.method_profile_id.strip():
            raise ValueError("domain-side record requires paper and profile identity")
        if not self.title.strip() or not self.primary_domain.strip():
            raise ValueError("domain-side record requires paper metadata")
        if self.status == "ready" and self.blockers:
            raise ValueError("ready domain-side record must not carry blockers")
        if self.status == "blocked_missing_evidence" and not self.blockers:
            raise ValueError("blocked record must list at least one blocker")
        if self.in_scope and not self.resolved_route_ids and self.status == "ready":
            raise ValueError("in-scope ready record must resolve at least one route")
        return self


class PaperDomainSideAudit(BaseModel, YAMLModelMixin):
    """Complete distillation/domain-adaptation audit over the frozen campaign."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_domain_side_audit.v1"
    manifest_path: str
    plan_path: str
    plan_source_path: str
    manifest_membership_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    paper_count: int = Field(ge=0)
    domain_side_paper_count: int = Field(ge=0)
    distillation_paper_count: int = Field(ge=0)
    domain_adaptation_paper_count: int = Field(ge=0)
    semi_supervised_paper_count: int = Field(ge=0)
    records: list[PaperDomainSideRecord] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    audit_hash: str = ""

    @model_validator(mode="after")
    def validate_audit(self) -> "PaperDomainSideAudit":
        ids = [item.paper_id for item in self.records]
        if ids != sorted(set(ids)):
            raise ValueError("domain-side audit records must be unique and sorted")
        counts = {
            "ready": sum(item.status == "ready" for item in self.records),
            "blocked_missing_evidence": sum(
                item.status == "blocked_missing_evidence" for item in self.records
            ),
            "out_of_scope": sum(item.status == "out_of_scope" for item in self.records),
        }
        if self.summary and counts != self.summary:
            raise ValueError("domain-side audit summary does not match records")
        return self

    def with_hash(self) -> "PaperDomainSideAudit":
        import hashlib
        import json

        payload = self.model_dump(mode="json")
        payload.pop("audit_hash", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return self.model_copy(update={"audit_hash": digest})


__all__ = [
    "DomainSideBehaviorEvidence",
    "DomainSidePaperStatus",
    "DomainSideRouteKind",
    "DomainSideRouteSpec",
    "PaperDomainSideAudit",
    "PaperDomainSideRecord",
    "RequiredAssetRecord",
]
