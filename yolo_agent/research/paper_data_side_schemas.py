"""Schemas for the frozen campaign's paper-level data-side audit."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


DataSideRouteKind = Literal[
    "sampling",
    "augmentation",
    "annotation",
    "preprocessing",
    "active_learning",
]
DataSidePaperStatus = Literal["ready", "blocked_missing_evidence", "out_of_scope"]


class DataSideRouteSpec(BaseModel):
    """One explicit mechanism-to-runtime route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_id: str
    route_kind: DataSideRouteKind
    component_id: str | None = None
    adapter_id: str | None = None
    implementation_path: str
    adapter_class: str | None = None
    changed_variables: list[str] = Field(min_length=1)
    runtime_hooks: list[str] = Field(min_length=1)
    required_evidence: list[str] = Field(default_factory=list)
    required_config_keys: list[str] = Field(default_factory=list)
    test_refs: list[str] = Field(default_factory=list)
    inference_only: bool = False
    config_schema: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_route(self) -> "DataSideRouteSpec":
        if not self.mechanism_id.strip():
            raise ValueError("data-side route requires a mechanism_id")
        if not self.implementation_path.strip():
            raise ValueError("data-side route requires an implementation_path")
        if any(not value.strip() for value in self.changed_variables):
            raise ValueError("data-side route changed variables must be non-empty")
        if any(not value.strip() for value in self.runtime_hooks):
            raise ValueError("data-side route runtime hooks must be non-empty")
        if any(not value.strip() for value in self.required_config_keys):
            raise ValueError("data-side route required config keys must be non-empty")
        missing_config = set(self.required_config_keys) - set(self.config_schema)
        if missing_config:
            raise ValueError(
                "data-side route required config keys are absent from schema: "
                + ", ".join(sorted(missing_config))
            )
        if self.adapter_id and not self.component_id:
            raise ValueError("data-side adapter_id requires component_id")
        declared_imgsz = self.config_schema.get("imgsz")
        if declared_imgsz is not None and declared_imgsz != 640:
            raise ValueError("data-side routes require fixed imgsz=640")
        return self


class DataSideBehaviorEvidence(BaseModel):
    """Observed CPU behavior, separate from any mAP or training result."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    observed_changes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PaperDataSideRecord(BaseModel):
    """Per-paper data-side status that never collapses into shared coverage."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_data_side_record.v1"
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
    behavior: DataSideBehaviorEvidence
    status: DataSidePaperStatus
    blockers: list[str] = Field(default_factory=list)
    implementation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_status(self) -> "PaperDataSideRecord":
        if not self.paper_id.strip() or not self.method_profile_id.strip():
            raise ValueError("data-side record requires paper and profile identity")
        if not self.title.strip() or not self.primary_domain.strip():
            raise ValueError("data-side record requires paper metadata")
        if self.in_scope and self.status == "out_of_scope":
            raise ValueError("in-scope data-side paper cannot be out_of_scope")
        if not self.in_scope and self.status != "out_of_scope":
            raise ValueError("out-of-scope paper cannot have a data-side status")
        if self.status == "ready" and (self.blockers or not self.behavior.passed):
            raise ValueError("ready data-side record cannot retain blockers or failed behavior")
        if self.status == "blocked_missing_evidence" and not self.blockers:
            raise ValueError("blocked data-side record requires exact blockers")
        return self


class PaperDataSideAudit(BaseModel, YAMLModelMixin):
    """Complete audit over the frozen campaign, including out-of-scope papers."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_data_side_audit.v1"
    manifest_path: str
    plan_path: str
    plan_source_path: str
    manifest_membership_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    paper_count: int = Field(ge=0)
    data_side_paper_count: int = Field(ge=0)
    records: list[PaperDataSideRecord] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    audit_hash: str = ""

    @model_validator(mode="after")
    def validate_audit(self) -> "PaperDataSideAudit":
        ids = [item.paper_id for item in self.records]
        if ids != sorted(set(ids)):
            raise ValueError("data-side audit records must be sorted and unique")
        if self.paper_count != len(ids):
            raise ValueError("data-side audit paper_count must equal record count")
        actual_count = sum(item.in_scope for item in self.records)
        if self.data_side_paper_count != actual_count:
            raise ValueError("data_side_paper_count does not match record scope")
        if self.audit_hash and self.audit_hash != self.calculate_hash():
            raise ValueError("data-side audit hash mismatch")
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"audit_hash", "summary"},
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def with_hash(self) -> "PaperDataSideAudit":
        return self.model_copy(update={"audit_hash": self.calculate_hash()})


__all__ = [
    "DataSideBehaviorEvidence",
    "DataSidePaperStatus",
    "DataSideRouteKind",
    "DataSideRouteSpec",
    "PaperDataSideAudit",
    "PaperDataSideRecord",
]
