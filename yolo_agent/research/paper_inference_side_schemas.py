"""Paper-level inference-side audit schemas.

Mirrors the campaign's per-paper record contract.  The deployment boundary is
a first-class record: ``train_time`` and ``inference_time`` are Literal-typed
so an inference-only paper mechanism can never be serialized as a training
recipe, and every record carries latency, export, and NMS-free compatibility
notes required before a policy is proposed for YOLO26 deployment.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin

InferenceSidePaperStatus = Literal[
    "ready",
    "blocked_compatibility",
    "blocked_missing_evidence",
    "out_of_scope",
]

InferencePolicyKindName = Literal[
    "sahi_slicing",
    "tiled_multi_scale",
    "test_time_augmentation",
    "confidence_calibration",
    "class_aware_thresholding",
    "merge_policy",
]

NmsFreeCompatibility = Literal[
    "native_one_to_one",
    "requires_allow_cross_view_merge",
    "not_applicable",
]


class InferenceDeploymentBoundary(BaseModel, YAMLModelMixin):
    """Deployment boundary for one inference-only mechanism.

    ``train_time`` and ``inference_time`` are fixed Literals: the type system,
    not documentation, keeps inference-only policies out of training recipes.
    """

    model_config = ConfigDict(extra="forbid")

    train_time: Literal[False] = False
    inference_time: Literal[True] = True
    policy_kind: InferencePolicyKindName
    metric_namespace: str = Field(min_length=1)
    adapter_path: str = Field(min_length=1)
    latency_risk: Literal["low", "medium", "high"]
    export_compatibility_risk: str = Field(min_length=1)
    nms_free_compatibility: NmsFreeCompatibility
    one_to_one_guard_required: bool
    standard_640_namespace_preserved: bool = True
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_boundary(self) -> "InferenceDeploymentBoundary":
        if self.nms_free_compatibility == "requires_allow_cross_view_merge" and not (
            self.one_to_one_guard_required
        ):
            raise ValueError(
                "cross-view merge on a one-to-one head requires the explicit "
                "allow_cross_view_merge guard"
            )
        if not self.standard_640_namespace_preserved:
            raise ValueError(
                "standard imgsz=640 namespace must remain untouched by any "
                "inference policy"
            )
        return self


class InferenceSideBehaviorEvidence(BaseModel):
    """Observed behavior on synthetic detections; no model is executed."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    observed_changes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PaperInferenceSideRecord(BaseModel, YAMLModelMixin):
    """Per-paper inference-side status that never collapses into the registry."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_inference_side_record.v1"
    paper_id: str
    title: str
    year: int = Field(ge=1900, le=2100)
    primary_domain: str
    in_scope: bool
    scope_reason: str = Field(min_length=1)
    mechanism_id: str = ""
    policy_kind: InferencePolicyKindName | None = None
    parameter_semantics: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    deployment_boundary: InferenceDeploymentBoundary | None = None
    behavior: InferenceSideBehaviorEvidence
    remaining_dependencies: list[str] = Field(default_factory=list)
    status: InferenceSidePaperStatus
    blockers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_record(self) -> "PaperInferenceSideRecord":
        if not self.paper_id.strip() or not self.scope_reason.strip():
            raise ValueError("inference-side record requires identity and scope reason")
        if self.status == "ready":
            if self.blockers:
                raise ValueError("ready record must not carry blockers")
            if not self.in_scope or self.deployment_boundary is None:
                raise ValueError("ready record must be in scope with a deployment boundary")
            if not self.behavior.passed:
                raise ValueError("ready record requires real behavior evidence")
            if not self.parameter_semantics.strip():
                raise ValueError("ready record requires per-paper parameter semantics")
        if self.status == "blocked_compatibility" and not self.blockers:
            raise ValueError("blocked_compatibility record must list its blockers")
        if self.status == "blocked_missing_evidence" and not self.blockers:
            raise ValueError("blocked record must list at least one blocker")
        return self


class PaperInferenceSideAudit(BaseModel, YAMLModelMixin):
    """Complete inference-side audit over the frozen campaign."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_inference_side_audit.v1"
    manifest_membership_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    paper_count: int = Field(ge=0)
    inference_side_paper_count: int = Field(ge=0)
    records: list[PaperInferenceSideRecord] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    audit_hash: str = ""

    @model_validator(mode="after")
    def validate_audit(self) -> "PaperInferenceSideAudit":
        ids = [item.paper_id for item in self.records]
        if ids != sorted(set(ids)):
            raise ValueError("inference-side audit records must be unique and sorted")
        counts = {
            "ready": sum(item.status == "ready" for item in self.records),
            "blocked_compatibility": sum(
                item.status == "blocked_compatibility" for item in self.records
            ),
            "blocked_missing_evidence": sum(
                item.status == "blocked_missing_evidence" for item in self.records
            ),
            "out_of_scope": sum(item.status == "out_of_scope" for item in self.records),
        }
        if self.summary and counts != self.summary:
            raise ValueError("inference-side audit summary does not match records")
        return self

    def with_hash(self) -> "PaperInferenceSideAudit":
        import hashlib
        import json

        payload = self.model_dump(mode="json")
        payload.pop("audit_hash", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return self.model_copy(update={"audit_hash": digest})


__all__ = [
    "InferenceDeploymentBoundary",
    "InferencePolicyKindName",
    "InferenceSideBehaviorEvidence",
    "InferenceSidePaperStatus",
    "NmsFreeCompatibility",
    "PaperInferenceSideAudit",
    "PaperInferenceSideRecord",
]
