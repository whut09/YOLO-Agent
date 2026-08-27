"""CPU readiness certification for paper-specific distillation routes.

The certification is fail-closed and GPU-free: it resolves the real teacher
and student checkpoints, binds their hashes, split, and dataset protocol to
the paper route, and reports a recoverable disposition.  A paper without a
real checkpoint never reports runtime readiness, and mismatched teacher
bindings block the runtime instead of silently recovering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.distillation.paper_routes import (
    DistillationPaperRoute,
    default_paper_route_registry,
)
from yolo_agent.components.adapters.distillation.teacher_evidence import (
    CheckpointResolution,
    resolve_student_checkpoint,
    resolve_teacher_checkpoint,
)
from yolo_agent.core.yaml_io import YAMLModelMixin


PaperRouteDisposition = Literal["runtime_ready", "evidence_recovery", "blocked_runtime"]

_MISMATCH_MARKERS = (
    "sha256_mismatch",
    "split_mismatch",
    "imgsz_mismatch",
    "architecture_mismatch",
    "dataset_hash_mismatch",
)
class DistillationPaperRouteReport(BaseModel, YAMLModelMixin):
    """CPU certification result for one paper's independent route."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "distillation_paper_route_report.v1"
    paper_id: str
    paper_route_fingerprint: str
    component_id: str
    adapter_class: str
    recipe_id: str
    method_identity_status: str
    branch_id: str | None = None
    disposition: PaperRouteDisposition
    reason_codes: list[str] = Field(default_factory=list)
    teacher_disposition: str = ""
    student_disposition: str = ""
    teacher_recovery_action: str = ""
    student_recovery_action: str = ""
    route_checks: dict[str, bool] = Field(default_factory=dict)
    report_hash: str = ""

    @model_validator(mode="after")
    def bind_report_hash(self) -> "DistillationPaperRouteReport":
        expected = compute_paper_route_report_hash(self)
        if self.report_hash and self.report_hash != expected:
            raise ValueError("paper route report hash mismatch")
        self.report_hash = expected
        return self


def compute_paper_route_report_hash(report: DistillationPaperRouteReport) -> str:
    payload = report.model_dump(
        mode="json", exclude={"report_hash", "schema_version"}
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _route_checks(route: DistillationPaperRoute) -> dict[str, bool]:
    schema_keys = (
        "paper_id",
        "paper_route_fingerprint",
        "recipe_id",
        "teacher",
        "teacher_sha256",
        "teacher_architecture",
        "teacher_split",
        "student",
        "student_sha256",
        "student_architecture",
        "student_split",
        "dataset_hash",
        "matched_baseline",
    )
    return {
        "paper_route_identity_bound": (
            route.component_id == route.paper_specific_mechanism_id
        ),
        "paper_route_fingerprint_present": bool(route.execution_fingerprint),
        "changed_variables_present": "distillation.method" in route.changed_variables,
        "payload_schema_complete": all(key in route.runtime_payload_schema for key in schema_keys),
        "student_fixed_yolo26n": route.student_protocol.get("architecture") == "yolo26n",
        "student_only_export_protocol": route.student_only_export is True,
        "matched_baseline_required": route.matched_baseline_required is True,
        "branch_binding_consistent": (
            route.branch_id is None
        ) == (route.method_identity_status == "identity_recovery"),
        "independent_adapter_class": route.adapter_class.startswith("Distillation")
        and route.adapter_class.endswith("Adapter"),
    }


def _disposition_from_reasons(reasons: list[str]) -> PaperRouteDisposition:
    if any(any(marker in reason for marker in _MISMATCH_MARKERS) for reason in reasons):
        return "blocked_runtime"
    return "evidence_recovery"


def certify_distillation_paper_route(
    paper_id: str,
    *,
    workspace: Path | str,
    teacher: str = "yolo26s.pt",
    student: str = "yolo26n.pt",
    expected_teacher_sha256: str | None = None,
    expected_student_sha256: str | None = None,
    dataset_manifest_hash: str | None = None,
    split: str = "train",
    imgsz: int = 640,
    require_metadata: bool = True,
    matched_baseline: dict[str, Any] | None = None,
) -> DistillationPaperRouteReport:
    """Certify one paper's independent route on CPU without GPU training."""
    route = default_paper_route_registry().route(paper_id)
    checks = _route_checks(route)

    teacher_resolution: CheckpointResolution = resolve_teacher_checkpoint(
        teacher,
        workspace=workspace,
        expected_sha256=expected_teacher_sha256,
        expected_dataset_hash=dataset_manifest_hash,
        expected_split=split,
        expected_imgsz=imgsz,
        require_metadata=require_metadata,
    )
    student_resolution: CheckpointResolution = resolve_student_checkpoint(
        student,
        workspace=workspace,
        expected_sha256=expected_student_sha256,
        expected_dataset_hash=dataset_manifest_hash,
        expected_split=split,
        expected_imgsz=imgsz,
        require_metadata=require_metadata,
    )

    reason_codes: list[str] = []
    reason_codes.extend(teacher_resolution.reason_codes)
    reason_codes.extend(student_resolution.reason_codes)

    checks["teacher_checkpoint_resolved"] = (
        teacher_resolution.identity.sha256 is not None
    )
    checks["teacher_hash_bound"] = bool(
        expected_teacher_sha256
        and teacher_resolution.identity.sha256 == expected_teacher_sha256
    )
    checks["teacher_split_match"] = bool(
        teacher_resolution.identity.split is None
        or teacher_resolution.identity.split == split
    )
    checks["student_checkpoint_resolved"] = (
        student_resolution.identity.sha256 is not None
    )
    checks["student_architecture_yolo26n"] = (
        student_resolution.identity.architecture == "yolo26n"
    )
    checks["matched_baseline_present"] = matched_baseline is not None
    if matched_baseline is None:
        reason_codes.append("matched_baseline_missing")
    if route.method_identity_status == "identity_recovery":
        reason_codes.extend(route.reason_codes)

    if (
        teacher_resolution.disposition == "runtime_ready"
        and student_resolution.disposition == "runtime_ready"
        and matched_baseline is not None
    ):
        disposition: PaperRouteDisposition = "runtime_ready"
    else:
        disposition = _disposition_from_reasons(list(reason_codes))

    reason_codes = list(dict.fromkeys(reason_codes))
    return DistillationPaperRouteReport(
        paper_id=route.paper_id,
        paper_route_fingerprint=route.execution_fingerprint,
        component_id=route.component_id,
        adapter_class=route.adapter_class,
        recipe_id=route.recipe_id,
        method_identity_status=route.method_identity_status,
        branch_id=route.branch_id,
        disposition=disposition,
        reason_codes=reason_codes,
        teacher_disposition=teacher_resolution.disposition,
        student_disposition=student_resolution.disposition,
        teacher_recovery_action=teacher_resolution.recovery_action,
        student_recovery_action=student_resolution.recovery_action,
        route_checks=checks,
    )


class DistillationPaperRouteCertificationSummary(BaseModel, YAMLModelMixin):
    """Persistent coverage summary for one paper-route certification run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "distillation_paper_route_certification.v1"
    papers_total: int
    runtime_ready: int
    evidence_recovery: int
    blocked_runtime: int
    silent_drops: list[str] = Field(default_factory=list)
    reports: list[DistillationPaperRouteReport]
    summary_hash: str = ""

    @model_validator(mode="after")
    def bind_summary(self) -> "DistillationPaperRouteCertificationSummary":
        if self.silent_drops:
            raise ValueError(f"paper route certification silent drops: {self.silent_drops}")
        paper_ids = [item.paper_id for item in self.reports]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("paper route certification contains duplicate papers")
        if self.papers_total != len(paper_ids):
            raise ValueError("every paper must carry exactly one certification report")
        counts = (self.runtime_ready, self.evidence_recovery, self.blocked_runtime)
        if sum(counts) != self.papers_total:
            raise ValueError("paper route dispositions must cover every paper")
        expected = compute_certification_summary_hash(self)
        if self.summary_hash and self.summary_hash != expected:
            raise ValueError("paper route certification summary hash mismatch")
        self.summary_hash = expected
        return self


def compute_certification_summary_hash(
    summary: DistillationPaperRouteCertificationSummary,
) -> str:
    payload = summary.model_dump(mode="json", exclude={"summary_hash", "schema_version"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def certify_all_paper_routes(
    *,
    output_path: Path | str,
    workspace: Path | str,
    paper_ids: tuple[str, ...] | None = None,
    teacher: str = "yolo26s.pt",
    student: str = "yolo26n.pt",
    expected_teacher_sha256: str | None = None,
    expected_student_sha256: str | None = None,
    dataset_manifest_hash: str | None = None,
    split: str = "train",
    imgsz: int = 640,
    matched_baseline: dict[str, Any] | None = None,
) -> DistillationPaperRouteCertificationSummary:
    """Certify every paper route and persist one coverage summary."""
    reports = certify_distillation_paper_routes(
        workspace=workspace,
        paper_ids=paper_ids,
        teacher=teacher,
        student=student,
        expected_teacher_sha256=expected_teacher_sha256,
        expected_student_sha256=expected_student_sha256,
        dataset_manifest_hash=dataset_manifest_hash,
        split=split,
        imgsz=imgsz,
        matched_baseline=matched_baseline,
    )
    registry = default_paper_route_registry()
    if paper_ids is not None:
        expected_ids = tuple(paper_ids)
    else:
        expected_ids = tuple(item.paper_id for item in registry.routes())
    found = {item.paper_id for item in reports}
    summary = DistillationPaperRouteCertificationSummary(
        papers_total=len(expected_ids),
        runtime_ready=sum(item.disposition == "runtime_ready" for item in reports),
        evidence_recovery=sum(
            item.disposition == "evidence_recovery" for item in reports
        ),
        blocked_runtime=sum(item.disposition == "blocked_runtime" for item in reports),
        silent_drops=[paper_id for paper_id in expected_ids if paper_id not in found],
        reports=reports,
    )
    summary.to_yaml(output_path, exclude_none=True, sort_keys=False)
    return summary


def certify_distillation_paper_routes(
    *,
    workspace: Path | str,
    paper_ids: tuple[str, ...] | None = None,
    teacher: str = "yolo26s.pt",
    student: str = "yolo26n.pt",
    expected_teacher_sha256: str | None = None,
    expected_student_sha256: str | None = None,
    dataset_manifest_hash: str | None = None,
    split: str = "train",
    imgsz: int = 640,
    require_metadata: bool = True,
    matched_baseline: dict[str, Any] | None = None,
) -> list[DistillationPaperRouteReport]:
    """Certify every requested paper route without silently dropping any."""
    registry = default_paper_route_registry()
    if paper_ids is not None:
        ids = tuple(paper_ids)
    else:
        ids = tuple(item.paper_id for item in registry.routes())
    return [
        certify_distillation_paper_route(
            paper_id,
            workspace=workspace,
            teacher=teacher,
            student=student,
            expected_teacher_sha256=expected_teacher_sha256,
            expected_student_sha256=expected_student_sha256,
            dataset_manifest_hash=dataset_manifest_hash,
            split=split,
            imgsz=imgsz,
            require_metadata=require_metadata,
            matched_baseline=matched_baseline,
        )
        for paper_id in ids
    ]


__all__ = [
    "DistillationPaperRouteCertificationSummary",
    "DistillationPaperRouteReport",
    "PaperRouteDisposition",
    "certify_all_paper_routes",
    "certify_distillation_paper_route",
    "certify_distillation_paper_routes",
    "compute_certification_summary_hash",
    "compute_paper_route_report_hash",
]
