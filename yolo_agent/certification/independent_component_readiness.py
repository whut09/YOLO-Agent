"""Fail-closed readiness routing for independent YOLO26 paper components.

The existing independent-route certification proves that an adapter can be
imported and smoke-tested on CPU.  This module adds the real artifact gates
needed before a candidate can be marked runtime-ready or admitted to ASHA.
It never creates evidence and never probes CUDA.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.certification.independent_component_routes import (
    IndependentComponentRouteReport,
    certify_independent_component_route,
)
from yolo_agent.components.independent_component_router import (
    ASSIGNMENT_SHADOW_COMPONENTS,
    COMPONENT_CATALOG,
    INDEPENDENT_COMPONENT_IDS,
    IndependentComponentId,
)
from yolo_agent.core.yaml_io import YAMLModelMixin


IndependentReadinessDisposition = Literal[
    "runtime_ready",
    "queued",
    "evidence_recovery",
    "implementation_request",
    "blocked_runtime",
    "incompatible",
]

REQUIRED_READINESS_CHECKS: tuple[str, ...] = (
    "contract_present",
    "implementation_path",
    "adapter_class",
    "changed_variable",
    "runtime_hook",
    "payload_schema",
    "evidence_artifact",
    "adapter_hash",
    "protocol_hash",
    "fixed_imgsz_640",
    "yolo26_one_to_one_head",
    "native_dfl_free_regression",
    "matched_baseline",
    "cpu_contract_shape_forward_backward",
)


class IndependentComponentReadiness(BaseModel, YAMLModelMixin):
    """One independently routed component's runtime authorization result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "independent_component_readiness.v1"
    component_id: IndependentComponentId
    recipe_id: str
    implementation_path: str
    adapter_class: str
    changed_variable: str
    runtime_hook: str
    runtime_payload_field: str
    graph_identity: str
    evidence_artifact: str | None = None
    matched_baseline_artifact: str | None = None
    shadow_evidence_artifact: str | None = None
    adapter_hash: str | None = None
    protocol_hash: str | None = None
    fixed_imgsz: int = 640
    yolo26_one_to_one_head_compatible: bool = False
    native_dfl_free_regression_compatible: bool = False
    paired_baseline_required: bool = True
    cpu_checks: dict[str, bool] = Field(default_factory=dict)
    runtime_checks: dict[str, bool] = Field(default_factory=dict)
    runtime_ready: bool = False
    asha_eligible: bool = False
    training_candidate_allowed: bool = True
    disposition: IndependentReadinessDisposition
    reason_codes: list[str] = Field(default_factory=list)
    recovery_action: str
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_boundary(self) -> "IndependentComponentReadiness":
        required_cpu_checks = set(REQUIRED_READINESS_CHECKS) - {
            "matched_baseline",
        }
        missing_cpu_checks = sorted(required_cpu_checks - set(self.cpu_checks))
        if missing_cpu_checks:
            raise ValueError(
                "independent readiness is missing CPU checks: "
                + ", ".join(missing_cpu_checks)
            )
        required_runtime_checks = {
            "evidence_artifact",
            "matched_baseline",
            "protocol_bound",
            "shadow_evidence",
        }
        missing_runtime_checks = sorted(
            required_runtime_checks - set(self.runtime_checks)
        )
        if missing_runtime_checks:
            raise ValueError(
                "independent readiness is missing runtime checks: "
                + ", ".join(missing_runtime_checks)
            )
        if self.disposition == "runtime_ready" and self.reason_codes:
            raise ValueError("runtime-ready route cannot retain reason codes")
        if self.runtime_ready != (self.disposition == "runtime_ready"):
            raise ValueError("runtime_ready must match the route disposition")
        if self.fixed_imgsz != 640 and self.runtime_ready:
            raise ValueError("non-640 route cannot be runtime ready")
        if self.runtime_ready and (
            not all(self.cpu_checks.values()) or not all(self.runtime_checks.values())
        ):
            raise ValueError("runtime-ready route requires every readiness check")
        if self.runtime_ready and not all(
            getattr(self, field)
            for field in (
                "implementation_path",
                "adapter_class",
                "changed_variable",
                "runtime_hook",
                "runtime_payload_field",
                "graph_identity",
                "adapter_hash",
                "protocol_hash",
            )
        ):
            raise ValueError("runtime-ready route requires every route identity field")
        if self.runtime_ready and not self.paired_baseline_required:
            raise ValueError("runtime-ready route requires a matched baseline gate")
        if not self.runtime_ready and not self.reason_codes:
            raise ValueError("blocked readiness route requires reason codes")
        if self.asha_eligible:
            if not self.training_candidate_allowed:
                raise ValueError("non-training route cannot be ASHA eligible")
            if self.disposition != "runtime_ready":
                raise ValueError("ASHA eligible route must be runtime_ready")
        if self.component_id in ASSIGNMENT_SHADOW_COMPONENTS and self.asha_eligible:
            if not self.shadow_evidence_artifact:
                raise ValueError("assignment ASHA route requires shadow evidence")
        expected = compute_readiness_hash(self)
        if self.report_hash and self.report_hash != expected:
            raise ValueError("independent readiness report hash mismatch")
        self.report_hash = expected
        return self


class IndependentComponentReadinessSummary(BaseModel, YAMLModelMixin):
    """Complete readiness coverage for all requested independent routes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "independent_component_readiness_summary.v1"
    components_total: int
    runtime_ready_count: int
    asha_eligible_count: int
    blocked_count: int
    inference_only_components: list[str] = Field(default_factory=list)
    silent_drops: list[str] = Field(default_factory=list)
    reports: list[IndependentComponentReadiness]
    summary_hash: str = ""

    @model_validator(mode="after")
    def validate_summary(self) -> "IndependentComponentReadinessSummary":
        ids = [item.component_id for item in self.reports]
        if self.components_total != len(ids):
            raise ValueError("readiness summary must contain one report per component")
        if len(ids) != len(set(ids)):
            raise ValueError("readiness summary contains duplicate components")
        if self.silent_drops:
            raise ValueError(f"readiness summary contains silent drops: {self.silent_drops}")
        if self.runtime_ready_count != sum(
            item.disposition == "runtime_ready" for item in self.reports
        ):
            raise ValueError("runtime_ready_count does not match reports")
        if self.asha_eligible_count != sum(
            item.asha_eligible for item in self.reports
        ):
            raise ValueError("asha_eligible_count does not match reports")
        if self.blocked_count != sum(
            item.disposition
            in {"evidence_recovery", "implementation_request", "blocked_runtime"}
            for item in self.reports
        ):
            raise ValueError("blocked_count does not match reports")
        expected = compute_summary_hash(self)
        if self.summary_hash and self.summary_hash != expected:
            raise ValueError("independent readiness summary hash mismatch")
        self.summary_hash = expected
        return self


def compute_readiness_hash(report: IndependentComponentReadiness) -> str:
    payload = report.model_dump(
        mode="json", exclude={"report_hash", "schema_version", "generated_at"}
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compute_summary_hash(summary: IndependentComponentReadinessSummary) -> str:
    payload = summary.model_dump(
        mode="json", exclude={"summary_hash", "schema_version"}
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assess_independent_component_readiness(
    component_id: IndependentComponentId,
    *,
    evidence_artifact: Path | str | None = None,
    matched_baseline_artifact: Path | str | None = None,
    shadow_evidence_artifact: Path | str | None = None,
    workspace: Path | str | None = None,
    imgsz: int = 640,
) -> IndependentComponentReadiness:
    """Assess one route using only existing local files and CPU checks.

    ``None`` means the caller has not supplied a real artifact.  It is never
    converted into a guessed path.  A successful CPU smoke therefore remains
    blocked until the corresponding runtime evidence exists.
    """

    evidence_path = _existing_file(evidence_artifact)
    baseline_path = _existing_file(matched_baseline_artifact)
    shadow_path = _existing_file(shadow_evidence_artifact)
    catalog = COMPONENT_CATALOG[component_id]
    route = certify_independent_component_route(
        component_id,
        workspace=workspace,
        matched_baseline=baseline_path is not None,
        imgsz=imgsz,
    )
    reasons = list(route.reason_codes)
    cpu_checks = dict(route.checks)
    if route.recipe_id != str(catalog.get("recipe_id", "")):
        cpu_checks["recipe_id"] = False
        reasons.append("recipe_id_catalog_mismatch")
    if route.graph_identity != str(catalog.get("graph_identity", component_id)):
        cpu_checks["graph_identity"] = False
        reasons.append("graph_identity_catalog_mismatch")
    if route.runtime_payload_field != str(catalog.get("runtime_payload_field", "")):
        cpu_checks["runtime_payload_field"] = False
        reasons.append("runtime_payload_field_catalog_mismatch")
    if route.evidence_artifact != str(catalog.get("evidence_artifact", "")):
        cpu_checks["evidence_artifact"] = False
        reasons.append("evidence_artifact_catalog_mismatch")
    for check_name in REQUIRED_READINESS_CHECKS:
        if check_name == "matched_baseline":
            continue
        if not cpu_checks.get(check_name, False):
            reasons.append(
                "fixed_imgsz_640_required"
                if check_name == "fixed_imgsz_640"
                else f"missing_field:{check_name}"
            )
    cpu_checks["cpu_contract_shape_forward_backward"] = _cpu_smoke_complete(route)
    if not cpu_checks["cpu_contract_shape_forward_backward"]:
        reasons.append("cpu_contract_shape_forward_backward_failed")

    protocol_hash = route.protocol_hash
    runtime_checks: dict[str, bool] = {
        "evidence_artifact": False,
        "matched_baseline": False,
        "protocol_bound": bool(protocol_hash),
        "shadow_evidence": component_id not in ASSIGNMENT_SHADOW_COMPONENTS,
    }
    if evidence_path is None:
        reasons.append("evidence_artifact_missing")
    else:
        evidence, error = _read_mapping(evidence_path)
        if error:
            reasons.append(f"evidence_artifact_invalid:{error}")
        else:
            evidence_errors = _validate_evidence(
                evidence,
                component_id=component_id,
                protocol_hash=protocol_hash,
            )
            reasons.extend(evidence_errors)
            runtime_checks["evidence_artifact"] = not evidence_errors

    if baseline_path is None:
        reasons.append("matched_baseline_artifact_missing")
    else:
        baseline, error = _read_mapping(baseline_path)
        if error:
            reasons.append(f"matched_baseline_artifact_invalid:{error}")
        else:
            baseline_errors = _validate_baseline(
                baseline,
                protocol_hash=protocol_hash,
            )
            reasons.extend(baseline_errors)
            runtime_checks["matched_baseline"] = not baseline_errors

    if component_id in ASSIGNMENT_SHADOW_COMPONENTS:
        if shadow_path is None:
            reasons.append("assignment_shadow_evidence_missing")
        else:
            shadow, error = _read_mapping(shadow_path)
            if error:
                reasons.append(f"assignment_shadow_evidence_invalid:{error}")
            else:
                shadow_errors = _validate_shadow(
                    shadow,
                    component_id=component_id,
                    protocol_hash=protocol_hash,
                )
                reasons.extend(shadow_errors)
                runtime_checks["shadow_evidence"] = not shadow_errors

    if route.inference_only:
        reasons.append("inference_only_not_training_candidate")
    reasons = list(dict.fromkeys(reasons))
    cpu_ready = all(cpu_checks.get(name, False) for name in (
        "contract_present",
        "recipe_id",
        "implementation_path",
        "adapter_class",
        "changed_variable",
        "runtime_hook",
        "runtime_payload_field",
        "graph_identity",
        "payload_schema",
        "evidence_artifact",
        "adapter_hash",
        "protocol_hash",
        "fixed_imgsz_640",
        "yolo26_one_to_one_head",
        "native_dfl_free_regression",
        "cpu_contract_shape_forward_backward",
    ))
    runtime_ready = cpu_ready and all(runtime_checks.values())
    training_allowed = not route.inference_only
    asha = runtime_ready and training_allowed
    disposition = _disposition(
        reasons,
        runtime_ready=runtime_ready,
        inference_only=route.inference_only,
    )
    return IndependentComponentReadiness(
        component_id=component_id,
        recipe_id=route.recipe_id,
        implementation_path=route.implementation_path,
        adapter_class=route.adapter_class,
        changed_variable=route.changed_variable,
        runtime_hook=route.runtime_hook,
        runtime_payload_field=route.runtime_payload_field,
        graph_identity=route.graph_identity,
        evidence_artifact=str(evidence_path) if evidence_path else None,
        matched_baseline_artifact=str(baseline_path) if baseline_path else None,
        shadow_evidence_artifact=str(shadow_path) if shadow_path else None,
        adapter_hash=route.adapter_source_sha256,
        protocol_hash=protocol_hash,
        fixed_imgsz=imgsz,
        yolo26_one_to_one_head_compatible=bool(
            route.checks.get("yolo26_one_to_one_head")
        ),
        native_dfl_free_regression_compatible=bool(
            route.checks.get("native_dfl_free_regression")
        ),
        paired_baseline_required=route.paired_baseline_required,
        cpu_checks=cpu_checks,
        runtime_checks=runtime_checks,
        asha_eligible=asha,
        runtime_ready=runtime_ready,
        training_candidate_allowed=training_allowed,
        disposition=disposition,
        reason_codes=[] if runtime_ready else reasons,
        recovery_action=_recovery_action(reasons, inference_only=route.inference_only),
    )


def assess_independent_component_routes(
    *,
    component_ids: Iterable[IndependentComponentId] = INDEPENDENT_COMPONENT_IDS,
    evidence_by_component: dict[str, Path | str] | None = None,
    matched_baseline_by_component: dict[str, Path | str] | None = None,
    shadow_evidence_by_component: dict[str, Path | str] | None = None,
    workspace: Path | str | None = None,
    imgsz: int = 640,
) -> IndependentComponentReadinessSummary:
    """Assess every requested route and fail if any route silently disappears."""

    ids = tuple(component_ids)
    reports = [
        assess_independent_component_readiness(
            component_id,
            evidence_artifact=(evidence_by_component or {}).get(component_id),
            matched_baseline_artifact=(matched_baseline_by_component or {}).get(component_id),
            shadow_evidence_artifact=(shadow_evidence_by_component or {}).get(component_id),
            workspace=workspace,
            imgsz=imgsz,
        )
        for component_id in ids
    ]
    found = {item.component_id for item in reports}
    return IndependentComponentReadinessSummary(
        components_total=len(ids),
        runtime_ready_count=sum(item.disposition == "runtime_ready" for item in reports),
        asha_eligible_count=sum(item.asha_eligible for item in reports),
        blocked_count=sum(
            item.disposition
            in {"evidence_recovery", "implementation_request", "blocked_runtime"}
            for item in reports
        ),
        inference_only_components=[
            item.component_id for item in reports if not item.training_candidate_allowed
        ],
        silent_drops=[item for item in ids if item not in found],
        reports=reports,
    )


def _existing_file(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve(strict=False)
    return path if path.is_file() else None


def _read_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return {}, type(exc).__name__
    if not isinstance(payload, dict):
        return {}, "root_not_mapping"
    return payload, None


def _validate_evidence(
    payload: dict[str, Any],
    *,
    component_id: str,
    protocol_hash: str | None,
) -> list[str]:
    errors: list[str] = []
    if payload.get("component_id") != component_id:
        errors.append("evidence_component_mismatch")
    evidence_protocol = payload.get("protocol_hash") or payload.get("runtime_protocol_hash")
    if not evidence_protocol:
        errors.append("evidence_protocol_hash_missing")
    elif protocol_hash and evidence_protocol != protocol_hash:
        errors.append("evidence_protocol_hash_mismatch")
    passed = payload.get("passed") is True or payload.get("status") in {
        "passed",
        "completed",
        "runtime_ready",
        "shadow_evidence_complete",
    }
    if not passed:
        errors.append("evidence_not_passed")
    return errors


def _validate_baseline(payload: dict[str, Any], *, protocol_hash: str | None) -> list[str]:
    errors: list[str] = []
    if not payload.get("protocol_hash"):
        errors.append("matched_baseline_protocol_hash_missing")
    elif protocol_hash and payload["protocol_hash"] != protocol_hash:
        errors.append("matched_baseline_protocol_mismatch")
    if payload.get("imgsz") is None:
        errors.append("matched_baseline_imgsz_missing")
    elif payload.get("imgsz") != 640:
        errors.append("matched_baseline_imgsz_mismatch")
    if payload.get("split") in {"", None}:
        errors.append("matched_baseline_split_missing")
    if payload.get("status") not in {"passed", "completed", "verified"}:
        errors.append("matched_baseline_not_completed")
    return errors


def _validate_shadow(
    payload: dict[str, Any],
    *,
    component_id: str,
    protocol_hash: str | None,
) -> list[str]:
    errors = _validate_evidence(
        payload,
        component_id=component_id,
        protocol_hash=protocol_hash,
    )
    minimum_batches = payload.get("minimum_batches", payload.get("batches"))
    if minimum_batches is not None and int(minimum_batches) < 1:
        errors.append("assignment_shadow_minimum_batches_missing")
    for key in ("positive_assignment_valid", "native_loss_equivalence"):
        if key in payload and payload[key] is not True:
            errors.append(f"assignment_shadow_{key}_failed")
    return list(dict.fromkeys(errors))


def _cpu_smoke_complete(report: IndependentComponentRouteReport) -> bool:
    checks = report.cpu_smoke_checks
    return bool(
        report.disposition == "certified_route"
        and report.checks.get("cpu_smoke") is True
        and checks.get("forward", True) is True
        and checks.get("backward", True) is True
        and checks.get("shape", True) is not False
    )


def _disposition(
    reasons: list[str],
    *,
    runtime_ready: bool,
    inference_only: bool,
) -> IndependentReadinessDisposition:
    if runtime_ready:
        return "runtime_ready"
    if inference_only:
        return "incompatible"
    if any(
        token in reason
        for reason in reasons
        for token in (
            "evidence",
            "baseline",
            "shadow",
            "protocol",
        )
    ):
        return "evidence_recovery"
    if any(reason.startswith("probe_failed") for reason in reasons):
        return "implementation_request"
    return "blocked_runtime"


def _recovery_action(reasons: list[str], *, inference_only: bool) -> str:
    if inference_only:
        return "run the inference-only route; do not enqueue it in training ASHA"
    actions: list[str] = []
    if any("evidence" in reason for reason in reasons):
        actions.append("write a real passed evidence artifact bound to the candidate protocol hash")
    if any("baseline" in reason for reason in reasons):
        actions.append("produce a verified matched baseline with the same protocol, split, and imgsz=640")
    if any("shadow" in reason for reason in reasons):
        actions.append("complete assignment shadow evidence before active candidate admission")
    if any("probe" in reason or "missing_field" in reason for reason in reasons):
        actions.append("repair the component contract or runtime adapter")
    return "; ".join(dict.fromkeys(actions)) or "resolve the recorded readiness blocker"


__all__ = [
    "IndependentComponentReadiness",
    "IndependentComponentReadinessSummary",
    "REQUIRED_READINESS_CHECKS",
    "assess_independent_component_readiness",
    "assess_independent_component_routes",
    "compute_readiness_hash",
    "compute_summary_hash",
]
