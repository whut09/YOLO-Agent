"""CPU readiness certification for paper-specific domain routes.

The certification is fail-closed and GPU-free: it resolves the real source
and target domain manifests, binds their hashes, splits, and label
availability to the paper route's protocol, and reports a recoverable
disposition.  A paper without real, distinct, hash-verified domains never
reports runtime readiness, and mismatched domain bindings block the runtime
instead of silently recovering.  COCO supervised data can never stand in
for a paper domain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.domain_adaptation.branches import (
    DOMAIN_BRANCH_PROFILES,
)
from yolo_agent.components.adapters.domain_adaptation.domain_evidence import (
    DomainDatasetManifest,
    DomainEvidenceError,
    LabelAvailability,
    manifest_from_file,
    resolve_domain_protocol,
)
from yolo_agent.components.adapters.domain_adaptation.domain_paper_routes import (
    DomainPaperRoute,
    default_domain_paper_route_registry,
)
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.paper_asset_schemas import (
    PaperAssetRecord,
    PaperAssetRegistry,
)


DomainRouteDisposition = Literal[
    "runtime_ready",
    "evidence_recovery",
    "blocked_runtime",
]

_MISMATCH_MARKERS = (
    "sha256_mismatch",
    "identity_collision",
    "domain_id_collision",
    "domain_pair_id_mismatch",
    "coco_supervised_data_cannot_be_domain_pair",
    "label_availability_mismatch",
    "protocol_hash_mismatch",
    "domain_loss_disabled",
    "split_mismatch",
    "source_free_source_data_forbidden",
)

_TARGET_LABEL_BY_REQUIREMENT = {
    "source_labeled_target_unlabeled": "unlabeled",
    "pseudo": "pseudo",
    "unlabeled": "unlabeled",
    "partial": "partial",
}


class DomainPaperRouteReport(BaseModel, YAMLModelMixin):
    """CPU certification result for one paper's independent domain route."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "domain_paper_route_report.v1"
    paper_id: str
    paper_route_fingerprint: str
    component_id: str
    adapter_class: str
    recipe_id: str
    branch_id: str
    adaptation_mode: str
    source_free: bool = False
    domain_pair_id: str = ""
    domain_protocol_hash: str = ""
    disposition: DomainRouteDisposition
    reason_codes: list[str] = Field(default_factory=list)
    source_disposition: str = ""
    target_disposition: str = ""
    source_recovery_action: str = ""
    target_recovery_action: str = ""
    route_checks: dict[str, bool] = Field(default_factory=dict)
    allows_asha: bool = False
    report_hash: str = ""

    @model_validator(mode="after")
    def bind_report_hash(self) -> "DomainPaperRouteReport":
        if self.allows_asha and self.disposition != "runtime_ready":
            raise ValueError(
                "domain route certification allows ASHA only when runtime ready"
            )
        expected = compute_domain_route_report_hash(self)
        if self.report_hash and self.report_hash != expected:
            raise ValueError("domain route report hash mismatch")
        self.report_hash = expected
        return self


def compute_domain_route_report_hash(report: DomainPaperRouteReport) -> str:
    payload = report.model_dump(
        mode="json", exclude={"report_hash", "schema_version"}
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _route_checks(route: DomainPaperRoute) -> dict[str, bool]:
    return {
        "paper_route_identity_bound": (
            route.component_id == route.paper_specific_mechanism_id
        ),
        "paper_route_fingerprint_present": bool(route.execution_fingerprint),
        "changed_variables_present": "domain_adaptation.method"
        in route.changed_variables,
        "branch_component_not_reused": (
            route.component_id != route.branch_component_id
        ),
        "independent_adapter_class": (
            route.adapter_class.startswith("DomainAdaptation")
            and route.adapter_class != "DomainAdaptationBranchAdapter"
        ),
        "adapter_hash_bound": len(route.adapter_hash) == 64,
        "coco_forbidden_in_protocols": (
            route.source_protocol.get("coco_supervised_forbidden") is True
            and route.target_protocol.get("coco_supervised_forbidden") is True
        ),
        "mock_target_forbidden": route.target_protocol.get("mock_forbidden") is True,
    }


def _disposition_from_reasons(reasons: list[str]) -> DomainRouteDisposition:
    if any(
        any(marker in reason for marker in _MISMATCH_MARKERS) for reason in reasons
    ):
        return "blocked_runtime"
    return "evidence_recovery"


def _resolve_manifest(
    path: str | None,
    *,
    role: Literal["source", "target"],
    dataset_hash: str | None,
    domain_id: str,
    domain_name: str,
    split: str,
    label_availability: LabelAvailability,
    expected_sha256: str | None,
    coco: bool,
) -> tuple[DomainDatasetManifest | None, list[str], str]:
    """Load one domain manifest; return (manifest, reason_codes, disposition)."""
    if path is None:
        return None, [f"{role}_domain_manifest_missing"], "evidence_recovery"
    if not dataset_hash:
        return None, [f"{role}_dataset_hash_unbound"], "evidence_recovery"
    try:
        manifest = manifest_from_file(
            path,
            role=role,
            dataset_hash=dataset_hash,
            domain_id=domain_id,
            domain_name=domain_name,
            split=split,
            label_availability=label_availability,
        )
    except DomainEvidenceError as exc:
        message = str(exc)
        if "COCO" in message:
            return (
                None,
                ["coco_supervised_data_cannot_be_domain_pair"],
                "blocked_runtime",
            )
        return None, [f"{role}_domain_manifest_unreadable"], "evidence_recovery"
    if coco:
        return (
            None,
            ["coco_supervised_data_cannot_be_domain_pair"],
            "blocked_runtime",
        )
    if expected_sha256 and manifest.sha256 != expected_sha256:
        return (
            None,
            [f"{role}_manifest_sha256_mismatch"],
            "blocked_runtime",
        )
    disposition = (
        "runtime_ready" if expected_sha256 else "evidence_recovery"
    )
    reasons = [] if expected_sha256 else [f"{role}_manifest_sha256_unbound"]
    return manifest, reasons, disposition


def certify_domain_paper_route(
    paper_id: str,
    *,
    workspace: Path | str,
    source: str | None = None,
    target: str | None = None,
    source_sha256: str | None = None,
    target_sha256: str | None = None,
    source_dataset_hash: str | None = None,
    target_dataset_hash: str | None = None,
    source_domain_id: str = "source",
    target_domain_id: str = "target",
    source_domain_name: str = "source",
    target_domain_name: str = "target",
    source_split: str = "train",
    target_split: str = "train",
    source_label_availability: LabelAvailability | None = None,
    target_label_availability: LabelAvailability | None = None,
    source_coco: bool = False,
    target_coco: bool = False,
    domain_pair_id: str | None = None,
    expected_protocol_hash: str | None = None,
    adaptation_weight: float = 0.05,
    teacher_checkpoint: str | None = None,
    teacher_sha256: str | None = None,
    source_model_checkpoint: str | None = None,
    source_model_sha256: str | None = None,
    source_model_protocol_hash: str | None = None,
) -> DomainPaperRouteReport:
    """Certify one paper's independent domain route on CPU without training."""
    route = default_domain_paper_profile(paper_id)
    profile = DOMAIN_BRANCH_PROFILES[route.branch_id]
    checks = _route_checks(route)

    reason_codes: list[str] = []
    source_disposition = "not_required"
    target_disposition = "missing"
    source_recovery = ""
    target_recovery = "provide a real, distinct, hash-bound target domain manifest"
    protocol = None

    default_source_label: LabelAvailability = "labeled"
    default_target_label: LabelAvailability = (
        _TARGET_LABEL_BY_REQUIREMENT[profile["required_label_availability"]]
    )
    source_label = source_label_availability or default_source_label
    target_label = target_label_availability or default_target_label

    source_manifest: DomainDatasetManifest | None = None
    if route.requires_source_domain:
        source_manifest, source_reasons, source_disposition = _resolve_manifest(
            source,
            role="source",
            dataset_hash=source_dataset_hash,
            domain_id=source_domain_id,
            domain_name=source_domain_name,
            split=source_split,
            label_availability=source_label,
            expected_sha256=source_sha256,
            coco=source_coco,
        )
        reason_codes.extend(source_reasons)
        if source_disposition != "runtime_ready":
            source_recovery = (
                "provide a real, distinct, hash-bound source domain manifest"
            )
    target_manifest: DomainDatasetManifest | None = None
    target_manifest, target_reasons, target_disposition = _resolve_manifest(
        target,
        role="target",
        dataset_hash=target_dataset_hash,
        domain_id=target_domain_id,
        domain_name=target_domain_name,
        split=target_split,
        label_availability=target_label,
        expected_sha256=target_sha256,
        coco=target_coco,
    )
    reason_codes.extend(target_reasons)

    if route.source_free and source is not None:
        reason_codes.append("source_free_source_data_forbidden")

    source_resolved = source_manifest is not None or not route.requires_source_domain
    target_resolved = target_manifest is not None
    if source_resolved and target_resolved:
        try:
            protocol = resolve_domain_protocol(
                source=source_manifest,
                target=target_manifest,
                adaptation_mode=route.adaptation_mode,  # type: ignore[arg-type]
                domain_pair_id=domain_pair_id,
                source_free=route.source_free,
                source_model_checkpoint_sha256=source_model_sha256,
                source_model_protocol_hash=source_model_protocol_hash,
            )
        except DomainEvidenceError as exc:
            reason_codes.append(f"domain_protocol_error:{exc}")
        if protocol is not None and protocol.reason_codes:
            reason_codes.extend(protocol.reason_codes)
        if protocol is not None and protocol.ok and domain_pair_id:
            if protocol.pair is not None and protocol.pair.domain_pair_id != domain_pair_id:
                reason_codes.append("domain_pair_id_mismatch")

    # The declared label availability must match the branch requirement.
    checks["target_label_availability_match"] = (
        target_manifest is None
        or target_manifest.label_availability == default_target_label
    )
    if not checks["target_label_availability_match"]:
        reason_codes.append("target_label_availability_mismatch")
    checks["source_label_availability_match"] = (
        not route.requires_source_domain
        or source_manifest is None
        or source_manifest.label_availability == "labeled"
    )
    if not checks["source_label_availability_match"]:
        reason_codes.append("source_label_availability_mismatch")

    # Protocol hash binding: a bound hash must match; an unbound one stays
    # in evidence recovery instead of silently authorizing.
    checks["domain_protocol_hash_bound"] = bool(expected_protocol_hash)
    checks["domain_protocol_hash_match"] = (
        protocol is not None
        and expected_protocol_hash is not None
        and protocol.ok
        and protocol.protocol_hash == expected_protocol_hash
    )
    if protocol is not None and protocol.ok and expected_protocol_hash is None:
        reason_codes.append("domain_protocol_hash_unbound")
    if (
        protocol is not None
        and protocol.ok
        and expected_protocol_hash is not None
        and protocol.protocol_hash != expected_protocol_hash
    ):
        reason_codes.append("domain_protocol_hash_mismatch")

    checks["domain_loss_enabled"] = adaptation_weight > 0
    if adaptation_weight <= 0:
        reason_codes.append("domain_loss_disabled")

    checks["source_disposition_runtime_ready"] = (
        source_disposition == "runtime_ready"
        or source_disposition == "not_required"
    )
    checks["target_disposition_runtime_ready"] = (
        target_disposition == "runtime_ready"
    )
    checks["strategy_assets_ready"] = _strategy_assets_ready(
        route,
        reason_codes,
        teacher_checkpoint=teacher_checkpoint,
        teacher_sha256=teacher_sha256,
        source_model_checkpoint=source_model_checkpoint,
        source_model_sha256=source_model_sha256,
    )
    checks["label_availability_match"] = (
        checks["target_label_availability_match"]
        and checks["source_label_availability_match"]
    )

    runtime_ready = (
        protocol is not None
        and protocol.ok
        and source_disposition in {"runtime_ready", "not_required"}
        and target_disposition == "runtime_ready"
        and checks["domain_protocol_hash_match"]
        and checks["domain_loss_enabled"]
        and checks["strategy_assets_ready"]
        and checks["label_availability_match"]
    )
    if runtime_ready:
        disposition: DomainRouteDisposition = "runtime_ready"
    else:
        disposition = _disposition_from_reasons(list(reason_codes))

    reason_codes = list(dict.fromkeys(reason_codes))
    return DomainPaperRouteReport(
        paper_id=route.paper_id,
        paper_route_fingerprint=route.execution_fingerprint,
        component_id=route.component_id,
        adapter_class=route.adapter_class,
        recipe_id=route.recipe_id,
        branch_id=route.branch_id,
        adaptation_mode=route.adaptation_mode,
        source_free=route.source_free,
        domain_pair_id=(
            protocol.pair.domain_pair_id
            if protocol is not None and protocol.ok and protocol.pair
            else (domain_pair_id or "")
        ),
        domain_protocol_hash=(
            protocol.protocol_hash
            if protocol is not None and protocol.ok
            else ""
        ),
        disposition=disposition,
        reason_codes=reason_codes,
        source_disposition=source_disposition,
        target_disposition=target_disposition,
        source_recovery_action=source_recovery,
        target_recovery_action=target_recovery,
        route_checks=checks,
        allows_asha=disposition == "runtime_ready",
    )


def _verify_checkpoint(
    name: str,
    checkpoint: str | None,
    expected_sha256: str | None,
    reason_codes: list[str],
) -> bool:
    """Verify one frozen strategy checkpoint file against its bound hash."""
    if not checkpoint or not expected_sha256:
        reason_codes.append(f"{name}_checkpoint_missing")
        return False
    path = Path(checkpoint)
    if not path.is_file():
        reason_codes.append(f"{name}_checkpoint_missing")
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        reason_codes.append(f"{name}_checkpoint_sha256_mismatch")
        return False
    return True


def _strategy_assets_ready(
    route: DomainPaperRoute,
    reason_codes: list[str],
    *,
    teacher_checkpoint: str | None,
    teacher_sha256: str | None,
    source_model_checkpoint: str | None,
    source_model_sha256: str | None,
) -> bool:
    """Check branch strategy assets; append mismatch reasons when blocked."""
    ready = True
    if route.branch_id in {"domain_distillation", "cross_domain_teacher"}:
        ready = _verify_checkpoint(
            "teacher",
            teacher_checkpoint,
            teacher_sha256,
            reason_codes,
        )
    if route.branch_id == "source_free_adaptation":
        ready = (
            ready
            and _verify_checkpoint(
                "source_model",
                source_model_checkpoint,
                source_model_sha256,
                reason_codes,
            )
        )
    return ready


def default_domain_paper_profile(paper_id: str) -> DomainPaperRoute:
    return default_domain_paper_route_registry().route(paper_id)


class DomainPaperRouteCertificationSummary(BaseModel, YAMLModelMixin):
    """Persistent coverage summary for one domain-route certification run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "domain_paper_route_certification.v1"
    papers_total: int
    runtime_ready: int
    evidence_recovery: int
    blocked_runtime: int
    silent_drops: list[str] = Field(default_factory=list)
    reports: list[DomainPaperRouteReport]
    summary_hash: str = ""

    @model_validator(mode="after")
    def bind_summary(self) -> "DomainPaperRouteCertificationSummary":
        if self.silent_drops:
            raise ValueError(
                f"domain route certification silent drops: {self.silent_drops}"
            )
        paper_ids = [item.paper_id for item in self.reports]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("domain route certification contains duplicate papers")
        if self.papers_total != len(paper_ids):
            raise ValueError("every paper must carry exactly one certification report")
        counts = (self.runtime_ready, self.evidence_recovery, self.blocked_runtime)
        if sum(counts) != self.papers_total:
            raise ValueError("domain route dispositions must cover every paper")
        expected = compute_domain_certification_summary_hash(self)
        if self.summary_hash and self.summary_hash != expected:
            raise ValueError("domain route certification summary hash mismatch")
        self.summary_hash = expected
        return self


def compute_domain_certification_summary_hash(
    summary: DomainPaperRouteCertificationSummary,
) -> str:
    payload = summary.model_dump(
        mode="json", exclude={"summary_hash", "schema_version"}
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_asset_records(
    asset_registry_path: Path | str | None,
) -> dict[str, PaperAssetRecord]:
    if asset_registry_path is None:
        return {}
    registry = PaperAssetRegistry.from_yaml(asset_registry_path)
    return {record.paper_id: record for record in registry.records}


def certify_all_domain_routes(
    *,
    output_path: Path | str,
    workspace: Path | str,
    paper_ids: tuple[str, ...] | None = None,
    source: str | None = None,
    target: str | None = None,
    source_sha256: str | None = None,
    target_sha256: str | None = None,
    source_dataset_hash: str | None = None,
    target_dataset_hash: str | None = None,
    source_domain_id: str = "source",
    target_domain_id: str = "target",
    domain_pair_id: str | None = None,
    expected_protocol_hash: str | None = None,
    adaptation_weight: float = 0.05,
    teacher_checkpoint: str | None = None,
    teacher_sha256: str | None = None,
    source_model_checkpoint: str | None = None,
    source_model_sha256: str | None = None,
    source_model_protocol_hash: str | None = None,
    asset_registry_path: Path | str | None = None,
) -> DomainPaperRouteCertificationSummary:
    """Certify every domain paper route and persist one coverage summary.

    When a paper asset registry is supplied, each route is certified
    against that paper's real source/target manifests, their bound
    per-file SHA-256 values, protocol hash, and strategy checkpoints
    instead of the caller defaults.
    """
    asset_records = _load_asset_records(asset_registry_path)
    registry = default_domain_paper_route_registry()
    if paper_ids is not None:
        ids = tuple(paper_ids)
    else:
        ids = tuple(item.paper_id for item in registry.routes())
    reports = []
    for paper_id in ids:
        record = asset_records.get(paper_id)
        paper_source = source
        paper_target = target
        paper_source_sha = source_sha256
        paper_target_sha = target_sha256
        paper_protocol_hash = expected_protocol_hash
        paper_teacher = teacher_checkpoint
        paper_teacher_sha = teacher_sha256
        paper_source_model = source_model_checkpoint
        paper_source_model_sha = source_model_sha256
        paper_source_model_protocol = source_model_protocol_hash
        if record is not None:
            if record.source_dataset_manifest is not None:
                paper_source = record.source_dataset_manifest
                paper_source_sha = record.asset_hashes.get(
                    "source_dataset_manifest"
                )
            if record.target_dataset_manifest is not None:
                paper_target = record.target_dataset_manifest
                paper_target_sha = record.asset_hashes.get(
                    "target_dataset_manifest"
                )
            if paper_protocol_hash is None:
                paper_protocol_hash = record.protocol_hash
            if record.teacher_checkpoint is not None:
                paper_teacher = record.teacher_checkpoint
                paper_teacher_sha = record.teacher_sha256
        reports.append(
            certify_domain_paper_route(
                paper_id,
                workspace=workspace,
                source=paper_source,
                target=paper_target,
                source_sha256=paper_source_sha,
                target_sha256=paper_target_sha,
                source_dataset_hash=source_dataset_hash,
                target_dataset_hash=target_dataset_hash,
                source_domain_id=source_domain_id,
                target_domain_id=target_domain_id,
                domain_pair_id=domain_pair_id,
                expected_protocol_hash=paper_protocol_hash,
                adaptation_weight=adaptation_weight,
                teacher_checkpoint=paper_teacher,
                teacher_sha256=paper_teacher_sha,
                source_model_checkpoint=paper_source_model,
                source_model_sha256=paper_source_model_sha,
                source_model_protocol_hash=paper_source_model_protocol,
            )
        )
    found = {item.paper_id for item in reports}
    summary = DomainPaperRouteCertificationSummary(
        papers_total=len(ids),
        runtime_ready=sum(item.disposition == "runtime_ready" for item in reports),
        evidence_recovery=sum(
            item.disposition == "evidence_recovery" for item in reports
        ),
        blocked_runtime=sum(
            item.disposition == "blocked_runtime" for item in reports
        ),
        silent_drops=[paper_id for paper_id in ids if paper_id not in found],
        reports=reports,
    )
    summary.to_yaml(output_path, exclude_none=True, sort_keys=False)
    return summary


def certify_domain_paper_routes(
    *,
    workspace: Path | str,
    paper_ids: tuple[str, ...] | None = None,
    source: str | None = None,
    target: str | None = None,
    source_sha256: str | None = None,
    target_sha256: str | None = None,
    source_dataset_hash: str | None = None,
    target_dataset_hash: str | None = None,
    source_domain_id: str = "source",
    target_domain_id: str = "target",
    domain_pair_id: str | None = None,
    expected_protocol_hash: str | None = None,
    adaptation_weight: float = 0.05,
) -> list[DomainPaperRouteReport]:
    """Certify every requested domain paper route without silent drops."""
    registry = default_domain_paper_route_registry()
    if paper_ids is not None:
        ids = tuple(paper_ids)
    else:
        ids = tuple(item.paper_id for item in registry.routes())
    return [
        certify_domain_paper_route(
            paper_id,
            workspace=workspace,
            source=source,
            target=target,
            source_sha256=source_sha256,
            target_sha256=target_sha256,
            source_dataset_hash=source_dataset_hash,
            target_dataset_hash=target_dataset_hash,
            source_domain_id=source_domain_id,
            target_domain_id=target_domain_id,
            domain_pair_id=domain_pair_id,
            expected_protocol_hash=expected_protocol_hash,
            adaptation_weight=adaptation_weight,
        )
        for paper_id in ids
    ]


__all__ = [
    "DomainPaperRouteCertificationSummary",
    "DomainPaperRouteReport",
    "DomainRouteDisposition",
    "certify_all_domain_routes",
    "certify_domain_paper_route",
    "certify_domain_paper_routes",
    "compute_domain_certification_summary_hash",
    "compute_domain_route_report_hash",
]
