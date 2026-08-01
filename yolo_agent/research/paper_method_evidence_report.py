"""Coverage and field-gap reporting for offline paper method evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)


class PaperMethodEvidenceAudit(BaseModel):
    """Field-level evidence status for one paper."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    decision: str
    authorizes_method_profile: bool = False
    method_families: list[str] = Field(default_factory=list)
    canonical_mechanisms: list[str] = Field(default_factory=list)
    insertion_points: list[str] = Field(default_factory=list)
    changed_variables: list[str] = Field(default_factory=list)
    detector_families: list[str] = Field(default_factory=list)
    component_types: list[str] = Field(default_factory=list)
    required_runtime_hooks: list[str] = Field(default_factory=list)
    source_locations: list[str] = Field(default_factory=list)
    authorizing_source_locations: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    insufficiency_reasons: list[str] = Field(default_factory=list)


class PaperMethodEvidenceCoverageReport(BaseModel, YAMLModelMixin):
    """Deterministic aggregate of extraction coverage and decision movement."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_method_evidence_coverage.v1"
    paper_count: int = Field(default=0, ge=0)
    audited_paper_count: int = Field(default=0, ge=0)
    authorizing_profile_count: int = Field(default=0, ge=0)
    prior_only_profile_count: int = Field(default=0, ge=0)
    insufficient_information_count: int = Field(default=0, ge=0)
    baseline_insufficient_information_count: int | None = Field(default=None, ge=0)
    insufficient_information_delta: int | None = None
    converted_from_insufficient_count: int = Field(default=0, ge=0)
    converted_from_insufficient_paper_ids: list[str] = Field(default_factory=list)
    evidence_source_counts: dict[str, int] = Field(default_factory=dict)
    extracted_field_counts: dict[str, int] = Field(default_factory=dict)
    missing_field_counts: dict[str, int] = Field(default_factory=dict)
    decision_counts: dict[str, int] = Field(default_factory=dict)
    audits: list[PaperMethodEvidenceAudit] = Field(default_factory=list)
    report_hash: str = ""

    def with_hash(self) -> "PaperMethodEvidenceCoverageReport":
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self.model_copy(update={"report_hash": digest})


def build_method_evidence_coverage_report(
    current: PaperMethodCoverageReport,
    *,
    previous: PaperMethodCoverageReport | None = None,
) -> PaperMethodEvidenceCoverageReport:
    """Audit every profile and compare only against an explicit prior report."""
    decisions = {item.paper_id: item for item in current.decisions}
    audits = [
        _audit_profile(profile, decisions[profile.paper_id])
        for profile in sorted(current.profiles, key=lambda item: item.paper_id)
    ]
    previous_decisions = (
        {item.paper_id: item.decision for item in previous.decisions}
        if previous is not None
        else {}
    )
    converted = sorted(
        audit.paper_id
        for audit in audits
        if previous_decisions.get(audit.paper_id) == "insufficient_information"
        and audit.decision != "insufficient_information"
    )
    source_counts = Counter(
        observation.source
        for profile in current.profiles
        if profile.structured_method_evidence is not None
        for observation in profile.structured_method_evidence.observations
    )
    field_counts = Counter(
        observation.field_name
        for profile in current.profiles
        if profile.structured_method_evidence is not None
        for observation in profile.structured_method_evidence.observations
    )
    missing_counts = Counter(field for audit in audits for field in audit.missing_fields)
    current_insufficient = current.decision_counts.get("insufficient_information", 0)
    baseline_insufficient = (
        previous.decision_counts.get("insufficient_information", 0)
        if previous is not None
        else None
    )
    return PaperMethodEvidenceCoverageReport(
        paper_count=current.paper_count,
        audited_paper_count=len(audits),
        authorizing_profile_count=sum(audit.authorizes_method_profile for audit in audits),
        prior_only_profile_count=sum(
            bool(audit.source_locations) and not audit.authorizes_method_profile
            for audit in audits
        ),
        insufficient_information_count=current_insufficient,
        baseline_insufficient_information_count=baseline_insufficient,
        insufficient_information_delta=(
            current_insufficient - baseline_insufficient
            if baseline_insufficient is not None
            else None
        ),
        converted_from_insufficient_count=len(converted),
        converted_from_insufficient_paper_ids=converted,
        evidence_source_counts=dict(sorted(source_counts.items())),
        extracted_field_counts=dict(sorted(field_counts.items())),
        missing_field_counts=dict(sorted(missing_counts.items())),
        decision_counts=dict(sorted(current.decision_counts.items())),
        audits=audits,
    ).with_hash()


def write_method_evidence_coverage_markdown(
    report: PaperMethodEvidenceCoverageReport,
    path: Path | str,
) -> Path:
    """Write a compact human report; YAML retains all per-paper details."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    delta = (
        f"{report.insufficient_information_delta:+d}"
        if report.insufficient_information_delta is not None
        else "not_available"
    )
    lines = [
        "# Offline Paper Method Evidence Coverage",
        "",
        f"- Papers audited: {report.audited_paper_count}",
        f"- Authorizing method profiles: {report.authorizing_profile_count}",
        f"- Prior-only profiles: {report.prior_only_profile_count}",
        f"- Insufficient information: {report.insufficient_information_count}",
        f"- Explicit baseline insufficient: {report.baseline_insufficient_information_count}",
        f"- Insufficient delta: {delta}",
        f"- Converted with explicit local evidence: {report.converted_from_insufficient_count}",
        "",
        "MethodProfile evidence does not imply an implemented adapter, smoke evidence,",
        "runtime readiness, or permission to enqueue training.",
        "",
        "## Missing Fields",
        "",
    ]
    lines.extend(
        f"- `{field}`: {count}"
        for field, count in sorted(report.missing_field_counts.items())
    )
    lines.extend(["", "## Converted Papers", ""])
    lines.extend(
        f"- `{paper_id}`"
        for paper_id in report.converted_from_insufficient_paper_ids
    )
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target


def _audit_profile(
    profile: PaperMethodProfile,
    decision: PaperImplementationDecision,
) -> PaperMethodEvidenceAudit:
    evidence = profile.structured_method_evidence
    if evidence is None:
        return PaperMethodEvidenceAudit(
            paper_id=profile.paper_id,
            decision=decision.decision,
            missing_fields=["structured_method_evidence"],
            insufficiency_reasons=list(decision.reasons),
        )
    missing: list[str] = []
    if not evidence.method_families and not evidence.canonical_mechanisms:
        missing.append("method_family_or_canonical_mechanism")
    if not evidence.insertion_points:
        missing.append("insertion_points")
    if not evidence.changed_variables:
        missing.append("changed_variables")
    if not evidence.component_types:
        missing.append("component_types")
    if not evidence.required_runtime_hooks:
        missing.append("required_runtime_hooks")
    return PaperMethodEvidenceAudit(
        paper_id=profile.paper_id,
        decision=decision.decision,
        authorizes_method_profile=evidence.authorizes_method_profile,
        method_families=evidence.method_families,
        canonical_mechanisms=evidence.canonical_mechanisms,
        insertion_points=evidence.insertion_points,
        changed_variables=evidence.changed_variables,
        detector_families=evidence.detector_families,
        component_types=list(evidence.component_types),
        required_runtime_hooks=evidence.required_runtime_hooks,
        source_locations=sorted({
            item.source_location for item in evidence.observations
        }),
        authorizing_source_locations=sorted({
            item.source_location
            for item in evidence.observations
            if item.authorizes_method_profile
        }),
        missing_fields=missing,
        insufficiency_reasons=(
            list(decision.reasons)
            if decision.decision == "insufficient_information"
            else []
        ),
    )


__all__ = [
    "PaperMethodEvidenceAudit",
    "PaperMethodEvidenceCoverageReport",
    "build_method_evidence_coverage_report",
    "write_method_evidence_coverage_markdown",
]
