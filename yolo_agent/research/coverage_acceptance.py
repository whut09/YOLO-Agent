"""Traceable acceptance metrics for scalable paper implementation coverage."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import maturity_rank
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.executable_coverage_schemas import (
    ExecutablePaperCoverageBaseline,
)
from yolo_agent.research.method_profiles import PaperMethodCoverageReport


CoverageMetricId = Literal[
    "compatible_paper_method_profiles",
    "compatible_mechanism_reusable_adapters",
    "compatible_mechanism_runtime_integrated",
    "compatible_mechanism_smoke_passed",
    "compatible_papers_certified_adapter",
]


class CoverageAcceptanceThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compatible_paper_method_profiles: float = 0.85
    compatible_mechanism_reusable_adapters: float = 0.80
    compatible_mechanism_runtime_integrated: float = 0.70
    compatible_mechanism_smoke_passed: float = 0.60
    compatible_papers_certified_adapter: float = 0.70


class CoverageRatio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: CoverageMetricId
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    ratio: float = Field(ge=0.0, le=1.0)
    target: float = Field(ge=0.0, le=1.0)
    passed: bool
    numerator_ids: list[str] = Field(default_factory=list)
    denominator_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_membership(self) -> "CoverageRatio":
        if self.numerator_ids != sorted(set(self.numerator_ids)):
            raise ValueError("coverage numerator IDs must be sorted and unique")
        if self.denominator_ids != sorted(set(self.denominator_ids)):
            raise ValueError("coverage denominator IDs must be sorted and unique")
        if self.numerator != len(self.numerator_ids):
            raise ValueError("coverage numerator does not match numerator IDs")
        if self.denominator != len(self.denominator_ids):
            raise ValueError("coverage denominator does not match denominator IDs")
        expected = self.numerator / self.denominator if self.denominator else 0.0
        if abs(self.ratio - expected) > 1e-12:
            raise ValueError("coverage ratio does not match membership")
        if self.passed != (self.ratio >= self.target):
            raise ValueError("coverage pass state does not match target")
        return self


class AdapterArtifactTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_maturity: str
    artifact_type: str
    artifact_path: str
    artifact_sha256: str
    protocol_hash: str | None = None
    status: str
    mock: bool = False


class AdapterCoverageTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter_id: str
    effective_maturity: str
    runtime_integrated: bool
    smoke_passed: bool
    artifacts: list[AdapterArtifactTrace] = Field(default_factory=list)


class MechanismCoverageTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mechanism_id: str
    compatibility: str
    paper_ids: list[str] = Field(default_factory=list)
    reusable_adapter_ids: list[str] = Field(default_factory=list)
    runtime_integrated_adapter_ids: list[str] = Field(default_factory=list)
    smoke_passed_adapter_ids: list[str] = Field(default_factory=list)
    adapter_traces: list[AdapterCoverageTrace] = Field(default_factory=list)


class PaperCoverageTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    profile_id: str
    compatibility_class: str
    method_profile_valid: bool
    mechanism_ids: list[str] = Field(default_factory=list)
    reusable_adapter_ids: list[str] = Field(default_factory=list)
    certified_adapter_ids: list[str] = Field(default_factory=list)
    source_locations: list[str] = Field(default_factory=list)


class CoverageGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: CoverageMetricId
    target: float
    actual: float
    additional_required: int = Field(ge=0)
    missing_ids: list[str] = Field(default_factory=list)


class MechanismImplementationPriority(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mechanism_id: str
    covered_paper_count: int = Field(ge=0)
    paper_ids: list[str] = Field(default_factory=list)
    reusable_adapter_ids: list[str] = Field(default_factory=list)
    missing_runtime_integration: bool = False
    missing_smoke_certification: bool = False
    reason: str


class PaperCoverageAcceptanceReport(BaseModel, YAMLModelMixin):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_coverage_acceptance.v1"
    source_method_coverage_hash: str
    source_executable_coverage_hash: str
    source_registry_hash: str | None = None
    status: Literal["passed", "failed"]
    metrics: dict[CoverageMetricId, CoverageRatio]
    paper_traces: list[PaperCoverageTrace]
    mechanism_traces: list[MechanismCoverageTrace]
    adapter_traces: list[AdapterCoverageTrace]
    separate_detector_family_paper_ids: list[str] = Field(default_factory=list)
    insufficient_information_paper_ids: list[str] = Field(default_factory=list)
    exact_reproduction_paper_ids: list[str] = Field(default_factory=list)
    gaps: list[CoverageGap] = Field(default_factory=list)
    next_mechanisms: list[MechanismImplementationPriority] = Field(
        default_factory=list
    )
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "PaperCoverageAcceptanceReport":
        expected_status = (
            "passed" if all(item.passed for item in self.metrics.values()) else "failed"
        )
        if self.status != expected_status:
            raise ValueError("coverage acceptance status does not match metrics")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("paper coverage acceptance report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class PaperCoverageAcceptanceBuilder:
    """Build denominator-correct acceptance from method and local runtime evidence."""

    def __init__(
        self,
        *,
        effective_contracts: Mapping[str, ComponentContract],
        thresholds: CoverageAcceptanceThresholds | None = None,
    ) -> None:
        self.contracts = dict(effective_contracts)
        self.thresholds = thresholds or CoverageAcceptanceThresholds()

    def build(
        self,
        method_coverage: PaperMethodCoverageReport,
        executable_coverage: ExecutablePaperCoverageBaseline,
        *,
        source_method_coverage_hash: str,
        source_registry_hash: str | None = None,
    ) -> PaperCoverageAcceptanceReport:
        profiles = {item.paper_id: item for item in method_coverage.profiles}
        entries = {item.paper_id: item for item in executable_coverage.entries}
        compatible_papers = executable_coverage.denominators[
            "yolo26_compatible_papers"
        ].paper_ids
        compatible_mechanisms = sorted(
            item.canonical_component_id
            for item in method_coverage.compatible_mechanism_coverage.mechanisms
            if item.yolo26_compatibility in {"compatible", "adapter_required"}
        )
        mechanism_papers = {
            item.canonical_component_id: sorted(set(item.paper_ids))
            for item in method_coverage.compatible_mechanism_coverage.mechanisms
            if item.canonical_component_id in compatible_mechanisms
        }
        mechanism_adapters: dict[str, set[str]] = defaultdict(set)
        for decision in method_coverage.decisions:
            for mapping in decision.mechanism_mappings:
                adapter_ids = list(
                    getattr(mapping, "reusable_adapter_ids", []) or []
                )
                if mapping.reusable_adapter_id:
                    adapter_ids.append(mapping.reusable_adapter_id)
                mechanism_adapters[mapping.canonical_component_id].update(adapter_ids)
        adapter_traces = {
            adapter_id: _adapter_trace(adapter_id, self.contracts.get(adapter_id))
            for adapter_id in sorted(
                {
                    adapter_id
                    for values in mechanism_adapters.values()
                    for adapter_id in values
                }
            )
        }
        mechanism_traces = [
            _mechanism_trace(
                mechanism_id,
                mechanism_papers.get(mechanism_id, []),
                sorted(mechanism_adapters.get(mechanism_id, set())),
                adapter_traces,
                method_coverage,
            )
            for mechanism_id in compatible_mechanisms
        ]
        smoke_ids = {
            adapter_id for adapter_id, trace in adapter_traces.items() if trace.smoke_passed
        }
        paper_traces = []
        for paper_id in sorted(entries):
            profile = profiles[paper_id]
            entry = entries[paper_id]
            certified = sorted(set(entry.reusable_adapter_candidates) & smoke_ids)
            paper_traces.append(
                PaperCoverageTrace(
                    paper_id=paper_id,
                    profile_id=profile.profile_id,
                    compatibility_class=entry.compatibility_class,
                    method_profile_valid=bool(
                        profile.source_locations and profile.canonical_component_ids
                    ),
                    mechanism_ids=entry.canonical_mechanisms,
                    reusable_adapter_ids=entry.reusable_adapter_candidates,
                    certified_adapter_ids=certified,
                    source_locations=profile.source_locations,
                )
            )
        compatible_trace = {
            item.paper_id: item
            for item in paper_traces
            if item.paper_id in compatible_papers
        }
        metric_members = {
            "compatible_paper_method_profiles": (
                [
                    paper_id
                    for paper_id in compatible_papers
                    if compatible_trace[paper_id].method_profile_valid
                ],
                compatible_papers,
            ),
            "compatible_mechanism_reusable_adapters": (
                [item.mechanism_id for item in mechanism_traces if item.reusable_adapter_ids],
                compatible_mechanisms,
            ),
            "compatible_mechanism_runtime_integrated": (
                [
                    item.mechanism_id
                    for item in mechanism_traces
                    if item.runtime_integrated_adapter_ids
                ],
                compatible_mechanisms,
            ),
            "compatible_mechanism_smoke_passed": (
                [item.mechanism_id for item in mechanism_traces if item.smoke_passed_adapter_ids],
                compatible_mechanisms,
            ),
            "compatible_papers_certified_adapter": (
                [
                    paper_id
                    for paper_id in compatible_papers
                    if compatible_trace[paper_id].certified_adapter_ids
                ],
                compatible_papers,
            ),
        }
        metrics = {
            metric_id: _ratio(
                metric_id,  # type: ignore[arg-type]
                numerator_ids,
                denominator_ids,
                getattr(self.thresholds, metric_id),
            )
            for metric_id, (numerator_ids, denominator_ids) in metric_members.items()
        }
        gaps = [_gap(item) for item in metrics.values() if not item.passed]
        next_mechanisms = sorted(
            (
                MechanismImplementationPriority(
                    mechanism_id=item.mechanism_id,
                    covered_paper_count=len(item.paper_ids),
                    paper_ids=item.paper_ids,
                    reusable_adapter_ids=item.reusable_adapter_ids,
                    missing_runtime_integration=not bool(
                        item.runtime_integrated_adapter_ids
                    ),
                    missing_smoke_certification=not bool(item.smoke_passed_adapter_ids),
                    reason=_priority_reason(item),
                )
                for item in mechanism_traces
                if not item.smoke_passed_adapter_ids
            ),
            key=lambda item: (-item.covered_paper_count, item.mechanism_id),
        )
        status = "passed" if all(item.passed for item in metrics.values()) else "failed"
        return PaperCoverageAcceptanceReport(
            source_method_coverage_hash=source_method_coverage_hash,
            source_executable_coverage_hash=executable_coverage.report_hash,
            source_registry_hash=source_registry_hash,
            status=status,
            metrics=metrics,  # type: ignore[arg-type]
            paper_traces=paper_traces,
            mechanism_traces=mechanism_traces,
            adapter_traces=list(adapter_traces.values()),
            separate_detector_family_paper_ids=[
                item.paper_id
                for item in paper_traces
                if item.compatibility_class == "separate_detector_family"
            ],
            insufficient_information_paper_ids=[
                item.paper_id
                for item in paper_traces
                if item.compatibility_class == "insufficient_information"
            ],
            exact_reproduction_paper_ids=executable_coverage.denominators[
                "exact_reproduction_candidates"
            ].paper_ids,
            gaps=gaps,
            next_mechanisms=next_mechanisms,
        )


def render_coverage_acceptance_markdown(
    report: PaperCoverageAcceptanceReport,
) -> str:
    """Render a compact human audit while YAML retains complete trace membership."""
    lines = [
        "# Paper Implementation Coverage Acceptance",
        "",
        f"Status: **{report.status}**",
        "",
        "## Acceptance Metrics",
        "",
        "| Metric | Result | Target | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for metric in report.metrics.values():
        lines.append(
            f"| `{metric.metric_id}` | {metric.numerator}/{metric.denominator} "
            f"({metric.ratio:.1%}) | >={metric.target:.0%} | "
            f"{'passed' if metric.passed else 'failed'} |"
        )
    lines.extend(
        [
            "",
            "## Independent Categories",
            "",
            f"- All paper traces: {len(report.paper_traces)}",
            "- Exact reproduction candidates: "
            f"{len(report.exact_reproduction_paper_ids)}",
            "- Separate detector family: "
            f"{len(report.separate_detector_family_paper_ids)}",
            "- Insufficient information: "
            f"{len(report.insufficient_information_paper_ids)}",
            "",
            "Exact reproduction is not inferred from component adaptation.",
            "",
            "## Residual Mechanisms",
            "",
            "These mechanisms remain useful follow-up work even when aggregate "
            "acceptance thresholds pass.",
            "",
            "| Mechanism | Papers | Adapter | Remaining work |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for item in report.next_mechanisms:
        adapters = ", ".join(f"`{value}`" for value in item.reusable_adapter_ids)
        lines.append(
            f"| `{item.mechanism_id}` | {item.covered_paper_count} | "
            f"{adapters or '-'} | {item.reason} |"
        )
    if not report.next_mechanisms:
        lines.append("| - | 0 | - | none |")
    lines.extend(
        [
            "",
            "## Traceability",
            "",
            f"- Method coverage SHA-256: `{report.source_method_coverage_hash}`",
            f"- Executable coverage hash: `{report.source_executable_coverage_hash}`",
            f"- Maturity registry SHA-256: `{report.source_registry_hash or 'none'}`",
            f"- Acceptance report hash: `{report.report_hash}`",
            "",
            "The adjacent YAML artifact contains every numerator and denominator "
            "ID plus paper, mechanism, adapter, protocol, artifact path, and "
            "artifact SHA-256 trace.",
            "",
        ]
    )
    return "\n".join(lines)


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _adapter_trace(
    adapter_id: str,
    contract: ComponentContract | None,
) -> AdapterCoverageTrace:
    if contract is None:
        return AdapterCoverageTrace(
            adapter_id=adapter_id,
            effective_maturity="metadata_only",
            runtime_integrated=False,
            smoke_passed=False,
        )
    return AdapterCoverageTrace(
        adapter_id=adapter_id,
        effective_maturity=contract.maturity,
        runtime_integrated=maturity_rank(contract.maturity)
        >= maturity_rank("runtime_integrated"),
        smoke_passed=maturity_rank(contract.maturity) >= maturity_rank("smoke_passed"),
        artifacts=[
            AdapterArtifactTrace(
                target_maturity=item.target_maturity,
                artifact_type=item.artifact_type,
                artifact_path=item.artifact_path.as_posix(),
                artifact_sha256=item.artifact_sha256,
                protocol_hash=item.protocol_hash,
                status=item.status,
                mock=item.mock,
            )
            for item in contract.maturity_artifacts
        ],
    )


def _mechanism_trace(
    mechanism_id: str,
    paper_ids: list[str],
    adapter_ids: list[str],
    adapter_traces: Mapping[str, AdapterCoverageTrace],
    method_coverage: PaperMethodCoverageReport,
) -> MechanismCoverageTrace:
    mechanism = next(
        item
        for item in method_coverage.compatible_mechanism_coverage.mechanisms
        if item.canonical_component_id == mechanism_id
    )
    traces = [adapter_traces[item] for item in adapter_ids]
    return MechanismCoverageTrace(
        mechanism_id=mechanism_id,
        compatibility=mechanism.yolo26_compatibility,
        paper_ids=paper_ids,
        reusable_adapter_ids=adapter_ids,
        runtime_integrated_adapter_ids=[
            item.adapter_id for item in traces if item.runtime_integrated
        ],
        smoke_passed_adapter_ids=[item.adapter_id for item in traces if item.smoke_passed],
        adapter_traces=traces,
    )


def _ratio(
    metric_id: CoverageMetricId,
    numerator_ids: list[str],
    denominator_ids: list[str],
    target: float,
) -> CoverageRatio:
    numerator = sorted(set(numerator_ids))
    denominator = sorted(set(denominator_ids))
    ratio = len(numerator) / len(denominator) if denominator else 0.0
    return CoverageRatio(
        metric_id=metric_id,
        numerator=len(numerator),
        denominator=len(denominator),
        ratio=ratio,
        target=target,
        passed=ratio >= target,
        numerator_ids=numerator,
        denominator_ids=denominator,
    )


def _gap(metric: CoverageRatio) -> CoverageGap:
    required = math.ceil(metric.target * metric.denominator)
    return CoverageGap(
        metric_id=metric.metric_id,
        target=metric.target,
        actual=metric.ratio,
        additional_required=max(0, required - metric.numerator),
        missing_ids=sorted(set(metric.denominator_ids) - set(metric.numerator_ids)),
    )


def _priority_reason(item: MechanismCoverageTrace) -> str:
    if not item.reusable_adapter_ids:
        return "no reusable adapter mapping; implement or map an evidence-equivalent adapter"
    if not item.runtime_integrated_adapter_ids:
        return "adapter exists but lacks valid runtime-integrated artifact identity"
    return "runtime is integrated but artifact-backed smoke certification is missing"


__all__ = [
    "CoverageAcceptanceThresholds",
    "PaperCoverageAcceptanceBuilder",
    "PaperCoverageAcceptanceReport",
    "file_sha256",
    "render_coverage_acceptance_markdown",
]
