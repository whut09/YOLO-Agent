"""Aggregate paper mechanism clusters and rank adapter coverage opportunities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from yolo_agent.research.mechanism_clusters import (
    AdapterCoverageOpportunity,
    MechanismClusterConfig,
    MechanismClusterConflict,
    MechanismClusterSummary,
    PaperMechanismClusterMatch,
    PaperMechanismClusterReport,
)
from yolo_agent.research.method_profiles import PaperMethodCoverageReport


def build_mechanism_cluster_report(
    coverage: PaperMethodCoverageReport,
    *,
    config: MechanismClusterConfig,
    matches: list[PaperMechanismClusterMatch],
    conflicts: list[MechanismClusterConflict],
) -> PaperMechanismClusterReport:
    """Build a deterministic report without changing adapter maturity."""
    profiles = {item.paper_id: item for item in coverage.profiles}
    definitions = {item.cluster_id: item for item in config.clusters}
    grouped: dict[str, list[PaperMechanismClusterMatch]] = defaultdict(list)
    for match in matches:
        if match.cluster_id is not None:
            grouped[match.cluster_id].append(match)
    summaries: list[MechanismClusterSummary] = []
    for cluster_id, cluster_matches in sorted(grouped.items()):
        definition = definitions[cluster_id]
        paper_ids = sorted({item.paper_id for item in cluster_matches})
        parameter_values: dict[str, set[str]] = defaultdict(set)
        limitations: set[str] = set()
        locations: set[str] = set()
        for paper_id in paper_ids:
            profile = profiles[paper_id]
            for key, values in _parameter_surface(profile).items():
                parameter_values[key].update(values)
            limitations.update(profile.limitations)
            locations.update(profile.source_locations)
        summaries.append(MechanismClusterSummary(
            cluster_id=cluster_id,
            adapter_family=definition.adapter_family,
            training_semantic=definition.training_semantic,
            paper_ids=paper_ids,
            paper_count=len(paper_ids),
            parameter_differences={
                key: sorted(values)
                for key, values in sorted(parameter_values.items())
            },
            limitations=sorted(limitations),
            source_locations=sorted(locations),
            adapter_available=any(item.adapter_available for item in cluster_matches),
            runtime_ready=any(item.runtime_ready for item in cluster_matches),
        ))
    opportunities = _rank_opportunities(summaries, definitions)
    matched_papers = {item.paper_id for item in matches if item.cluster_id is not None}
    unresolved_papers = {item.paper_id for item in matches if item.cluster_id is None}
    return PaperMechanismClusterReport(
        paper_count=coverage.paper_count,
        matched_paper_count=len(matched_papers),
        unresolved_paper_count=len(unresolved_papers),
        matches=sorted(
            matches,
            key=lambda item: (item.paper_id, item.cluster_id or "~"),
        ),
        clusters=summaries,
        conflicts=sorted(
            conflicts,
            key=lambda item: (item.paper_id, item.reason),
        ),
        implementation_opportunities=opportunities,
    ).with_hash()


def _parameter_surface(profile: Any) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    parameters = profile.paper_parameters
    protocol = profile.protocol_constraints
    for value in parameters.get("changed_variables", []) or []:
        result["changed_variables"].add(str(value))
    for key in (
        "training_costs",
        "inference_costs",
    ):
        for value in parameters.get(key, []) or []:
            result[key].add(str(value))
    for key in (
        "insertion_points",
        "detector_families",
        "required_runtime_hooks",
        "compatibility_constraints",
    ):
        for value in protocol.get(key, []) or []:
            result[key].add(str(value))
    for key in ("training_only", "inference_changed"):
        value = protocol.get(key)
        if value is not None:
            result[key].add(str(value).lower())
    return result


def _rank_opportunities(
    summaries: list[MechanismClusterSummary],
    definitions: dict[str, Any],
) -> list[AdapterCoverageOpportunity]:
    status_type = Literal[
        "adapter_available",
        "runtime_ready",
        "adapter_required",
        "separate_detector_family",
        "incompatible",
    ]
    ranked: list[
        tuple[float, MechanismClusterSummary, Any, status_type, list[str]]
    ] = []
    for summary in summaries:
        definition = definitions[summary.cluster_id]
        if summary.runtime_ready:
            status = "runtime_ready"
            reasons = ["runtime_ready_adapter_already_covers_cluster"]
        elif summary.adapter_available:
            status = "adapter_available"
            reasons = ["adapter_exists_but_runtime_readiness_is_incomplete"]
        elif definition.yolo26_compatibility == "separate_detector_family":
            status = "separate_detector_family"
            reasons = ["requires_separate_detector_family_track"]
        elif definition.yolo26_compatibility == "incompatible":
            status = "incompatible"
            reasons = ["yolo26_runtime_semantics_are_incompatible"]
        else:
            status = "adapter_required"
            reasons = [
                "one_adapter_family_can_cover_multiple_source_papers",
                "implementation_does_not_imply_smoke_or_pilot_maturity",
            ]
        coverage_score = float(summary.paper_count * 100)
        hook_penalty = float(len(definition.required_runtime_hooks) * 2)
        compatibility_penalty = (
            1000.0
            if status in {"separate_detector_family", "incompatible"}
            else 500.0
            if status in {"runtime_ready", "adapter_available"}
            else 0.0
        )
        score = coverage_score - hook_penalty - compatibility_penalty
        ranked.append((score, summary, definition, status, reasons))
    ranked.sort(
        key=lambda item: (-item[0], -item[1].paper_count, item[1].cluster_id)
    )
    return [
        AdapterCoverageOpportunity(
            rank=index,
            cluster_id=summary.cluster_id,
            adapter_family=summary.adapter_family,
            paper_ids=summary.paper_ids,
            paper_count=summary.paper_count,
            runtime_hooks=definition.required_runtime_hooks,
            implementation_status=status,
            score=score,
            reasons=reasons,
        )
        for index, (score, summary, definition, status, reasons) in enumerate(
            ranked,
            start=1,
        )
    ]


__all__ = ["build_mechanism_cluster_report"]
