"""Concise terminal rendering for paper auto-optimization acceptance."""

from __future__ import annotations

from yolo_agent.certification.paper_auto_optimization_schemas import (
    PaperAutoOptimizationReport,
)


def render_paper_auto_optimization_report(
    report: PaperAutoOptimizationReport,
) -> list[str]:
    """Show execution identity, paired evidence, and the exact stop reason."""
    lines = [
        "YOLO Agent Paper Auto-Optimization Acceptance",
        "---------------------------------------------",
        f"Status:    {report.status}",
        f"Papers:    {','.join(report.paper_ids) or '-'}",
        f"Components: {','.join(report.component_ids) or '-'}",
        f"Snapshot:  {report.research_snapshot_hash or '-'}",
        "Scalar HPO: disabled",
    ]
    certified = next(
        (item for item in report.stages if item.stage_id == "certified_adapter"),
        None,
    )
    tracks = certified.metrics.get("tracks", {}) if certified is not None else {}
    if isinstance(tracks, dict):
        for component_id, raw in tracks.items():
            item = raw if isinstance(raw, dict) else {}
            lines.append(
                "Track:     "
                f"component={component_id} "
                f"family={item.get('component_family', '-')} "
                f"adapter={item.get('adapter_hash', '-')} "
                f"maturity={item.get('maturity', '-')}"
            )
    for paired in report.paired_deltas:
        primary_delta = paired.metric_deltas.get(paired.primary_metric)
        error_delta = ",".join(
            f"{name}={_delta(value)}"
            for name, value in sorted(paired.target_error_fact_deltas.items())
        ) or "-"
        lines.extend(
            [
                f"{paired.stage_id} {paired.component_id}:",
                f"  family={paired.component_family} recipe={paired.recipe_id}",
                f"  baseline={paired.baseline_id or '-'}",
                f"  candidate={paired.candidate_id}",
                "  paired_delta "
                f"{paired.primary_metric}={_delta(primary_delta)}",
                f"  target_error_delta {error_delta}",
                "  guards "
                f"mAP50-95={_delta(paired.overall_map50_95_delta)} "
                f"latency_ms={_delta(paired.latency_delta_ms)} "
                f"model_size_mb={_delta(paired.model_size_delta_mb)}",
                "  decision="
                + (
                    "promoted"
                    if not paired.rejection_reasons
                    else "eliminated: " + ", ".join(paired.rejection_reasons)
                ),
            ]
        )
    if report.evidence_recovery_actions:
        lines.append(
            "Recovery:  " + ", ".join(report.evidence_recovery_actions)
        )
        lines.append("Training:  blocked until evidence recovery completes")
    elif report.failures:
        lines.append(f"Reason:    {report.failures[0]}")
    elif report.pilot_reproduced:
        lines.append(
            "Result:    pilot_reproduced="
            + ",".join(report.pilot_reproduced_component_ids)
        )
        lines.append(
            "Full:      blocked; use explicit --confirm-full-run for full and seeds 2/3"
        )
    return lines


def _delta(value: float | None) -> str:
    return "-" if value is None else f"{value:+.6f}"


__all__ = ["render_paper_auto_optimization_report"]
