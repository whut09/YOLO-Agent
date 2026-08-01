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
        f"Paper:     {','.join(report.paper_ids) or '-'}",
        f"Component: {report.component_id}",
        f"Adapter:   {report.adapter_hash or '-'}",
        f"Maturity:  {report.maturity or '-'}",
        f"Snapshot:  {report.research_snapshot_hash or '-'}",
        "Scalar HPO: disabled",
    ]
    for paired in report.paired_deltas:
        lines.extend(
            [
                f"{paired.stage_id}:",
                f"  baseline={paired.baseline_id or '-'}",
                f"  candidate={paired.candidate_id}",
                "  paired_delta "
                f"AP_small={_delta(paired.ap_small_delta)} "
                f"target_recall={_delta(paired.target_recall_delta)} "
                f"FN={_delta(paired.false_negative_delta)}",
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
        lines.append("Result:    pilot_reproduced")
        lines.append(
            "Full:      blocked; use explicit --confirm-full-run for full and seeds 2/3"
        )
    return lines


def _delta(value: float | None) -> str:
    return "-" if value is None else f"{value:+.6f}"


__all__ = ["render_paper_auto_optimization_report"]
