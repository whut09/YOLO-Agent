"""Human-readable summary for reusable paper mechanism clustering."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.research.mechanism_clusters import PaperMechanismClusterReport


def render_mechanism_cluster_markdown(
    report: PaperMechanismClusterReport,
) -> str:
    lines = [
        "# Paper Mechanism Cluster Report",
        "",
        f"- Papers: {report.paper_count}",
        f"- Matched papers: {report.matched_paper_count}",
        f"- Unresolved papers: {report.unresolved_paper_count}",
        f"- Semantic conflicts: {len(report.conflicts)}",
        f"- Report hash: `{report.report_hash}`",
        "",
        "Mechanism mapping is not runtime implementation or reproduction evidence.",
        "",
        "## Reusable Clusters",
        "",
        "| Cluster | Adapter family | Papers | Adapter status |",
        "|---|---|---:|---|",
    ]
    for cluster in report.clusters:
        status = (
            "runtime_ready"
            if cluster.runtime_ready
            else "adapter_available"
            if cluster.adapter_available
            else "adapter_required"
        )
        lines.append(
            f"| `{cluster.cluster_id}` | `{cluster.adapter_family}` | "
            f"{cluster.paper_count} | {status} |"
        )
    lines.extend(["", "## Adapter Coverage Queue", ""])
    actionable = [
        item
        for item in report.implementation_opportunities
        if item.implementation_status == "adapter_required"
    ]
    if not actionable:
        lines.append("No compatible adapter implementation opportunity is currently ranked.")
    for item in actionable:
        hooks = ", ".join(f"`{hook}`" for hook in item.runtime_hooks) or "none"
        lines.extend([
            f"### {item.rank}. `{item.cluster_id}`",
            "",
            f"Covers {item.paper_count} papers through `{item.adapter_family}`. "
            f"Required runtime hooks: {hooks}.",
            "",
        ])
    lines.extend(["## Merge Conflicts", ""])
    if not report.conflicts:
        lines.append("No semantic merge conflicts were detected.")
    else:
        for conflict in report.conflicts:
            clusters = ", ".join(f"`{item}`" for item in conflict.candidate_cluster_ids)
            lines.append(
                f"- `{conflict.paper_id}`: {conflict.reason}; candidates: {clusters}"
            )
    lines.append("")
    return "\n".join(lines)


def write_mechanism_cluster_markdown(
    report: PaperMechanismClusterReport,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_mechanism_cluster_markdown(report), encoding="utf-8")
    return target


__all__ = [
    "render_mechanism_cluster_markdown",
    "write_mechanism_cluster_markdown",
]
