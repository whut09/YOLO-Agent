"""Cross-run comparison reports for loop harness runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.pareto import ParetoFront, ParetoSelector, candidate_metrics_from_row
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import Evidence
from yolo_agent.reports.experiment_report import _candidate_rows, _experiment_nodes, _load_context


class RunComparisonSnapshot(BaseModel):
    """Comparable snapshot for one run."""

    run_id: str
    run_dir: Path
    dataset_version: str = "unknown"
    dataset_manifest_sha256: str = "unknown"
    candidate_rows: list[dict[str, Any]] = Field(default_factory=list)
    pareto_front: ParetoFront = Field(default_factory=ParetoFront)
    best_candidate: dict[str, Any] | None = None


class CrossRunComparisonReport(BaseModel):
    """Structured cross-run comparison."""

    runs: list[RunComparisonSnapshot]
    dataset_version_consistent: bool
    manifest_sha_consistent: bool

    def to_markdown(self) -> str:
        """Render the comparison as Markdown."""
        lines = [
            "# YOLO Agent Cross-Run Comparison",
            "",
            "## Dataset Consistency",
            "",
            _dataset_table(self.runs),
            "",
            f"- Dataset version consistent: `{self.dataset_version_consistent}`",
            f"- Dataset manifest SHA consistent: `{self.manifest_sha_consistent}`",
            "",
            "## Best Candidate Metrics",
            "",
            _best_metrics_table(self.runs),
            "",
            "## Metric Delta",
            "",
            _metric_delta_table(self.runs),
            "",
            "## Pareto Front Changes",
            "",
            _pareto_changes_text(self.runs),
            "",
            "## Possible Positive Contributions",
            "",
            _positive_contributions_text(self.runs),
            "",
        ]
        return "\n".join(lines)


def generate_cross_run_comparison_report(run_paths: list[Path | str], out_path: Path | str) -> str:
    """Generate a Markdown comparison report for multiple run directories."""
    report = build_cross_run_comparison(run_paths)
    markdown = report.to_markdown()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    return markdown


def build_cross_run_comparison(run_paths: list[Path | str]) -> CrossRunComparisonReport:
    """Build a structured cross-run comparison."""
    if len(run_paths) < 2:
        raise ValueError("At least two runs are required for comparison.")
    snapshots = [_load_snapshot(Path(path)) for path in run_paths]
    versions = {snapshot.dataset_version for snapshot in snapshots}
    shas = {snapshot.dataset_manifest_sha256 for snapshot in snapshots}
    return CrossRunComparisonReport(
        runs=snapshots,
        dataset_version_consistent=len(versions) == 1,
        manifest_sha_consistent=len(shas) == 1,
    )


def _load_snapshot(run_dir: Path) -> RunComparisonSnapshot:
    evidence = EvidenceStore(run_dir.parent).load_run(run_dir.name)
    context = _load_context(run_dir, evidence)
    rows = _candidate_rows(evidence, context, _experiment_nodes(context))
    pareto_front = ParetoSelector().select(
        [metrics for row in rows if (metrics := candidate_metrics_from_row(row)) is not None]
    )
    run_context = _read_run_context(run_dir)
    return RunComparisonSnapshot(
        run_id=run_dir.name,
        run_dir=run_dir,
        dataset_version=str(run_context.get("dataset_version", "unknown")),
        dataset_manifest_sha256=str(run_context.get("dataset_manifest_sha256", "unknown")),
        candidate_rows=rows,
        pareto_front=pareto_front,
        best_candidate=_best_candidate(rows),
    )


def _read_run_context(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_context.yaml"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def _best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    evidence_rows = [row for row in rows if row.get("has_evidence")]
    if not evidence_rows:
        return None
    return max(evidence_rows, key=_best_score)


def _best_score(row: dict[str, Any]) -> tuple[float, float]:
    metrics = row.get("metrics", {})
    accuracy = _number(metrics, "map50", "mAP", "map", "map50_95", "recall")
    latency = _number(metrics, "latency_ms", "latency")
    return (accuracy if accuracy is not None else float("-inf"), -(latency if latency is not None else float("inf")))


def _dataset_table(runs: list[RunComparisonSnapshot]) -> str:
    lines = ["| run | dataset version | manifest sha |", "|---|---|---|"]
    for snapshot in runs:
        lines.append(
            f"| {snapshot.run_id} | {snapshot.dataset_version} | {_short_sha(snapshot.dataset_manifest_sha256)} |"
        )
    return "\n".join(lines)


def _best_metrics_table(runs: list[RunComparisonSnapshot]) -> str:
    lines = ["| run | best candidate | map50 | recall | latency_ms | action |", "|---|---|---:|---:|---:|---|"]
    for snapshot in runs:
        row = snapshot.best_candidate
        if row is None:
            lines.append(f"| {snapshot.run_id} | unknown | unknown | unknown | unknown | unknown |")
            continue
        metrics = row.get("metrics", {})
        lines.append(
            "| {run} | {candidate} | {map50} | {recall} | {latency} | {action} |".format(
                run=snapshot.run_id,
                candidate=row.get("id", "unknown"),
                map50=_metric_text(metrics, "map50", "mAP", "map", "map50_95"),
                recall=_metric_text(metrics, "recall"),
                latency=_metric_text(metrics, "latency_ms", "latency"),
                action=_action_text(row),
            )
        )
    return "\n".join(lines)


def _metric_delta_table(runs: list[RunComparisonSnapshot]) -> str:
    lines = ["| transition | delta map50 | delta recall | delta latency_ms |", "|---|---:|---:|---:|"]
    for previous, current in zip(runs, runs[1:]):
        previous_metrics = _best_metrics(previous)
        current_metrics = _best_metrics(current)
        lines.append(
            "| {transition} | {map_delta} | {recall_delta} | {latency_delta} |".format(
                transition=f"{previous.run_id} -> {current.run_id}",
                map_delta=_delta_text(current_metrics, previous_metrics, "map50", "mAP", "map", "map50_95"),
                recall_delta=_delta_text(current_metrics, previous_metrics, "recall"),
                latency_delta=_delta_text(current_metrics, previous_metrics, "latency_ms", "latency"),
            )
        )
    return "\n".join(lines) if len(lines) > 2 else "- Need at least two evidence-backed runs."


def _pareto_changes_text(runs: list[RunComparisonSnapshot]) -> str:
    lines: list[str] = []
    for previous, current in zip(runs, runs[1:]):
        previous_ids = {point.candidate_id for point in previous.pareto_front.points}
        current_ids = {point.candidate_id for point in current.pareto_front.points}
        added = sorted(current_ids - previous_ids)
        removed = sorted(previous_ids - current_ids)
        kept = sorted(previous_ids & current_ids)
        lines.append(f"- `{previous.run_id}` -> `{current.run_id}`")
        lines.append(f"  - Added: {', '.join(added) if added else 'none'}")
        lines.append(f"  - Removed: {', '.join(removed) if removed else 'none'}")
        lines.append(f"  - Kept: {', '.join(kept) if kept else 'none'}")
    return "\n".join(lines) if lines else "- Pareto front change is unknown."


def _positive_contributions_text(runs: list[RunComparisonSnapshot]) -> str:
    lines: list[str] = []
    for previous, current in zip(runs, runs[1:]):
        previous_metrics = _best_metrics(previous)
        current_metrics = _best_metrics(current)
        improved = _improved_metrics(previous_metrics, current_metrics)
        if not improved:
            continue
        row = current.best_candidate or {}
        lines.append(
            "- `{transition}`: `{action}` may have contributed positively to {metrics}.".format(
                transition=f"{previous.run_id} -> {current.run_id}",
                action=_action_text(row),
                metrics=", ".join(improved),
            )
        )
    return "\n".join(lines) if lines else "- No evidence-backed positive contribution detected."


def _best_metrics(snapshot: RunComparisonSnapshot) -> dict[str, Any]:
    row = snapshot.best_candidate
    metrics = row.get("metrics", {}) if isinstance(row, dict) else {}
    return metrics if isinstance(metrics, dict) else {}


def _improved_metrics(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    improved: list[str] = []
    for name in ("map50", "recall"):
        previous_value = _number(previous, name)
        current_value = _number(current, name)
        if previous_value is not None and current_value is not None and current_value > previous_value:
            improved.append(name)
    previous_latency = _number(previous, "latency_ms", "latency")
    current_latency = _number(current, "latency_ms", "latency")
    if previous_latency is not None and current_latency is not None and current_latency < previous_latency:
        improved.append("latency_ms")
    return improved


def _action_text(row: dict[str, Any]) -> str:
    changed = row.get("changed_variables")
    if isinstance(changed, dict) and changed:
        parts = [f"{key}={value}" for key, value in changed.items()]
        return "; ".join(parts)
    components = row.get("components")
    if isinstance(components, list) and components:
        return ", ".join(str(component) for component in components)
    return "baseline_or_unknown_action"


def _delta_text(current: dict[str, Any], previous: dict[str, Any], *keys: str) -> str:
    current_value = _number(current, *keys)
    previous_value = _number(previous, *keys)
    if current_value is None or previous_value is None:
        return "unknown"
    delta = current_value - previous_value
    return f"{delta:+.4g}"


def _metric_text(metrics: Any, *keys: str) -> str:
    value = _number(metrics, *keys)
    return "unknown" if value is None else f"{value:g}"


def _number(metrics: Any, *keys: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _short_sha(value: str) -> str:
    return value[:12] if value and value != "unknown" else "unknown"
