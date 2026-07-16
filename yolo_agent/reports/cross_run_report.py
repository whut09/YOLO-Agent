"""Cross-run comparison reports for loop harness runs."""

from __future__ import annotations

from pathlib import Path
from statistics import stdev
from typing import Any

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.pareto import ParetoFront, ParetoSelector, candidate_metrics_from_row
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.evidence_selector import EvidenceSelector, select_metric_evidence
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
            "## Contribution Confidence",
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
    run_context = _read_run_context(run_dir)
    metadata = run_context.get("metadata", {}) if isinstance(run_context.get("metadata"), dict) else {}
    protocol_hash = str(metadata.get("baseline_protocol_hash") or "") or None
    manifest_sha = str(run_context.get("dataset_manifest_sha256") or "") or None
    current_records = select_metric_evidence(
        evidence.metric_records,
        EvidenceSelector(
            current_run_id=run_dir.name,
            current_run_only=True,
            inherited_context=False,
            baseline_reference=False,
            same_protocol_hash=protocol_hash,
            same_dataset_manifest=manifest_sha,
            verified=True,
        ),
    ).records
    evidence = evidence.model_copy(update={"metric_records": current_records})
    context = _load_context(run_dir, evidence)
    rows = _candidate_rows(evidence, context, _experiment_nodes(context))
    pareto_front = ParetoSelector().select(
        [metrics for row in rows if (metrics := candidate_metrics_from_row(row)) is not None]
    )
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
    for snapshot in runs:
        for contribution in _single_variable_contributions(snapshot):
            lines.append(
                "- `{run}`: {status} contribution from single-variable ablation `{candidate}` "
                "changed `{variable}`; parent=`{parent}`; improved {metrics}; "
                "deltas {deltas}; confidence={reason}.".format(
                    run=snapshot.run_id,
                    status=contribution["confidence_status"],
                    candidate=contribution["candidate_id"],
                    variable=contribution["changed_variable"],
                    parent=contribution["parent_id"],
                    metrics=", ".join(contribution["improved_metrics"]),
                    deltas=_deltas_summary(contribution["deltas"]),
                    reason=contribution["confidence_reason"],
                )
            )
    if lines:
        return "\n".join(lines)
    return "- No single-variable ablation contribution with trusted parent evidence detected."


def _single_variable_contributions(snapshot: RunComparisonSnapshot) -> list[dict[str, Any]]:
    contributions: list[dict[str, Any]] = []
    rows = [row for row in snapshot.candidate_rows if row.get("has_evidence")]
    rows_by_id = {str(row.get("id", "")): row for row in rows}
    rows_by_node = {str(row.get("node_id", "")): row for row in rows if row.get("node_id")}
    baseline = _baseline_row(rows)
    for row in rows:
        metrics = row.get("metrics", {})
        if row.get("inference_policy_changed") or (
            isinstance(metrics, dict) and metrics.get("inference_policy_changed")
        ):
            # Slicing is an inference policy experiment, not evidence that a
            # training component caused the metric delta.
            continue
        changed = row.get("changed_variables")
        if not isinstance(changed, dict) or len(changed) != 1:
            continue
        parent = _parent_row(row, rows_by_id, rows_by_node, baseline)
        if parent is None:
            continue
        deltas = _metric_deltas(parent.get("metrics", {}), row.get("metrics", {}))
        improved = _positive_delta_metrics(deltas)
        if not improved:
            continue
        variable, value = next(iter(changed.items()))
        confidence = _contribution_confidence(row, parent, rows, improved)
        contributions.append(
            {
                "candidate_id": str(row.get("id", "unknown")),
                "parent_id": str(parent.get("id", row.get("parent_id") or "baseline")),
                "changed_variable": f"{variable}={value}",
                "deltas": deltas,
                "improved_metrics": improved,
                "confidence_status": confidence["status"],
                "confidence_reason": confidence["reason"],
            }
        )
    return contributions


def _contribution_confidence(
    row: dict[str, Any],
    parent: dict[str, Any],
    rows: list[dict[str, Any]],
    improved_metrics: list[str],
) -> dict[str, str]:
    """Return confirmed only for repeated-seed single-variable evidence."""
    repeated_deltas = _repeated_seed_deltas(row, parent, rows, improved_metrics)
    repeat_count = len(repeated_deltas)
    ci_metrics = [
        metric
        for metric in improved_metrics
        if _has_confidence_interval_support(row.get("metrics", {}), parent.get("metrics", {}), metric)
        or _repeated_delta_interval_support(repeated_deltas, metric)
    ]
    if repeat_count >= 3 and set(improved_metrics) <= set(ci_metrics):
        reason = f"repeated_seeds:{repeat_count};confidence_interval:" + ",".join(ci_metrics)
        return {
            "status": "confirmed",
            "reason": reason,
        }
    reason = f"insufficient_repeated_seeds:{repeat_count}/3"
    if repeat_count >= 3:
        reason = f"repeated_seeds_without_positive_confidence_interval:{repeat_count}"
    elif ci_metrics:
        reason += ";confidence_interval_present_but_not_confirmatory:" + ",".join(ci_metrics)
    return {
        "status": "possible",
        "reason": reason,
    }


def _has_confidence_interval_support(
    current: Any,
    previous: Any,
    metric: str,
) -> bool:
    """Return whether confidence intervals support a positive contribution."""
    if not isinstance(current, dict) or not isinstance(previous, dict):
        return False
    delta_low = _number(current, f"delta_{metric}_ci_low")
    delta_high = _number(current, f"delta_{metric}_ci_high")
    if delta_low is not None and delta_high is not None:
        if metric == "latency_ms":
            return delta_high < 0
        return delta_low > 0
    current_low = _number(current, f"{metric}_ci_low")
    current_high = _number(current, f"{metric}_ci_high")
    previous_low = _number(previous, f"{metric}_ci_low")
    previous_high = _number(previous, f"{metric}_ci_high")
    if None in {current_low, current_high, previous_low, previous_high}:
        return False
    if metric == "latency_ms":
        return bool(current_high < previous_low)  # type: ignore[operator]
    return bool(current_low > previous_high)  # type: ignore[operator]


def _repeated_seed_deltas(
    row: dict[str, Any],
    parent: dict[str, Any],
    rows: list[dict[str, Any]],
    improved_metrics: list[str],
) -> dict[str, dict[str, float]]:
    """Collect distinct-seed deltas for matching single-variable rows."""
    changed = row.get("changed_variables")
    parent_id = str(parent.get("id", row.get("parent_id") or "baseline"))
    seeds: dict[str, dict[str, float]] = {}
    for candidate in rows:
        if candidate.get("changed_variables") != changed:
            continue
        candidate_parent_id = str(candidate.get("parent_id") or parent_id)
        if candidate_parent_id != parent_id:
            continue
        deltas = _metric_deltas(parent.get("metrics", {}), candidate.get("metrics", {}))
        if not all(metric in _positive_delta_metrics(deltas) for metric in improved_metrics):
            continue
        seed = candidate.get("seed")
        if seed is not None:
            seeds[str(seed)] = deltas
    return seeds


def _repeated_delta_interval_support(
    repeated_deltas: dict[str, dict[str, float]], metric: str,
) -> bool:
    values = [deltas[metric] for deltas in repeated_deltas.values() if metric in deltas]
    if len(values) < 3:
        return False
    mean = sum(values) / len(values)
    critical = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(values) - 1, 1.96)
    margin = critical * stdev(values) / (len(values) ** 0.5)
    low, high = mean - margin, mean + margin
    return high < 0 if metric == "latency_ms" else low > 0


def _baseline_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if "baseline" in str(row.get("id", "")).lower():
            return row
    for row in rows:
        changed = row.get("changed_variables")
        if not changed:
            return row
    return None


def _parent_row(
    row: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
    rows_by_node: dict[str, dict[str, Any]],
    baseline: dict[str, Any] | None,
) -> dict[str, Any] | None:
    parent_id = row.get("parent_id")
    if parent_id is None:
        return baseline
    parent_key = str(parent_id)
    return rows_by_id.get(parent_key) or rows_by_node.get(parent_key) or baseline


def _metric_deltas(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    metric_keys = {
        "map50": ("map50", "mAP", "map", "map50_95"),
        "recall": ("recall",),
        "latency_ms": ("latency_ms", "latency"),
    }
    for metric_name, keys in metric_keys.items():
        previous_value = _number(previous, *keys)
        current_value = _number(current, *keys)
        if previous_value is not None and current_value is not None:
            deltas[metric_name] = current_value - previous_value
    return deltas


def _positive_delta_metrics(deltas: dict[str, float]) -> list[str]:
    improved: list[str] = []
    for name in ("map50", "recall"):
        if deltas.get(name, 0.0) > 0:
            improved.append(name)
    if deltas.get("latency_ms", 0.0) < 0:
        improved.append("latency_ms")
    return improved


def _deltas_summary(deltas: dict[str, float]) -> str:
    if not deltas:
        return "unknown"
    return ", ".join(f"{name}={value:+.4g}" for name, value in deltas.items())


def _best_metrics(snapshot: RunComparisonSnapshot) -> dict[str, Any]:
    row = snapshot.best_candidate
    metrics = row.get("metrics", {}) if isinstance(row, dict) else {}
    return metrics if isinstance(metrics, dict) else {}


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
