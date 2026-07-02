"""Markdown experiment report generation from local evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from yolo_agent.agents.pareto import ParetoFront, ParetoSelector, candidate_metrics_from_row
from yolo_agent.core.evidence_contract import NO_EVIDENCE_WARNING
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import Evidence, ExperimentPlan


METRIC_COLUMNS = ["mAP", "precision", "recall", "latency", "model_size"]


class ExperimentReportGenerator:
    """Generate Markdown reports from EvidenceStore run directories."""

    def generate(self, run_path: Path | str, out_path: Path | str) -> str:
        """Generate and write an experiment report."""
        run_dir = Path(run_path)
        evidence = EvidenceStore(run_dir.parent).load_run(run_dir.name)
        context = _load_context(run_dir, evidence)
        markdown = self.render(evidence, context)
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        return markdown

    def render(self, evidence: Evidence, context: dict[str, Any]) -> str:
        """Render a Markdown report without inventing missing metrics."""
        experiment_nodes = _experiment_nodes(context)
        candidate_rows = _candidate_rows(evidence, context, experiment_nodes)
        pareto_front = _pareto_front(candidate_rows)
        evidence_trusted = _evidence_trusted(context)

        lines = [
            "# YOLO Agent Experiment Report",
            "",
            "## Evidence Gate",
            "",
            _evidence_gate_text(context),
            "",
            "## Task Profile",
            "",
            _task_profile_text(evidence, context),
            "",
            "## Data Diagnosis",
            "",
            _data_diagnosis_text(context),
            "",
            "## Candidate Models",
            "",
            _candidate_list_text(candidate_rows),
            "",
            "## Ablation Variables",
            "",
            _ablation_text(context, experiment_nodes),
            "",
            "## Metrics",
            "",
            _metrics_table(candidate_rows),
            "",
            "## Pareto Front",
            "",
            _pareto_front_text(pareto_front),
            "",
            "## Best Model Recommendation",
            "",
            _recommendation_text(pareto_front, evidence_trusted),
            "",
            "## Why Recommended",
            "",
            _why_text(pareto_front, evidence_trusted),
            "",
            "## Next Round Suggestions",
            "",
            _next_round_text(context, candidate_rows),
            "",
            "## Risks And Unverified Items",
            "",
            _risks_text(candidate_rows),
            "",
        ]
        return "\n".join(lines)


def generate_experiment_report(run_path: Path | str, out_path: Path | str) -> str:
    """Generate a Markdown experiment report from a run directory."""
    return ExperimentReportGenerator().generate(run_path, out_path)


def _load_context(run_dir: Path, evidence: Evidence) -> dict[str, Any]:
    context: dict[str, Any] = {"config": evidence.config, "metrics": evidence.metrics}
    for name, loader in {
        "dataset_report": _read_json_or_yaml,
        "experiment_plan": _read_yaml,
        "ablation_plan": _read_yaml,
        "evidence_status": _read_json_or_yaml,
    }.items():
        path = _find_context_file(run_dir, name)
        context[name] = loader(path) if path is not None else None
    return context


def _find_context_file(run_dir: Path, stem: str) -> Path | None:
    candidates = [
        run_dir / f"{stem}.json",
        run_dir / f"{stem}.yaml",
        run_dir / f"{stem}.yml",
        run_dir / "artifacts" / f"{stem}.json",
        run_dir / "artifacts" / f"{stem}.yaml",
        run_dir / "artifacts" / f"{stem}.yml",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _evidence_gate_text(context: dict[str, Any]) -> str:
    status = context.get("evidence_status")
    if not isinstance(status, dict):
        return f"- {NO_EVIDENCE_WARNING}"
    trusted = bool(status.get("trusted"))
    lines = [f"- Trusted: `{trusted}`"]
    warning = status.get("warning")
    if warning:
        lines.append(f"- Warning: {warning}")
    missing = status.get("missing_required", [])
    if isinstance(missing, list) and missing:
        lines.append("- Missing required: " + ", ".join(str(item) for item in missing))
    return "\n".join(lines)


def _evidence_trusted(context: dict[str, Any]) -> bool:
    status = context.get("evidence_status")
    return isinstance(status, dict) and bool(status.get("trusted"))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def _read_json_or_yaml(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    return _read_yaml(path)


def _experiment_nodes(context: dict[str, Any]) -> list[dict[str, Any]]:
    raw_plan = context.get("experiment_plan")
    if not isinstance(raw_plan, dict):
        return []
    try:
        return [node.model_dump(mode="json") for node in ExperimentPlan.model_validate(raw_plan).nodes]
    except Exception:
        nodes = raw_plan.get("nodes", [])
        return nodes if isinstance(nodes, list) else []


def _candidate_rows(
    evidence: Evidence,
    context: dict[str, Any],
    experiment_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if experiment_nodes:
        rows = []
        for node in experiment_nodes:
            candidate = node.get("candidate_config", {})
            metrics = node.get("metrics") if isinstance(node.get("metrics"), dict) else {}
            if not metrics and node.get("node_id") == evidence.run_id:
                metrics = evidence.metrics
            rows.append(
                {
                    "id": str(candidate.get("candidate_id", node.get("node_id", "unknown"))),
                    "base_model": _unknown(candidate.get("base_model")),
                    "scale": _unknown(candidate.get("scale")),
                    "components": candidate.get("components", []),
                    "risk": _unknown(candidate.get("risk")),
                    "metrics": metrics,
                    "has_evidence": bool(metrics),
                    "changed_variables": node.get("changed_variables", {}),
                }
            )
        return rows

    config = context.get("config", {})
    candidates = config.get("candidates") if isinstance(config, dict) else None
    if isinstance(candidates, list) and candidates:
        return [
            {
                "id": str(candidate.get("candidate_id", "unknown")),
                "base_model": _unknown(candidate.get("base_model")),
                "scale": _unknown(candidate.get("scale")),
                "components": candidate.get("components", []),
                "risk": _unknown(candidate.get("risk")),
                "metrics": evidence.metrics if len(candidates) == 1 else {},
                "has_evidence": len(candidates) == 1 and bool(evidence.metrics),
                "changed_variables": candidate.get("changed_variables", {}),
            }
            for candidate in candidates
            if isinstance(candidate, dict)
        ]

    return [
        {
            "id": evidence.run_id,
            "base_model": _unknown(config.get("model") if isinstance(config, dict) else None),
            "scale": "unknown",
            "components": [],
            "risk": "unknown",
            "metrics": evidence.metrics,
            "has_evidence": bool(evidence.metrics),
            "changed_variables": {},
        }
    ]


def _task_profile_text(evidence: Evidence, context: dict[str, Any]) -> str:
    config = context.get("config", {})
    task = config.get("task_spec") or config.get("task_profile") if isinstance(config, dict) else None
    if isinstance(task, dict):
        items = [f"- {key}: `{value}`" for key, value in task.items() if not isinstance(value, (dict, list))]
        return "\n".join(items) if items else "- unknown"
    return f"- Run ID: `{evidence.run_id}`\n- Task profile: unknown"


def _data_diagnosis_text(context: dict[str, Any]) -> str:
    report = context.get("dataset_report")
    if not isinstance(report, dict):
        return "- DatasetReport: unknown"
    lines = [
        f"- Images: {_unknown(report.get('image_count'))}",
        f"- Labels: {_unknown(report.get('label_count'))}",
        f"- Small object ratio: {_unknown(_nested(report, 'object_size_ratio', 'small'))}",
        f"- Empty label images: {_unknown(report.get('empty_label_images'))}",
        f"- Missing label files: {_unknown(report.get('missing_label_files'))}",
    ]
    recommendations = report.get("recommendations", [])
    if isinstance(recommendations, list) and recommendations:
        lines.append("- Data recommendations: " + "; ".join(str(item) for item in recommendations))
    return "\n".join(lines)


def _candidate_list_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- No candidates found."
    lines = []
    for row in rows:
        components = row.get("components") or []
        component_text = ", ".join(str(item) for item in components) if components else "none"
        lines.append(
            f"- `{row['id']}`: base={row['base_model']}, scale={row['scale']}, components={component_text}, risk={row['risk']}"
        )
    return "\n".join(lines)


def _ablation_text(context: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    ablation = context.get("ablation_plan")
    if isinstance(ablation, dict) and isinstance(ablation.get("nodes"), list):
        items = []
        for node in ablation["nodes"]:
            if isinstance(node, dict):
                items.append(f"- `{node.get('node_id', 'unknown')}`: `{node.get('changed_variables', {})}`")
        return "\n".join(items) if items else "- No valid ablations."
    items = [f"- `{node.get('node_id', 'unknown')}`: `{node.get('changed_variables', {})}`" for node in nodes]
    return "\n".join(items) if items else "- Ablation variables: unknown"


def _metrics_table(rows: list[dict[str, Any]]) -> str:
    header = "| candidate | mAP | precision | recall | latency | model size | evidence |\n"
    separator = "|---|---:|---:|---:|---:|---:|---|\n"
    body = []
    for row in rows:
        metrics = row.get("metrics", {})
        evidence_text = "ok" if row.get("has_evidence") else NO_EVIDENCE_WARNING
        body.append(
            "| {id} | {map} | {precision} | {recall} | {latency} | {model_size} | {evidence} |".format(
                id=row["id"],
                map=_metric(metrics, "map", "mAP", "map50", "map50_95"),
                precision=_metric(metrics, "precision"),
                recall=_metric(metrics, "recall"),
                latency=_metric(metrics, "latency", "latency_ms"),
                model_size=_metric(metrics, "model_size", "model_size_mb"),
                evidence=evidence_text,
            )
        )
    return header + separator + "\n".join(body)


def _pareto_front(rows: list[dict[str, Any]]) -> ParetoFront:
    candidates = [
        metrics
        for row in rows
        if (metrics := candidate_metrics_from_row(row)) is not None
    ]
    return ParetoSelector().select(candidates)


def _pareto_front_text(pareto_front: ParetoFront) -> str:
    if not pareto_front.points:
        return "- No evidence-backed Pareto front can be computed."
    lines = ["| model | mAP | latency | model size | robustness | tradeoff |", "|---|---:|---:|---:|---:|---|"]
    for point in pareto_front.points:
        lines.append(
            "| {model} | {accuracy} | {latency} | {model_size} | {robustness} | {tradeoff} |".format(
                model=point.model,
                accuracy=_unknown(point.accuracy),
                latency=_unknown(point.latency),
                model_size=_unknown(point.model_size),
                robustness=_unknown(point.robustness),
                tradeoff=point.tradeoff_summary,
            )
        )
    return "\n".join(lines)


def _recommendation_text(pareto_front: ParetoFront, evidence_trusted: bool) -> str:
    if not evidence_trusted:
        return f"- {NO_EVIDENCE_WARNING}"
    if not pareto_front.points:
        return "- No evidence-backed model can be recommended."
    ids = ", ".join(f"`{point.candidate_id}`" for point in pareto_front.points)
    return f"- Recommend evaluating Pareto-front candidates: {ids}."


def _why_text(pareto_front: ParetoFront, evidence_trusted: bool) -> str:
    if not evidence_trusted or not pareto_front.points:
        return f"- {NO_EVIDENCE_WARNING}"
    lines = [
        "- These candidates are non-dominated across available accuracy, latency, model size, and robustness metrics."
    ]
    lines.extend(f"- `{point.candidate_id}`: {point.tradeoff_summary}" for point in pareto_front.points)
    return "\n".join(lines)


def _next_round_text(context: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    suggestions = []
    if any(not row.get("has_evidence") for row in rows):
        suggestions.append("Run smoke and evidence logging for candidates without evidence.")
    ablation = context.get("ablation_plan")
    invalid = ablation.get("invalid_candidates") if isinstance(ablation, dict) else None
    if isinstance(invalid, list) and invalid:
        suggestions.append("Split invalid multi-variable candidates before training.")
    if not suggestions:
        suggestions.append("Continue with the next single-variable ablation round.")
    return "\n".join(f"- {item}" for item in suggestions)


def _risks_text(rows: list[dict[str, Any]]) -> str:
    risks = []
    for row in rows:
        if row.get("risk") not in {"low", "unknown"}:
            risks.append(f"- `{row['id']}` risk={row['risk']}")
        if not row.get("has_evidence"):
            risks.append(f"- `{row['id']}`: {NO_EVIDENCE_WARNING}")
    return "\n".join(risks) if risks else "- No elevated risks recorded."


def _metric(metrics: Any, *keys: str) -> str:
    if not isinstance(metrics, dict):
        return "unknown"
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _nested(data: dict[str, Any], key: str, nested_key: str) -> Any:
    value = data.get(key)
    return value.get(nested_key) if isinstance(value, dict) else None


def _unknown(value: Any) -> str:
    return "unknown" if value is None else str(value)
