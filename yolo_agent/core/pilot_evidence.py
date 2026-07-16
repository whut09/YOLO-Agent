"""Completeness gate for candidate-specific pilot COCO evidence."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.evidence_store import EvidenceStore


class PilotEvidenceCompletenessResult(BaseModel):
    """Evidence readiness for one executed pilot node."""

    run_id: str
    candidate_id: str
    node_id: str
    complete: bool = False
    missing_metrics: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    missing_error_report_fields: list[str] = Field(default_factory=list)
    missing_fact_groups: list[str] = Field(default_factory=list)
    evidence_actions: list[str] = Field(default_factory=list)


class PilotEvidenceCompletenessGate:
    """Require current-node COCO evidence before another training proposal."""

    required_metrics = ("ap_small", "ap_medium", "ap_large")
    required_metric_prefixes = ("per_class_ap/", "per_class_ar/")
    required_artifacts = ("coco_predictions", "coco_eval", "coco_error_report")
    required_report_fields = (
        "false_negative_top_classes",
        "localization_error_top_classes",
        "background_false_positive_top_classes",
    )
    required_fact_groups = ("area_metric", "per_class_metric")

    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store

    def evaluate(self, *, run_id: str, candidate_id: str, node_id: str) -> PilotEvidenceCompletenessResult:
        self.evidence_store.create_run(run_id)
        evidence = self.evidence_store.load_run(run_id)
        records = [
            item
            for item in evidence.metric_records
            if item.candidate_id == candidate_id and item.node_id == node_id and item.verified
        ]
        metric_names = {item.metric_name for item in records}
        missing_metrics = [name for name in self.required_metrics if name not in metric_names]
        missing_metrics.extend(
            prefix.rstrip("/")
            for prefix in self.required_metric_prefixes
            if not any(name.startswith(prefix) for name in metric_names)
        )

        node_artifacts = {
            entry.name: entry.path
            for entry in evidence.artifact_manifest
            if entry.name.startswith(f"{node_id}_") and entry.verify()
        }
        missing_artifacts = [
            suffix for suffix in self.required_artifacts if not any(name.endswith(suffix) for name in node_artifacts)
        ]
        report_path = next(
            (path for name, path in node_artifacts.items() if name.endswith("coco_error_report")),
            None,
        )
        report = _read_mapping(report_path)
        missing_report_fields = [name for name in self.required_report_fields if name not in report]

        facts = [
            fact
            for fact in ErrorFactStore(self.evidence_store.root).read(run_id)
            if fact.run_id == run_id and fact.candidate_id == candidate_id and fact.node_id == node_id
        ]
        fact_groups = {fact.fact_type for fact in facts}
        missing_fact_groups = [name for name in self.required_fact_groups if name not in fact_groups]
        actions: list[str] = []
        if "coco_predictions" in missing_artifacts:
            actions.append("run_coco_post_eval")
        if "coco_eval" in missing_artifacts or missing_metrics:
            actions.append("import_coco_eval")
        if "coco_error_report" in missing_artifacts or missing_report_fields:
            actions.append("mine_coco_errors")
        if missing_fact_groups:
            actions.append("import_current_node_error_facts")

        return PilotEvidenceCompletenessResult(
            run_id=run_id,
            candidate_id=candidate_id,
            node_id=node_id,
            complete=not (missing_metrics or missing_artifacts or missing_report_fields or missing_fact_groups),
            missing_metrics=missing_metrics,
            missing_artifacts=missing_artifacts,
            missing_error_report_fields=missing_report_fields,
            missing_fact_groups=missing_fact_groups,
            evidence_actions=list(dict.fromkeys(actions)),
        )


def _read_mapping(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
