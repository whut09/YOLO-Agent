"""COCO baseline evidence contract for trusted optimization loops."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import Evidence, MetricEvidence, MetricValue


COCO_BASELINE_PROFILES = {"baseline_full", "baseline_confirm"}
COCO_BASELINE_REQUIRED_METRICS = [
    "map50_95",
    "ap_small",
    "ap_medium",
    "ap_large",
    "latency_ms",
    "model_size_mb",
]
COCO_BASELINE_REQUIRED_ARTIFACTS = [
    "results_csv",
    "best_pt",
    "args_yaml",
    "runtime_profile",
    "coco_eval",
]


class CocoBaselineNodeStatus(BaseModel):
    """Contract status for one baseline experiment node."""

    candidate_id: str
    node_id: str
    profile: str | None = None
    ok: bool
    missing_metrics: list[str] = Field(default_factory=list)
    missing_metric_groups: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    stale_artifacts: list[str] = Field(default_factory=list)


class CocoBaselineEvidenceResult(BaseModel):
    """Batch COCO baseline evidence contract result."""

    ok: bool
    trusted: bool
    baseline_nodes: list[CocoBaselineNodeStatus] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CocoBaselineEvidenceContract:
    """Validate that baseline_full/baseline_confirm nodes have complete COCO evidence."""

    def __init__(
        self,
        required_metrics: list[str] | None = None,
        required_artifacts: list[str] | None = None,
    ) -> None:
        self.required_metrics = required_metrics or list(COCO_BASELINE_REQUIRED_METRICS)
        self.required_artifacts = required_artifacts or list(COCO_BASELINE_REQUIRED_ARTIFACTS)

    def evaluate(self, evidence: Evidence) -> CocoBaselineEvidenceResult:
        """Evaluate baseline nodes against the COCO evidence contract."""
        node_profiles = _baseline_node_profiles(evidence.metric_records)
        statuses = [
            self._node_status(evidence, node_id, profile)
            for node_id, profile in sorted(node_profiles.items())
        ]
        missing_required: list[str] = []
        if evidence.artifact_manifest_path is None or not evidence.artifact_manifest_path.is_file():
            missing_required.append("artifact_manifest")
        if not statuses:
            missing_required.append("baseline_full_or_confirm_node")
        for status in statuses:
            missing_required.extend(f"{status.node_id}:{name}" for name in status.missing_metrics)
            missing_required.extend(f"{status.node_id}:{name}" for name in status.missing_metric_groups)
            missing_required.extend(f"{status.node_id}:artifact:{name}" for name in status.missing_artifacts)
            missing_required.extend(f"{status.node_id}:stale_artifact:{name}" for name in status.stale_artifacts)
        ok = not missing_required
        return CocoBaselineEvidenceResult(
            ok=ok,
            trusted=ok,
            baseline_nodes=statuses,
            missing_required=list(dict.fromkeys(missing_required)),
            warnings=[] if ok else ["COCO baseline evidence is incomplete; do not trust full-candidate promotion."],
        )

    def persist_result(
        self,
        store: EvidenceStore,
        run_id: str,
        result: CocoBaselineEvidenceResult,
        dataset_version: str = "coco2017",
    ) -> Path:
        """Persist contract result as artifact and run-level metric evidence."""
        artifact_path = store.create_run(run_id) / "artifacts" / "coco_baseline_evidence.json"
        artifact_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        store.log_artifact_manifest(
            run_id=run_id,
            name="coco_baseline_evidence",
            artifact_path=artifact_path,
            producer_stage="coco_baseline_evidence_contract",
        )
        metrics = _current_run_metrics(store, run_id)
        metrics.update(
            {
                "coco_baseline_evidence_trusted": result.trusted,
                "coco_baseline_node_count": len(result.baseline_nodes),
                "coco_baseline_missing_required": ";".join(result.missing_required),
            }
        )
        store.log_metrics(run_id, metrics)
        store.log_candidate_metrics(
            run_id=run_id,
            candidate_id="coco_baseline_evidence",
            node_id="coco_baseline_evidence",
            metrics={
                "coco_baseline_evidence_trusted": result.trusted,
                "coco_baseline_node_count": len(result.baseline_nodes),
                "coco_baseline_missing_required": ";".join(result.missing_required),
            },
            dataset_version=dataset_version,
            split="runtime",
            source="coco_baseline_evidence_contract",
            verified=True,
            validator="coco_baseline_evidence_contract",
            source_artifact=artifact_path,
        )
        return artifact_path

    def _node_status(self, evidence: Evidence, node_id: str, profile: str) -> CocoBaselineNodeStatus:
        index = EvidenceIndex(evidence.metric_records)
        candidate_id = _candidate_id_for_node(evidence.metric_records, node_id)
        missing_metrics = [
            metric
            for metric in self.required_metrics
            if _metric(index, node_id, metric) is None
        ]
        missing_groups = []
        if not _metric_group(evidence.metric_records, node_id, "per_class_ap/"):
            missing_groups.append("per_class_ap")
        if not _metric_group(evidence.metric_records, node_id, "per_class_ar/"):
            missing_groups.append("per_class_ar")
        artifact_status = {
            name: _artifact_ok(evidence, node_id, name)
            for name in self.required_artifacts
        }
        missing_artifacts = [name for name, state in artifact_status.items() if state == "missing"]
        stale_artifacts = [name for name, state in artifact_status.items() if state == "stale"]
        ok = not missing_metrics and not missing_groups and not missing_artifacts and not stale_artifacts
        return CocoBaselineNodeStatus(
            candidate_id=candidate_id or "",
            node_id=node_id,
            profile=profile,
            ok=ok,
            missing_metrics=missing_metrics,
            missing_metric_groups=missing_groups,
            missing_artifacts=missing_artifacts,
            stale_artifacts=stale_artifacts,
        )


def coco_metric_aliases(metrics: dict[str, MetricValue]) -> dict[str, MetricValue]:
    """Return canonical COCO metric aliases for node-level evidence."""
    aliases: dict[str, MetricValue] = {}
    mapping = {
        "coco_ap50_95": "map50_95",
        "coco_ap50": "map50",
        "coco_ap75": "map75",
        "AP_small": "ap_small",
        "AP_medium": "ap_medium",
        "AP_large": "ap_large",
        "AR_small": "ar_small",
        "AR_medium": "ar_medium",
        "AR_large": "ar_large",
    }
    for source, target in mapping.items():
        if source in metrics and target not in metrics:
            aliases[target] = metrics[source]
    return aliases


def _baseline_node_profiles(records: list[MetricEvidence]) -> dict[str, str]:
    profiles: dict[str, str] = {}
    for record in records:
        if record.metric_name != "training_budget_profile" or not record.verified:
            continue
        profile = str(record.value)
        if profile in COCO_BASELINE_PROFILES:
            profiles[record.node_id] = profile
    return profiles


def _metric(index: EvidenceIndex, node_id: str, metric_name: str) -> MetricEvidence | None:
    record = index.select_one(node_id=node_id, metric_name=metric_name, verified=True)
    return record if record is not None and record.value is not None else None


def _metric_group(records: list[MetricEvidence], node_id: str, prefix: str) -> bool:
    return any(
        record.node_id == node_id
        and record.metric_name.startswith(prefix)
        and record.value is not None
        and record.verified
        for record in records
    )


def _artifact_ok(evidence: Evidence, node_id: str, name: str) -> str:
    aliases = {f"{node_id}_{name}", name}
    entries = [entry for entry in evidence.artifact_manifest if entry.name in aliases]
    if not entries:
        return "missing"
    return "ok" if any(entry.verify() for entry in entries) else "stale"


def _candidate_id_for_node(records: list[MetricEvidence], node_id: str) -> str | None:
    for record in records:
        if record.node_id == node_id and record.candidate_id:
            return record.candidate_id
    return None


def _current_run_metrics(store: EvidenceStore, run_id: str) -> dict[str, MetricValue]:
    try:
        return dict(store.load_run(run_id).metrics)
    except FileNotFoundError:
        return {}
