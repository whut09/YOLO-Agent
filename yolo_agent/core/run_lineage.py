"""Run lineage graph for cross-round loop queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.experiment_graph import MetricEvidence, MetricValue


LINEAGE_SCHEMA_VERSION = "1.0"


class RunLineageRecord(BaseModel):
    """One append-only lineage snapshot for a run."""

    run_id: str
    run_dir: Path
    parent_run_id: str | None = None
    dataset_version: str = "unversioned"
    dataset_manifest_sha256: str | None = None
    inherited_missing_evidence: list[str] = Field(default_factory=list)
    current_missing_evidence: list[str] = Field(default_factory=list)
    resolved_evidence: list[str] = Field(default_factory=list)
    trusted: bool = False
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    best_candidate_id: str | None = None
    best_node_id: str | None = None
    best_metric_source: str | None = None
    best_metric_scope: str | None = None
    best_candidate_metric: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = LINEAGE_SCHEMA_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_serializer("run_dir")
    def serialize_run_dir(self, value: Path) -> str:
        """Serialize paths portably."""
        return value.as_posix()


class RunLineageGraph(BaseModel):
    """Queryable latest-state lineage graph."""

    records: dict[str, RunLineageRecord] = Field(default_factory=dict)

    def parent_of(self, run_id: str) -> str | None:
        """Return the parent run id for a run."""
        record = self.records.get(run_id)
        return record.parent_run_id if record is not None else None

    def children_of(self, run_id: str) -> list[str]:
        """Return child run ids for a parent run."""
        return sorted(
            record.run_id
            for record in self.records.values()
            if record.parent_run_id == run_id
        )

    def inherited_dataset_manifest_sha(self, run_id: str) -> str | None:
        """Return the dataset manifest hash attached to a run."""
        record = self.records.get(run_id)
        return record.dataset_manifest_sha256 if record is not None else None

    def evidence_delta(self, run_id: str) -> dict[str, list[str]]:
        """Return inherited, current, and resolved evidence for a run."""
        record = self.records.get(run_id)
        if record is None:
            return {"inherited_missing": [], "current_missing": [], "resolved": []}
        return {
            "inherited_missing": list(record.inherited_missing_evidence),
            "current_missing": list(record.current_missing_evidence),
            "resolved": list(record.resolved_evidence),
        }

    def best_trusted_run(self, metric_names: list[str] | None = None) -> RunLineageRecord | None:
        """Return the trusted run with the best available score."""
        preferred = metric_names or ["map50", "mAP", "map", "map50_95", "recall"]
        candidates: list[RunLineageRecord] = []
        for record in self.records.values():
            if not record.trusted:
                continue
            scored = record
            if metric_names is not None:
                scored = _rescore_record(scored, preferred)
            elif scored.best_metric_value is None:
                metric_name, metric_value = _best_metric(scored.metrics, preferred)
                scored = scored.model_copy(update={"best_metric_name": metric_name, "best_metric_value": metric_value})
            if scored.best_metric_value is not None:
                candidates.append(scored)
        if not candidates:
            return None
        return max(candidates, key=lambda record: record.best_metric_value or float("-inf"))


class RunLineageStore:
    """Append-only JSONL store under runs/lineage.jsonl."""

    def __init__(self, root: Path | str = "runs") -> None:
        self.root = Path(root)
        self.path = self.root / "lineage.jsonl"

    def append(self, record: RunLineageRecord) -> RunLineageRecord:
        """Append one lineage snapshot."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        return record

    def read(self) -> list[RunLineageRecord]:
        """Read all lineage snapshots."""
        if not self.path.exists():
            return []
        records: list[RunLineageRecord] = []
        with self.path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                text = line.strip()
                if text:
                    records.append(RunLineageRecord.model_validate(json.loads(text)))
        return records

    def graph(self) -> RunLineageGraph:
        """Return a latest-record graph keyed by run id."""
        latest: dict[str, RunLineageRecord] = {}
        for record in self.read():
            latest[record.run_id] = record
        return RunLineageGraph(records=latest)


def build_lineage_record(
    run_id: str,
    run_dir: Path | str,
    parent_run_id: str | None = None,
    dataset_version: str = "unversioned",
    dataset_manifest_sha256: str | None = None,
    inherited_missing_evidence: list[str] | None = None,
    current_missing_evidence: list[str] | None = None,
    trusted: bool = False,
    metrics: dict[str, float | int | str | bool | None] | None = None,
    metric_records: list[MetricEvidence] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunLineageRecord:
    """Create a lineage record and compute resolved evidence."""
    inherited = list(dict.fromkeys(inherited_missing_evidence or []))
    current = list(dict.fromkeys(current_missing_evidence or []))
    resolved = [item for item in inherited if item not in set(current)]
    metric_data = metrics or {}
    best_summary = _best_node_metric(metric_records or []) or _best_run_metric_summary(run_id, metric_data)
    return RunLineageRecord(
        run_id=run_id,
        run_dir=Path(run_dir),
        parent_run_id=parent_run_id,
        dataset_version=dataset_version,
        dataset_manifest_sha256=dataset_manifest_sha256,
        inherited_missing_evidence=inherited,
        current_missing_evidence=current,
        resolved_evidence=resolved,
        trusted=trusted,
        best_metric_name=best_summary.get("metric_name"),
        best_metric_value=best_summary.get("metric_value"),
        best_candidate_id=best_summary.get("candidate_id"),
        best_node_id=best_summary.get("node_id"),
        best_metric_source=best_summary.get("source"),
        best_metric_scope=best_summary.get("scope"),
        best_candidate_metric=best_summary,
        metrics=metric_data,
        metadata=metadata or {},
    )


def _rescore_record(record: RunLineageRecord, preferred: list[str]) -> RunLineageRecord:
    if record.best_metric_name in preferred and record.best_metric_value is not None:
        return record
    metric_name, metric_value = _best_metric(record.metrics, preferred)
    return record.model_copy(
        update={
            "best_metric_name": metric_name,
            "best_metric_value": metric_value,
            "best_candidate_id": record.run_id if metric_name is not None else None,
            "best_node_id": None,
            "best_metric_scope": "run" if metric_name is not None else None,
            "best_metric_source": "run_metrics" if metric_name is not None else None,
            "best_candidate_metric": _best_run_metric_summary(record.run_id, record.metrics, preferred),
        }
    )


def _best_node_metric(
    records: list[MetricEvidence],
    preferred: list[str] | None = None,
) -> dict[str, Any] | None:
    names = preferred or ["map50", "mAP", "map", "map50_95", "recall"]
    index = EvidenceIndex(records)
    for name in names:
        best = index.select_best(metric_name=name, verified=True)
        if best is None or _numeric(best.value) is None:
            continue
        return {
            "candidate_id": best.candidate_id,
            "node_id": best.node_id,
            "metric_name": best.metric_name,
            "metric_value": _numeric(best.value),
            "source": best.source,
            "scope": "node",
            "dataset_version": best.dataset_version,
            "split": best.split,
            "verified": best.verified,
            "validator": best.validator,
        }
    return None


def _best_run_metric_summary(
    run_id: str,
    metrics: dict[str, MetricValue],
    preferred: list[str] | None = None,
) -> dict[str, Any]:
    metric_name, metric_value = _best_metric(metrics, preferred)
    if metric_name is None:
        return {}
    return {
        "candidate_id": run_id,
        "node_id": None,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "source": "run_metrics",
        "scope": "run",
    }


def _best_metric(
    metrics: dict[str, float | int | str | bool | None],
    preferred: list[str] | None = None,
) -> tuple[str | None, float | None]:
    names = preferred or ["map50", "mAP", "map", "map50_95", "recall"]
    for name in names:
        value = metrics.get(name)
        numeric = _numeric(value)
        if numeric is not None:
            return name, numeric
    return None, None


def _numeric(value: MetricValue) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None
