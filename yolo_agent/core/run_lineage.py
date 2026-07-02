"""Run lineage graph for cross-round loop queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_serializer


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
            if scored.best_metric_value is None:
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
    metadata: dict[str, Any] | None = None,
) -> RunLineageRecord:
    """Create a lineage record and compute resolved evidence."""
    inherited = list(dict.fromkeys(inherited_missing_evidence or []))
    current = list(dict.fromkeys(current_missing_evidence or []))
    resolved = [item for item in inherited if item not in set(current)]
    metric_data = metrics or {}
    metric_name, metric_value = _best_metric(metric_data)
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
        best_metric_name=metric_name,
        best_metric_value=metric_value,
        metrics=metric_data,
        metadata=metadata or {},
    )


def _best_metric(
    metrics: dict[str, float | int | str | bool | None],
    preferred: list[str] | None = None,
) -> tuple[str | None, float | None]:
    names = preferred or ["map50", "mAP", "map", "map50_95", "recall"]
    for name in names:
        value = metrics.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return name, float(value)
    return None, None
