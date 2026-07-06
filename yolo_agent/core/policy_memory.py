"""Long-term policy memory learned from error-fact deltas."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, model_validator


PolicyConfidence = Literal["low", "medium", "high"]
PolicyTrend = Literal["improved", "regressed", "unchanged", "resolved", "new", "current"]

CONFIDENCE_RANK: dict[PolicyConfidence, int] = {"low": 0, "medium": 1, "high": 2}


class PolicyActionCost(BaseModel):
    """Runtime and deployment cost observed for one action effect."""

    latency_before_ms: float | None = None
    latency_after_ms: float | None = None
    latency_delta_ms: float | None = None
    latency_delta_pct: float | None = None
    model_size_before_mb: float | None = None
    model_size_after_mb: float | None = None
    model_size_delta_mb: float | None = None
    model_size_delta_pct: float | None = None
    gpu_hours: float | None = None


class PolicyMemoryRecord(BaseModel):
    """One learned action-effect observation from a closed-loop run."""

    record_id: str = ""
    run_id: str
    parent_run_id: str | None = None
    dataset_version: str = "unversioned"
    split: str = "val"
    scenario: str | None = None
    action: str
    target: str
    target_fact_type: str | None = None
    target_subject: str | None = None
    class_name: str | None = None
    class_pair: str | None = None
    area: str | None = None
    metric_name: str | None = None
    before: float | None = None
    after: float | None = None
    delta: float | None = None
    effect_delta: float | None = None
    higher_is_better: bool = True
    trend: PolicyTrend = "unchanged"
    candidate_id: str | None = None
    node_id: str | None = None
    cost: PolicyActionCost = Field(default_factory=PolicyActionCost)
    confidence: PolicyConfidence = "low"
    confidence_reason: str = "single observation"
    seed_count: int = 1
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    inferred_action: bool = False
    source: str = "error_fact_delta"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def fill_derived_fields(self) -> "PolicyMemoryRecord":
        """Fill deterministic id and normalized effect direction."""
        if self.effect_delta is None and self.delta is not None:
            self.effect_delta = self.delta if self.higher_is_better else -self.delta
        if not self.record_id:
            self.record_id = _record_id(self)
        return self


class PolicyMemorySummary(BaseModel):
    """Aggregated historical effect for one action/target/metric bucket."""

    action: str
    target: str | None = None
    metric_name: str | None = None
    record_count: int
    mean_delta: float | None = None
    mean_effect_delta: float | None = None
    mean_latency_delta_pct: float | None = None
    mean_model_size_delta_pct: float | None = None
    confidence_counts: dict[str, int] = Field(default_factory=dict)
    latest_record_ids: list[str] = Field(default_factory=list)


class PolicyMemoryStore:
    """Append-only JSONL memory at ``runs/policy_memory.jsonl``."""

    def __init__(self, root: Path | str = "runs") -> None:
        self.root = Path(root)

    @property
    def path(self) -> Path:
        """Return the policy memory JSONL path."""
        return self.root / "policy_memory.jsonl"

    def append(self, records: Iterable[PolicyMemoryRecord]) -> list[PolicyMemoryRecord]:
        """Append new records idempotently and return records actually written."""
        materialized = list(records)
        if not materialized:
            return []
        existing_ids = {record.record_id for record in self.read()}
        new_records = [record for record in materialized if record.record_id not in existing_ids]
        if not new_records:
            return []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            for record in new_records:
                file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        return new_records

    def read(self) -> list[PolicyMemoryRecord]:
        """Read all memory records."""
        if not self.path.is_file():
            return []
        records: list[PolicyMemoryRecord] = []
        with self.path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                text = line.strip()
                if text:
                    records.append(PolicyMemoryRecord.model_validate(json.loads(text)))
        return records

    def query(
        self,
        action: str | None = None,
        target: str | None = None,
        metric_name: str | None = None,
        dataset_version: str | None = None,
        run_id: str | None = None,
        min_confidence: PolicyConfidence | None = None,
    ) -> list[PolicyMemoryRecord]:
        """Return records matching all supplied filters."""
        records = self.read()
        if min_confidence is not None:
            min_rank = CONFIDENCE_RANK[min_confidence]
        else:
            min_rank = None
        return [
            record
            for record in records
            if (action is None or record.action == action)
            and (target is None or record.target == target)
            and (metric_name is None or record.metric_name == metric_name)
            and (dataset_version is None or record.dataset_version == dataset_version)
            and (run_id is None or record.run_id == run_id)
            and (min_rank is None or CONFIDENCE_RANK[record.confidence] >= min_rank)
        ]

    def summarize(
        self,
        action: str | None = None,
        target: str | None = None,
        metric_name: str | None = None,
        dataset_version: str | None = None,
    ) -> list[PolicyMemorySummary]:
        """Aggregate memory by action, target, and metric."""
        groups: dict[tuple[str, str | None, str | None], list[PolicyMemoryRecord]] = defaultdict(list)
        for record in self.query(
            action=action,
            target=target,
            metric_name=metric_name,
            dataset_version=dataset_version,
        ):
            groups[(record.action, record.target, record.metric_name)].append(record)
        summaries: list[PolicyMemorySummary] = []
        for (group_action, group_target, group_metric), records in sorted(groups.items()):
            confidence_counts: dict[str, int] = defaultdict(int)
            for record in records:
                confidence_counts[record.confidence] += 1
            summaries.append(
                PolicyMemorySummary(
                    action=group_action,
                    target=group_target,
                    metric_name=group_metric,
                    record_count=len(records),
                    mean_delta=_mean(record.delta for record in records),
                    mean_effect_delta=_mean(record.effect_delta for record in records),
                    mean_latency_delta_pct=_mean(record.cost.latency_delta_pct for record in records),
                    mean_model_size_delta_pct=_mean(record.cost.model_size_delta_pct for record in records),
                    confidence_counts=dict(confidence_counts),
                    latest_record_ids=[record.record_id for record in sorted(records, key=lambda item: item.created_at)[-5:]],
                )
            )
        return summaries


def _record_id(record: PolicyMemoryRecord) -> str:
    payload = {
        "run_id": record.run_id,
        "parent_run_id": record.parent_run_id,
        "dataset_version": record.dataset_version,
        "split": record.split,
        "action": record.action,
        "target": record.target,
        "metric_name": record.metric_name,
        "before": record.before,
        "after": record.after,
        "delta": record.delta,
        "candidate_id": record.candidate_id,
        "node_id": record.node_id,
        "changed_variables": record.changed_variables,
        "inferred_action": record.inferred_action,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 6)
