"""Queryable fact index for candidate and node metric evidence."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from yolo_agent.core.experiment_graph import MetricEvidence, MetricValue


SelectionMode = Literal["latest_trusted", "best_value"]


class MetricEvidenceQuery(BaseModel):
    """Filters for candidate/node metric facts."""

    candidate_id: str | None = None
    node_id: str | None = None
    dataset_version: str | None = None
    split: str | None = None
    metric_name: str | None = None
    verified: bool | None = None
    validator: str | None = None
    source: str | None = None


class EvidenceIndex:
    """Query and select metric evidence with deterministic duplicate rules."""

    def __init__(self, records: Iterable[MetricEvidence]) -> None:
        self.records = list(records)

    def query(
        self,
        candidate_id: str | None = None,
        node_id: str | None = None,
        dataset_version: str | None = None,
        split: str | None = None,
        metric_name: str | None = None,
        verified: bool | None = None,
        validator: str | None = None,
        source: str | None = None,
    ) -> list[MetricEvidence]:
        """Return metric records matching all supplied filters."""
        query = MetricEvidenceQuery(
            candidate_id=candidate_id,
            node_id=node_id,
            dataset_version=dataset_version,
            split=split,
            metric_name=metric_name,
            verified=verified,
            validator=validator,
            source=source,
        )
        return [record for record in self.records if _matches(record, query)]

    def select_one(
        self,
        candidate_id: str | None = None,
        node_id: str | None = None,
        dataset_version: str | None = None,
        split: str | None = None,
        metric_name: str | None = None,
        verified: bool | None = None,
        validator: str | None = None,
        source: str | None = None,
        preferred_validators: list[str] | None = None,
    ) -> MetricEvidence | None:
        """Select the most trustworthy duplicate metric observation.

        Duplicate fact selection does not cherry-pick the best score. It prefers
        present values, verified records, preferred validators, higher
        confidence, and then the newest created_at timestamp.
        """
        records = self.query(
            candidate_id=candidate_id,
            node_id=node_id,
            dataset_version=dataset_version,
            split=split,
            metric_name=metric_name,
            verified=verified,
            validator=validator,
            source=source,
        )
        if not records:
            return None
        return max(records, key=lambda record: _trust_rank(record, preferred_validators or []))

    def select_best(
        self,
        candidate_id: str | None = None,
        node_id: str | None = None,
        dataset_version: str | None = None,
        split: str | None = None,
        metric_name: str | None = None,
        verified: bool | None = None,
        validator: str | None = None,
        source: str | None = None,
        preferred_validators: list[str] | None = None,
    ) -> MetricEvidence | None:
        """Select the best metric value among matching records.

        This is intended for ranking candidates. It still requires present
        values and prefers verified/trusted records, but uses the metric's
        higher_is_better direction to compare numeric values.
        """
        records = self.query(
            candidate_id=candidate_id,
            node_id=node_id,
            dataset_version=dataset_version,
            split=split,
            metric_name=metric_name,
            verified=verified,
            validator=validator,
            source=source,
        )
        if not records:
            return None
        return max(records, key=lambda record: _best_value_rank(record, preferred_validators or []))

    def metric_value(self, **filters: object) -> MetricValue:
        """Return the selected duplicate metric value for supplied filters."""
        record = self.select_one(**filters)  # type: ignore[arg-type]
        return record.value if record is not None else None

    def best_metric_value(self, **filters: object) -> MetricValue:
        """Return the selected best metric value for supplied filters."""
        record = self.select_best(**filters)  # type: ignore[arg-type]
        return record.value if record is not None else None

    def metric_mapping(
        self,
        verified: bool | None = True,
        mode: SelectionMode = "best_value",
    ) -> dict[str, MetricValue]:
        """Return one selected value per metric name."""
        metrics: dict[str, MetricValue] = {}
        for metric_name in sorted({record.metric_name for record in self.records}):
            selector = self.select_best if mode == "best_value" else self.select_one
            record = selector(metric_name=metric_name, verified=verified)
            if record is not None and record.value is not None:
                metrics[metric_name] = record.value
        return metrics


def _matches(record: MetricEvidence, query: MetricEvidenceQuery) -> bool:
    return all(
        [
            query.candidate_id is None or record.candidate_id == query.candidate_id,
            query.node_id is None or record.node_id == query.node_id,
            query.dataset_version is None or record.dataset_version == query.dataset_version,
            query.split is None or record.split == query.split,
            query.metric_name is None or record.metric_name == query.metric_name,
            query.verified is None or record.verified is query.verified,
            query.validator is None or record.validator == query.validator,
            query.source is None or record.source == query.source,
        ]
    )


def _trust_rank(record: MetricEvidence, preferred_validators: list[str]) -> tuple[int, int, int, float, datetime]:
    confidence = record.confidence if record.confidence is not None else -1.0
    return (
        int(record.value is not None),
        int(record.verified),
        _validator_rank(record.validator, preferred_validators),
        confidence,
        record.created_at,
    )


def _best_value_rank(record: MetricEvidence, preferred_validators: list[str]) -> tuple[int, int, float, int, float, datetime]:
    numeric = _numeric(record.value)
    if numeric is None:
        score = float("-inf")
    else:
        score = numeric if record.higher_is_better is not False else -numeric
    confidence = record.confidence if record.confidence is not None else -1.0
    return (
        int(record.value is not None),
        int(record.verified),
        score,
        _validator_rank(record.validator, preferred_validators),
        confidence,
        record.created_at,
    )


def _validator_rank(validator: str, preferred_validators: list[str]) -> int:
    if not preferred_validators:
        return 0
    try:
        return len(preferred_validators) - preferred_validators.index(validator)
    except ValueError:
        return -1


def _numeric(value: MetricValue) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None
