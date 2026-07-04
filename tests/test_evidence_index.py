"""Evidence fact index tests."""

from __future__ import annotations

from datetime import datetime, timezone

from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.experiment_graph import MetricEvidence


def _record(
    candidate_id: str,
    node_id: str,
    metric_name: str,
    value: float,
    *,
    dataset_version: str = "dataset-v1",
    split: str = "val",
    verified: bool = True,
    validator: str = "official_eval",
    confidence: float | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> MetricEvidence:
    return MetricEvidence(
        candidate_id=candidate_id,
        node_id=node_id,
        dataset_version=dataset_version,
        split=split,
        metric_name=metric_name,
        value=value,
        verified=verified,
        validator=validator,
        confidence=confidence,
        created_at=datetime.fromisoformat(created_at).astimezone(timezone.utc),
    )


def test_evidence_index_filters_metric_records() -> None:
    """Index query should support candidate, node, dataset, split, metric, verified, and validator filters."""
    index = EvidenceIndex(
        [
            _record("baseline", "node-baseline", "map50", 0.6, validator="official_eval"),
            _record("baseline", "node-baseline", "recall", 0.7, validator="official_eval"),
            _record("nwd", "node-nwd", "map50", 0.72, split="test", validator="draft_eval", verified=False),
        ]
    )

    records = index.query(
        candidate_id="baseline",
        node_id="node-baseline",
        dataset_version="dataset-v1",
        split="val",
        metric_name="map50",
        verified=True,
        validator="official_eval",
    )

    assert len(records) == 1
    assert records[0].value == 0.6


def test_select_one_prefers_trusted_duplicate_not_best_score() -> None:
    """Duplicate selection should prefer trust metadata before metric magnitude."""
    index = EvidenceIndex(
        [
            _record(
                "baseline",
                "node-baseline",
                "map50",
                0.99,
                verified=False,
                validator="draft_eval",
                confidence=1.0,
            ),
            _record(
                "baseline",
                "node-baseline",
                "map50",
                0.61,
                validator="official_eval",
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
            ),
            _record(
                "baseline",
                "node-baseline",
                "map50",
                0.59,
                validator="official_eval",
                confidence=0.9,
                created_at="2026-01-02T00:00:00+00:00",
            ),
        ]
    )

    selected = index.select_one(
        candidate_id="baseline",
        node_id="node-baseline",
        metric_name="map50",
        preferred_validators=["official_eval"],
    )

    assert selected is not None
    assert selected.value == 0.59
    assert selected.verified is True


def test_select_best_respects_metric_direction() -> None:
    """Best-value selection should maximize accuracy metrics and minimize lower-is-better metrics."""
    index = EvidenceIndex(
        [
            _record("baseline", "node-baseline", "map50", 0.6),
            _record("nwd", "node-nwd", "map50", 0.72),
            _record("baseline", "node-baseline", "latency_ms", 14),
            _record("nwd", "node-nwd", "latency_ms", 9),
        ]
    )

    best_map = index.select_best(metric_name="map50", verified=True)
    best_latency = index.select_best(metric_name="latency_ms", verified=True)

    assert best_map is not None
    assert best_map.candidate_id == "nwd"
    assert best_latency is not None
    assert best_latency.candidate_id == "nwd"
    assert best_latency.value == 9


def test_metric_mapping_uses_best_verified_record_per_metric() -> None:
    """Run-level mappings derived from node metrics should ignore unverified records and use best values."""
    index = EvidenceIndex(
        [
            _record("baseline", "node-baseline", "map50", 0.6),
            _record("nwd", "node-nwd", "map50", 0.72),
            _record("draft", "node-draft", "map50", 0.99, verified=False),
            _record("baseline", "node-baseline", "latency_ms", 14),
            _record("nwd", "node-nwd", "latency_ms", 9),
        ]
    )

    assert index.metric_mapping() == {"latency_ms": 9, "map50": 0.72}
