"""Policy memory and learner tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.policy_learner import PolicyLearner
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.policy_memory import PolicyMemoryRecord, PolicyMemoryStore


def test_policy_memory_store_appends_idempotently_and_queries(tmp_path: Path) -> None:
    """Policy memory should be append-only but skip duplicate semantic records."""
    store = PolicyMemoryStore(tmp_path / "runs")
    record = PolicyMemoryRecord(
        run_id="child",
        parent_run_id="parent",
        dataset_version="coco2017",
        action="loss.bbox.nwd",
        target="area_metric:small:ap_small",
        metric_name="ap_small",
        before=0.214,
        after=0.229,
        delta=0.015,
        candidate_id="candidate_nwd",
        node_id="node_nwd",
        changed_variables={"bbox_loss": ["loss.bbox.nwd"]},
    )

    first = store.append([record])
    second = store.append([record])

    assert len(first) == 1
    assert second == []
    assert len(store.read()) == 1
    assert store.query(action="loss.bbox.nwd", metric_name="ap_small")[0].delta == 0.015
    assert store.query(action="loss.bbox.nwd", min_confidence="medium") == []


def test_policy_learner_records_action_effect_with_cost(tmp_path: Path) -> None:
    """Learner should convert comparable error deltas into action-effect memory."""
    evidence_store = EvidenceStore(tmp_path / "runs")
    evidence_store.log_candidate_metrics(
        "parent",
        candidate_id="baseline",
        node_id="node_baseline",
        metrics={"latency_ms": 12.0, "model_size_mb": 5.0, "ap_small": 0.214},
        dataset_version="coco2017",
    )
    evidence_store.log_candidate_metrics(
        "child",
        candidate_id="candidate_nwd",
        node_id="node_nwd_seed1",
        metrics={"latency_ms": 13.0, "model_size_mb": 5.5, "ap_small": 0.229},
        dataset_version="coco2017",
    )
    error_delta = {
        "improved_errors": [
            {
                "trend": "improved",
                "fact_type": "area_metric",
                "subject": "small",
                "area": "small",
                "metric_name": "ap_small",
                "parent_value": 0.214,
                "current_value": 0.229,
                "delta": 0.015,
                "action_candidates": ["small_object_recipe", "sahi_or_tiling_eval"],
                "candidate_id": "candidate_nwd",
                "node_id": "node_nwd_seed1",
            }
        ],
        "regressed_errors": [],
        "unchanged_errors": [],
    }
    learner = PolicyLearner(PolicyMemoryStore(tmp_path / "runs"))

    records = learner.learn_from_error_delta(
        run_id="child",
        parent_run_id="parent",
        dataset_version="coco2017",
        error_delta=error_delta,
        current_evidence=evidence_store.load_run("child"),
        parent_evidence=evidence_store.load_run("parent"),
        changed_variables={"bbox_loss": ["loss.bbox.nwd"]},
        scenario="generic",
    )

    assert len(records) == 1
    record = records[0]
    assert record.action == "loss.bbox.nwd"
    assert record.target == "area_metric:small:ap_small"
    assert record.before == 0.214
    assert record.after == 0.229
    assert record.delta == 0.015
    assert record.effect_delta == 0.015
    assert record.inferred_action is False
    assert record.confidence == "low"
    assert record.cost.latency_delta_pct == 8.333333
    assert record.cost.model_size_delta_pct == 10.0

    summary = PolicyMemoryStore(tmp_path / "runs").summarize(action="loss.bbox.nwd")
    assert summary[0].mean_effect_delta == 0.015
    assert summary[0].mean_latency_delta_pct == 8.333333


def test_policy_learner_marks_action_candidates_as_inferred(tmp_path: Path) -> None:
    """Without changed variables, learner must not pretend candidates were executed."""
    error_delta = {
        "improved_errors": [
            {
                "trend": "improved",
                "fact_type": "area_metric",
                "subject": "small",
                "area": "small",
                "metric_name": "ap_small",
                "parent_value": 0.2,
                "current_value": 0.21,
                "delta": 0.01,
                "action_candidates": ["small_object_recipe", "sahi_or_tiling_eval"],
                "candidate_id": "candidate",
                "node_id": "node",
            }
        ],
    }

    records = PolicyLearner(PolicyMemoryStore(tmp_path / "runs")).learn_from_error_delta(
        run_id="child",
        parent_run_id="parent",
        dataset_version="coco2017",
        error_delta=error_delta,
        changed_variables={},
    )

    assert {record.action for record in records} == {"small_object_recipe", "sahi_or_tiling_eval"}
    assert all(record.inferred_action for record in records)
    assert all(record.confidence == "low" for record in records)
    assert all("inferred" in record.confidence_reason for record in records)
