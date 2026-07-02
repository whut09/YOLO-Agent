"""Local evidence store tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.core.evidence_store import EvidenceStore


def test_evidence_store_logs_and_loads_run(tmp_path: Path) -> None:
    """EvidenceStore should persist config, metrics, and artifact files."""
    store = EvidenceStore(tmp_path / "runs")
    artifact_source = tmp_path / "model.yaml"
    artifact_source.write_text("nc: 1\n", encoding="utf-8")

    run_dir = store.create_run("run-001")
    config_path = store.log_config("run-001", {"seed": 42, "model": "yolo11n"})
    metrics_path = store.log_metrics("run-001", {"map50": 0.5, "ok": True})
    artifact_path = store.log_artifact("run-001", artifact_source)
    evidence = store.load_run("run-001")

    assert run_dir == tmp_path / "runs" / "run-001"
    assert config_path == run_dir / "config.yaml"
    assert metrics_path == run_dir / "metrics.json"
    assert artifact_path == run_dir / "artifacts" / "model.yaml"
    assert evidence.config == {"seed": 42, "model": "yolo11n"}
    assert evidence.metrics == {"map50": 0.5, "ok": True}
    assert evidence.artifacts["model.yaml"] == artifact_path


def test_evidence_store_logs_candidate_metric_records(tmp_path: Path) -> None:
    """EvidenceStore should persist candidate/node-level metrics separately."""
    store = EvidenceStore(tmp_path / "runs")

    records_path = store.log_candidate_metrics(
        run_id="run-001",
        candidate_id="baseline",
        node_id="node-baseline",
        dataset_version="dataset-v3",
        split="val",
        source="benchmark_csv",
        metrics={"map50": 0.6, "recall": 0.7, "latency_ms": 12},
    )
    evidence = store.load_run("run-001")

    assert records_path == tmp_path / "runs" / "run-001" / "metrics_by_node.jsonl"
    assert evidence.metric_records_path == records_path
    assert len(evidence.metric_records) == 3
    assert evidence.metric_records[0].candidate_id == "baseline"
    assert evidence.metric_records[0].node_id == "node-baseline"
    assert evidence.metric_records[0].dataset_version == "dataset-v3"
    assert {record.metric_name: record.value for record in evidence.metric_records} == {
        "map50": 0.6,
        "recall": 0.7,
        "latency_ms": 12,
    }


def test_evidence_store_rejects_nested_run_ids(tmp_path: Path) -> None:
    """Run IDs should not escape the store root."""
    store = EvidenceStore(tmp_path / "runs")

    with pytest.raises(ValueError):
        store.create_run("../bad")
