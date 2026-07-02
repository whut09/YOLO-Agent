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
    assert evidence.artifact_manifest_path == run_dir / "artifacts" / "artifact_manifest.jsonl"
    assert len(evidence.artifact_manifest) == 1
    assert evidence.artifact_manifest[0].name == "model.yaml"
    assert evidence.artifact_manifest[0].producer_stage == "evidence_store"
    assert evidence.artifact_manifest[0].verify() is True


def test_evidence_store_removes_tampered_manifest_artifacts(tmp_path: Path) -> None:
    """Manifest-tracked artifacts should not load when their hash no longer matches."""
    store = EvidenceStore(tmp_path / "runs")
    artifact_source = tmp_path / "model.yaml"
    artifact_source.write_text("nc: 1\n", encoding="utf-8")
    artifact_path = store.log_artifact("run-001", artifact_source, name="model")

    artifact_path.write_text("nc: 99\n", encoding="utf-8")
    evidence = store.load_run("run-001")

    assert evidence.artifact_manifest[0].verify() is False
    assert "model" not in evidence.artifacts
    assert "model.yaml" not in evidence.artifacts


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
        verified=True,
        validator="unit-test",
        source_artifact=tmp_path / "metrics.csv",
        metrics={"map50": 0.6, "recall": 0.7, "latency_ms": 12},
    )
    evidence = store.load_run("run-001")

    assert records_path == tmp_path / "runs" / "run-001" / "metrics_by_node.jsonl"
    assert evidence.metric_records_path == records_path
    assert len(evidence.metric_records) == 3
    assert evidence.metric_records[0].candidate_id == "baseline"
    assert evidence.metric_records[0].node_id == "node-baseline"
    assert evidence.metric_records[0].dataset_version == "dataset-v3"
    assert evidence.metric_records[0].verified is True
    assert evidence.metric_records[0].validator == "unit-test"
    assert evidence.metric_records[0].source_artifact == tmp_path / "metrics.csv"
    assert evidence.metric_records[0].metric_schema_version == "1.0"
    assert evidence.metric_records[0].higher_is_better is True
    latency_record = next(record for record in evidence.metric_records if record.metric_name == "latency_ms")
    assert latency_record.higher_is_better is False
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
