"""Evidence contract tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.evidence_contract import EvidenceGate, NO_EVIDENCE_WARNING, default_loop_evidence_requirements
from yolo_agent.core.evidence_store import EvidenceStore


def test_evidence_gate_reports_missing_required_items(tmp_path: Path) -> None:
    """Gate should make missing evidence explicit."""
    store = EvidenceStore(tmp_path / "runs")
    run_dir = store.create_run("gate-run")
    (run_dir / "artifacts" / "dataset_report.json").write_text("{}", encoding="utf-8")
    store.log_metrics("gate-run", {"map50": 0.5})
    evidence = store.load_run("gate-run")

    result = EvidenceGate(default_loop_evidence_requirements()).evaluate(evidence)

    assert result.ok is False
    assert result.trusted is False
    assert result.warning == NO_EVIDENCE_WARNING
    assert "dataset_report" not in result.missing_required
    assert "label_quality_report" in result.missing_required
    assert "smoke_result" in result.missing_required
    assert "recall" in result.missing_required


def test_evidence_gate_accepts_artifacts_and_metrics(tmp_path: Path) -> None:
    """Gate should pass when required artifacts and metrics are present."""
    store = EvidenceStore(tmp_path / "runs")
    run_dir = store.create_run("gate-ok")
    for name in ["dataset_report.json", "annotation_advice.json", "smoke_result.json"]:
        (run_dir / "artifacts" / name).write_text("{}", encoding="utf-8")
    store.log_metrics("gate-ok", {"map50": 0.5, "recall": 0.7, "latency_ms": 10})

    result = EvidenceGate(default_loop_evidence_requirements()).evaluate(store.load_run("gate-ok"))

    assert result.ok is True
    assert result.trusted is True
    assert result.missing_required == []


def test_evidence_gate_accepts_candidate_metric_records(tmp_path: Path) -> None:
    """Gate should count candidate/node metrics as metric evidence."""
    store = EvidenceStore(tmp_path / "runs")
    run_dir = store.create_run("gate-node-metrics")
    for name in ["dataset_report.json", "annotation_advice.json", "smoke_result.json"]:
        (run_dir / "artifacts" / name).write_text("{}", encoding="utf-8")
    store.log_candidate_metrics(
        "gate-node-metrics",
        candidate_id="baseline",
        node_id="node-baseline",
        metrics={"map50": 0.5, "recall": 0.7, "latency_ms": 10},
    )

    result = EvidenceGate(default_loop_evidence_requirements()).evaluate(store.load_run("gate-node-metrics"))

    assert result.ok is True
    assert result.trusted is True
    assert result.missing_required == []


def test_evidence_gate_rejects_unverified_candidate_metric_records(tmp_path: Path) -> None:
    """Gate should not count unverified candidate/node metrics as trusted evidence."""
    store = EvidenceStore(tmp_path / "runs")
    run_dir = store.create_run("gate-unverified")
    for name in ["dataset_report.json", "annotation_advice.json", "smoke_result.json"]:
        (run_dir / "artifacts" / name).write_text("{}", encoding="utf-8")
    store.log_candidate_metrics(
        "gate-unverified",
        candidate_id="baseline",
        node_id="node-baseline",
        metrics={"map50": 0.5, "recall": 0.7, "latency_ms": 10},
        verified=False,
        validator="draft_parser",
    )

    result = EvidenceGate(default_loop_evidence_requirements()).evaluate(store.load_run("gate-unverified"))

    assert result.ok is False
    assert "map50" in result.missing_required
    assert "recall" in result.missing_required
    assert "latency_ms" in result.missing_required


def test_evidence_gate_rejects_tampered_manifest_artifact_even_with_loop_path(tmp_path: Path) -> None:
    """Manifest hash mismatches should override raw loop artifact paths."""
    store = EvidenceStore(tmp_path / "runs")
    source = tmp_path / "dataset_report.json"
    source.write_text("{}", encoding="utf-8")
    artifact_path = store.log_artifact("tampered", source, name="dataset_report")
    artifact_path.write_text('{"changed": true}', encoding="utf-8")
    evidence = store.load_run("tampered")

    result = EvidenceGate(["dataset_report"]).evaluate(
        evidence,
        artifacts={"dataset_report": artifact_path},
    )

    assert result.ok is False
    assert "dataset_report" in result.missing_required
