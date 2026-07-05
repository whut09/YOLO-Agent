"""COCO baseline evidence contract tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.coco_baseline_evidence import (
    CocoBaselineEvidenceContract,
    coco_metric_aliases,
)
from yolo_agent.core.evidence_store import EvidenceStore


def test_coco_metric_aliases_standardize_official_ap_names() -> None:
    """Official COCO metric names should produce harness-standard aliases."""
    aliases = coco_metric_aliases(
        {
            "coco_ap50_95": 0.41,
            "coco_ap50": 0.62,
            "AP_small": 0.2,
        }
    )

    assert aliases == {"map50_95": 0.41, "map50": 0.62, "ap_small": 0.2}


def test_coco_baseline_contract_accepts_complete_node_evidence(tmp_path: Path) -> None:
    """baseline_confirm nodes should expose all required COCO evidence as node-level facts."""
    store = EvidenceStore(tmp_path / "runs")
    _write_complete_baseline_node(store, tmp_path, "exp001")

    result = CocoBaselineEvidenceContract().evaluate(store.load_run("exp001"))

    assert result.ok is True
    assert result.trusted is True
    assert result.missing_required == []
    assert result.baseline_nodes[0].ok is True


def test_coco_baseline_contract_rejects_missing_per_class_and_artifacts(tmp_path: Path) -> None:
    """Missing per-class AR and runtime artifacts should make baseline evidence untrusted."""
    store = EvidenceStore(tmp_path / "runs")
    _write_complete_baseline_node(
        store,
        tmp_path,
        "exp001",
        include_per_class_ar=False,
        include_runtime_profile=False,
    )

    result = CocoBaselineEvidenceContract().evaluate(store.load_run("exp001"))

    assert result.ok is False
    assert "node_baseline:per_class_ar" in result.missing_required
    assert "node_baseline:artifact:runtime_profile" in result.missing_required


def test_coco_baseline_contract_persists_status_artifact_and_metrics(tmp_path: Path) -> None:
    """Contract results should be stored as reusable evidence for later gates."""
    store = EvidenceStore(tmp_path / "runs")
    _write_complete_baseline_node(store, tmp_path, "exp001")
    result = CocoBaselineEvidenceContract().evaluate(store.load_run("exp001"))

    path = CocoBaselineEvidenceContract().persist_result(store, "exp001", result)
    evidence = store.load_run("exp001")

    assert path.name == "coco_baseline_evidence.json"
    assert "coco_baseline_evidence" in evidence.artifacts
    assert evidence.metrics["coco_baseline_evidence_trusted"] is True
    assert any(record.metric_name == "coco_baseline_evidence_trusted" for record in evidence.metric_records)


def _write_complete_baseline_node(
    store: EvidenceStore,
    tmp_path: Path,
    run_id: str,
    include_per_class_ar: bool = True,
    include_runtime_profile: bool = True,
) -> None:
    node_id = "node_baseline"
    candidate_id = "yolo26s_coco_baseline"
    artifact_dir = tmp_path / "artifacts" / node_id
    weights = artifact_dir / "weights"
    weights.mkdir(parents=True)
    results_csv = artifact_dir / "results.csv"
    args_yaml = artifact_dir / "args.yaml"
    best_pt = weights / "best.pt"
    runtime_profile = artifact_dir / "runtime_profile.json"
    coco_eval = artifact_dir / "coco_eval.json"
    results_csv.write_text("epoch,metrics/mAP50-95(B)\n0,0.4\n", encoding="utf-8")
    args_yaml.write_text("imgsz: 640\n", encoding="utf-8")
    best_pt.write_bytes(b"weights")
    runtime_profile.write_text("{}", encoding="utf-8")
    coco_eval.write_text("{}", encoding="utf-8")

    store.create_run(run_id)
    for name, path in {
        "results_csv": results_csv,
        "best_pt": best_pt,
        "args_yaml": args_yaml,
        "coco_eval": coco_eval,
    }.items():
        store.log_artifact_manifest(run_id, f"{node_id}_{name}", path, "test")
    if include_runtime_profile:
        store.log_artifact_manifest(run_id, f"{node_id}_runtime_profile", runtime_profile, "test")

    metrics = {
        "training_budget_profile": "baseline_confirm",
        "map50_95": 0.4,
        "ap_small": 0.2,
        "ap_medium": 0.4,
        "ap_large": 0.5,
        "latency_ms": 12,
        "model_size_mb": 5,
        "per_class_ap/person": 0.5,
    }
    if include_per_class_ar:
        metrics["per_class_ar/person"] = 0.6
    store.log_candidate_metrics(
        run_id,
        candidate_id,
        node_id,
        metrics,
        dataset_version="coco2017",
        split="val2017",
        source="test",
    )
