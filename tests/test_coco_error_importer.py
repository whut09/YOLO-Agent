"""COCO eval importer tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.cli import main
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.tools.coco_error_importer import import_coco_eval_metrics, parse_coco_eval_metrics


def test_parse_coco_eval_stats_and_per_class_ap(tmp_path: Path) -> None:
    """Importer should parse COCO stats array and per-class AP facts."""
    eval_path = tmp_path / "coco_eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "stats": [0.4, 0.61, 0.43, 0.21, 0.45, 0.58, 0.3, 0.5, 0.6, 0.25, 0.52, 0.66],
                "per_class_ap": {
                    "person": 0.52,
                    "traffic light": 0.31,
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = parse_coco_eval_metrics(eval_path)

    assert metrics["coco_ap50_95"] == 0.4
    assert metrics["ap_small"] == 0.21
    assert metrics["ap_medium"] == 0.45
    assert metrics["ap_large"] == 0.58
    assert metrics["per_class_ap/person"] == 0.52
    assert metrics["per_class_ap/traffic light"] == 0.31


def test_import_coco_eval_writes_node_level_evidence(tmp_path: Path) -> None:
    """COCO eval import should force candidate/node-level evidence."""
    eval_path = tmp_path / "coco_eval.json"
    eval_path.write_text(
        json.dumps({"AP_small": 0.2, "AP_medium": 0.4, "AP_large": 0.5}),
        encoding="utf-8",
    )
    store = EvidenceStore(tmp_path / "runs")

    result = import_coco_eval_metrics(
        eval_path=eval_path,
        evidence_store=store,
        run_id="exp001",
        candidate_id="yolo26n_seed1",
        node_id="node_yolo26n_seed1",
        dataset_version="coco2017",
    )
    evidence = store.load_run("exp001")

    assert result.metrics["ap_small"] == 0.2
    assert {record.metric_name for record in evidence.metric_records} >= {"ap_small", "ap_medium", "ap_large"}
    assert all(record.candidate_id == "yolo26n_seed1" for record in evidence.metric_records)
    assert all(record.node_id == "node_yolo26n_seed1" for record in evidence.metric_records)
    assert all(record.validator == "coco_error_importer" for record in evidence.metric_records)


def test_loop_import_coco_eval_cli(tmp_path: Path) -> None:
    """loop import-coco-eval should attach official COCO metrics to a loop run."""
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task_type": "detect",
                "scene": "generic",
                "class_names": ["object"],
                "primary_metric": {"name": "map50_95"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "labels" / "train").mkdir(parents=True)
    (dataset_root / "images" / "train" / "img1.jpg").write_bytes(b"image")
    (dataset_root / "labels" / "train" / "img1.txt").write_text("", encoding="utf-8")
    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nnames:\n  0: object\n", encoding="utf-8")
    run_root = tmp_path / "runs"
    eval_path = tmp_path / "coco_eval.json"
    eval_path.write_text(json.dumps({"stats": [0.5, 0.7, 0.55, 0.25, 0.44, 0.6]}), encoding="utf-8")

    assert main(
        [
            "loop",
            "init",
            "--run-id",
            "coco-eval-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
            "--dataset-version",
            "coco2017",
        ]
    ) == 0
    assert main(
        [
            "loop",
            "import-coco-eval",
            "--run",
            str(run_root / "coco-eval-run"),
            "--eval",
            str(eval_path),
            "--candidate-id",
            "yolo26n_seed1",
            "--node-id",
            "node_yolo26n_seed1",
        ]
    ) == 0

    evidence = EvidenceStore(run_root).load_run("coco-eval-run")
    assert any(record.metric_name == "ap_small" for record in evidence.metric_records)
    assert any(record.metric_name == "coco_ap50_95" for record in evidence.metric_records)
