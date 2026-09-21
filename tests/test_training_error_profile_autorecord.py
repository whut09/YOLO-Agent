"""Prompt-18I-v2 §6: per-node error profile auto-recorded after training import."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.adapters.ultralytics.training import UltralyticsRunImporter
from yolo_agent.core.evidence_store import EvidenceStore


def _make_coco_data_yaml(root: Path) -> Path:
    images = root / "images" / "val2017"
    annotations = root / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    (images / "000000000001.jpg").write_bytes(b"image")
    (annotations / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 100, "height": 100}],
                "categories": [{"id": 1, "name": "bottle"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 8, 8],
                        "area": 64,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [60, 60, 40, 40],
                        "area": 1600,
                        "iscrowd": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    data_yaml = root / "coco.yaml"
    data_yaml.write_text(
        "path: .\nval: images/val2017\nval_annotations: annotations/instances_val2017.json\nnames:\n  1: bottle\n",
        encoding="utf-8",
    )
    return data_yaml


def _make_run_dir(root: Path, predictions: list[dict]) -> Path:
    run_dir = root / "train_run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.40,0.50,0.55,0.30\n",
        encoding="utf-8",
    )
    (run_dir / "args.yaml").write_text("imgsz: 640\nepochs: 100\n", encoding="utf-8")
    (weights_dir / "best.pt").write_bytes(b"0" * 4096)
    (run_dir / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    return run_dir


def _node():
    from yolo_agent.agents.candidate_generator import CandidateConfig
    from yolo_agent.core.experiment_graph import ExperimentNode

    candidate = CandidateConfig(
        candidate_id="yolo26s_coco_baseline",
        base_model="yolo26s.pt",
        scale="s",
        framework="ultralytics",
        train_overrides={"imgsz": 768},
    )
    return ExperimentNode(
        node_id="node_yolo26s_coco_baseline",
        candidate_config=candidate,
        data_version="coco2017",
        seed=1,
    )


def test_importer_writes_error_profile(tmp_path: Path) -> None:
    """§6: a real training import auto-writes per-node error_profile.yaml."""
    dataset_yaml = _make_coco_data_yaml(tmp_path / "coco")
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 8, 8], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [500, 500, 30, 30], "score": 0.6},
    ]
    run_dir = _make_run_dir(tmp_path, predictions)
    runs_root = tmp_path / "runs"
    store = EvidenceStore(runs_root)
    UltralyticsRunImporter(store).import_run(
        "exp001", _node(), run_dir, sample_gpu=False, data_path=dataset_yaml
    )

    profile_paths = list((runs_root / "exp001" / "artifacts").glob("*/error_profile.yaml"))
    assert profile_paths, "training import must auto-write error_profile.yaml"
    payload = yaml.safe_load(profile_paths[0].read_text(encoding="utf-8"))
    assert payload["schema_version"].startswith("detection_error_profile.")
    assert payload["candidate_id"] == "yolo26s_coco_baseline"
    assert payload["node_id"] == "node_yolo26s_coco_baseline"
    # Real numbers from the real GT + predictions: 1 FN (the missed large GT)
    # and 1 background FP.
    assert payload["false_negative"]["total"] == 1
    assert payload["false_positive"]["background_fp"] == 1
    assert payload["scale"]["recall_small"] == 1.0
    assert payload["scale"]["recall_large"] is None

    evidence = store.load_run("exp001")
    assert any(
        "error_profile" in entry.name for entry in evidence.artifact_manifest
    ), "profile must be registered in the EvidenceStore manifest"


def test_importer_records_gap_on_unparsable_predictions(tmp_path: Path, monkeypatch) -> None:
    """When the profile builder fails, the import records an evidence gap."""
    import yolo_agent.core.detection_error_profile_builder as builder_module

    dataset_yaml = _make_coco_data_yaml(tmp_path / "coco")
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 8, 8], "score": 0.9}]
    run_dir = _make_run_dir(tmp_path, predictions)

    def _boom(source):
        raise ValueError("simulated profile failure")

    monkeypatch.setattr(builder_module, "build_detection_error_profile", _boom)
    runs_root = tmp_path / "runs"
    store = EvidenceStore(runs_root)
    UltralyticsRunImporter(store).import_run(
        "exp001", _node(), run_dir, sample_gpu=False, data_path=dataset_yaml
    )
    evidence = store.load_run("exp001")
    failed = [
        entry
        for entry in evidence.artifact_manifest
        if "error_profile_failed" in entry.producer_stage
    ]
    assert failed, "the failed profile must be recorded as an evidence gap"
