from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from yolo_agent.adapters.ultralytics.coco_post_eval import (
    CocoPostEvalConfig,
    build_coco_post_eval_spec,
    should_run_coco_post_eval,
    write_coco_eval_report,
)
from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.pilot_evidence import PilotEvidenceCompletenessGate


def test_coco_post_eval_builds_fixed_full_validation_command(tmp_path: Path) -> None:
    config = CocoPostEvalConfig(enabled=True)
    spec = build_coco_post_eval_spec(
        executable="yolo",
        checkpoint=tmp_path / "best.pt",
        data=tmp_path / "coco.yaml",
        output_dir=tmp_path / "run" / "coco_post_eval",
        device="0",
        workers=8,
        config=config,
    )

    assert spec.command_type == "benchmark"
    assert "val" in spec.argv
    assert "imgsz=640" in spec.argv
    assert "split=val" in spec.argv
    assert "save_json=True" in spec.argv
    assert not any(item.startswith("fraction=") for item in spec.argv)
    assert spec.metadata["evaluation_protocol"] == "coco_val2017_fixed_640"
    assert should_run_coco_post_eval("pilot", config) is True
    assert should_run_coco_post_eval("debug", config) is False


def test_yolo26_coco_config_enables_post_eval() -> None:
    config = UltralyticsTrainingConfig.from_yaml("configs/training/yolo26_coco_goal.yaml", budget_profile="pilot")

    assert config.coco_post_eval.enabled is True
    assert config.coco_post_eval.imgsz == 640
    assert "pilot" in config.coco_post_eval.profiles


def test_write_coco_eval_report_includes_area_and_per_class_metrics(tmp_path: Path, monkeypatch) -> None:
    class FakeCOCO:
        def __init__(self, path: str) -> None:
            self.path = path

        def loadRes(self, path: str):
            return {"predictions": path}

        def loadCats(self, category_ids):
            return [{"id": category_id, "name": name} for category_id, name in zip(category_ids, ["person", "bottle"])]

    class FakeParams:
        catIds = [1, 44]
        maxDets = [1, 10, 100]

    class FakeCOCOeval:
        def __init__(self, ground_truth, predictions, iou_type: str) -> None:
            self.params = FakeParams()
            self.stats = np.arange(12, dtype=float) / 100.0
            self.eval = {
                "precision": np.full((2, 3, 2, 1, 1), 0.5, dtype=float),
                "recall": np.full((2, 2, 1, 1), 0.4, dtype=float),
            }

        def evaluate(self) -> None:
            return None

        def accumulate(self) -> None:
            return None

        def summarize(self) -> None:
            print("COCO summary")

    package = types.ModuleType("pycocotools")
    coco_module = types.ModuleType("pycocotools.coco")
    cocoeval_module = types.ModuleType("pycocotools.cocoeval")
    coco_module.COCO = FakeCOCO
    cocoeval_module.COCOeval = FakeCOCOeval
    monkeypatch.setitem(sys.modules, "pycocotools", package)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", coco_module)
    monkeypatch.setitem(sys.modules, "pycocotools.cocoeval", cocoeval_module)

    annotations = tmp_path / "instances_val2017.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "coco_eval.json"
    annotations.write_text("{}", encoding="utf-8")
    predictions.write_text("[]", encoding="utf-8")

    write_coco_eval_report(annotations_path=annotations, predictions_path=predictions, output_path=output)
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["protocol"]["imgsz"] == 640
    assert report["AP_small"] == 0.03
    assert report["per_class_ap"] == {"bottle": 0.5, "person": 0.5}
    assert report["per_class_ar"] == {"bottle": 0.4, "person": 0.4}


def test_pilot_evidence_gate_requires_current_node_coco_evidence(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "runs")
    run_id = "candidate-run"
    candidate_id = "candidate"
    node_id = "node_candidate"

    incomplete = PilotEvidenceCompletenessGate(store).evaluate(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
    )
    assert incomplete.complete is False
    assert "run_coco_post_eval" in incomplete.evidence_actions

    run_dir = store.create_run(run_id)
    artifacts = run_dir / "artifacts"
    predictions = artifacts / "predictions.json"
    coco_eval = artifacts / "coco_eval.json"
    error_report = artifacts / "coco_error_report.json"
    predictions.write_text("[]", encoding="utf-8")
    coco_eval.write_text("{}", encoding="utf-8")
    error_report.write_text(
        json.dumps(
            {
                "false_negative_top_classes": [],
                "localization_error_top_classes": [],
                "background_false_positive_top_classes": [],
            }
        ),
        encoding="utf-8",
    )
    for name, path in {
        f"{node_id}_coco_predictions": predictions,
        f"{node_id}_coco_eval": coco_eval,
        f"{node_id}_coco_error_report": error_report,
    }.items():
        store.log_artifact_manifest(run_id, name, path, "test")
    store.log_candidate_metrics(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        metrics={
            "ap_small": 0.2,
            "ap_medium": 0.4,
            "ap_large": 0.5,
            "per_class_ap/person": 0.4,
            "per_class_ar/person": 0.5,
        },
        dataset_version="coco2017",
        split="val2017",
        source="test",
        verified=True,
        validator="test",
    )
    ErrorFactStore(store.root).append(
        run_id,
        [
            ErrorFact(
                run_id=run_id,
                candidate_id=candidate_id,
                node_id=node_id,
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.2,
            ),
            ErrorFact(
                run_id=run_id,
                candidate_id=candidate_id,
                node_id=node_id,
                fact_type="per_class_metric",
                subject="person",
                class_name="person",
                metric_name="per_class_ap",
                value=0.4,
            ),
        ],
    )

    complete = PilotEvidenceCompletenessGate(store).evaluate(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
    )
    wrong_node = PilotEvidenceCompletenessGate(store).evaluate(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id="node_other",
    )

    assert complete.complete is True
    assert wrong_node.complete is False
