from __future__ import annotations

import json
import sys
import types
from io import StringIO
from pathlib import Path

import numpy as np

from yolo_agent.adapters.ultralytics.coco_post_eval import (
    CocoPostEvalConfig,
    build_coco_post_eval_spec,
    should_run_coco_post_eval,
    write_coco_eval_report,
)
from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig
from yolo_agent.adapters.ultralytics.batch_tuner import BatchTuningConfig
from yolo_agent.adapters.ultralytics.data_cache_policy import DataCachePolicyConfig
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.executor import UltralyticsTrainExecutor
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.pilot_evidence import PilotEvidenceCompletenessGate
from yolo_agent.core.pilot_evidence import validate_coco_evidence_artifacts


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
        def __init__(self, path: str | None = None) -> None:
            self.path = path
            self.dataset = {"images": [], "categories": []}

        def loadRes(self, path: str):
            return {"predictions": path}

        def loadCats(self, category_ids):
            return [{"id": category_id, "name": name} for category_id, name in zip(category_ids, ["person", "bottle"])]

        def createIndex(self) -> None:
            return None

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
    protocol_hash = "protocol-1"

    incomplete = PilotEvidenceCompletenessGate(store).evaluate(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        protocol_hash=protocol_hash,
    )
    assert incomplete.complete is False
    assert "run_coco_post_eval" in incomplete.evidence_actions

    run_dir = store.create_run(run_id)
    artifacts = run_dir / "artifacts"
    predictions = artifacts / "predictions.json"
    coco_eval = artifacts / "coco_eval.json"
    error_report = artifacts / "coco_error_report.json"
    predictions.write_text("[]", encoding="utf-8")
    coco_eval.write_text(
        json.dumps(
            {
                "AP_small": 0.2,
                "AP_medium": 0.4,
                "AP_large": 0.5,
                "per_class_ap": {"person": 0.4},
                "per_class_ar": {"person": 0.5},
            }
        ),
        encoding="utf-8",
    )
    error_report.write_text(
        json.dumps(
            {
                "false_negative_top_classes": [],
                "localization_error_top_classes": [],
                "background_false_positive_top_classes": [],
                "class_confusion_pairs": {},
            }
        ),
        encoding="utf-8",
    )
    for name, path in {
        f"{node_id}_coco_predictions": predictions,
        f"{node_id}_coco_eval": coco_eval,
        f"{node_id}_coco_error_report": error_report,
    }.items():
        store.log_artifact_manifest(
            run_id,
            name,
            path,
            "test",
            candidate_id=candidate_id,
            node_id=node_id,
            protocol_hash=protocol_hash,
        )
    store.upsert_candidate_metrics(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        metrics={
            "ap_small": 0.2,
            "ap_medium": 0.4,
            "ap_large": 0.5,
            "per_class_ap/person": 0.4,
            "per_class_ar/person": 0.5,
            "fn_heavy_classes": "[]",
            "background_fp_classes": "[]",
            "localization_heavy_classes": "[]",
            "confusion_summary": "{}",
        },
        dataset_version="coco2017",
        split="val2017",
        source="test",
        verified=True,
        validator="test",
        protocol_hash=protocol_hash,
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
                protocol_hash=protocol_hash,
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
                protocol_hash=protocol_hash,
            ),
        ],
    )

    complete = PilotEvidenceCompletenessGate(store).evaluate(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        protocol_hash=protocol_hash,
    )
    wrong_node = PilotEvidenceCompletenessGate(store).evaluate(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id="node_other",
        protocol_hash=protocol_hash,
    )
    wrong_protocol = PilotEvidenceCompletenessGate(store).evaluate(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        protocol_hash="protocol-other",
    )

    assert complete.complete is True
    assert complete.invalid_artifacts == {}
    assert complete.artifact_contract_hash
    assert set(complete.artifact_hashes) == {
        "predictions.json",
        "coco_eval.json",
        "coco_error_report.json",
    }
    assert wrong_node.complete is False
    assert wrong_protocol.complete is False


def test_coco_artifact_contract_rejects_semantically_incomplete_json(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    evaluation = tmp_path / "coco_eval.json"
    errors = tmp_path / "coco_error_report.json"
    predictions.write_text(json.dumps([{"image_id": 1}]), encoding="utf-8")
    evaluation.write_text(json.dumps({"AP_small": 0.1}), encoding="utf-8")
    errors.write_text(json.dumps({"false_negative_top_classes": []}), encoding="utf-8")

    result = validate_coco_evidence_artifacts(
        predictions_path=predictions,
        eval_path=evaluation,
        error_report_path=errors,
    )

    assert result.valid is False
    assert result.contract_hash
    assert result.invalid_artifacts["predictions.json"] == [
        "prediction_0_missing_required_fields"
    ]
    assert "missing_per_class_ap" in result.invalid_artifacts["coco_eval.json"]
    assert "invalid_class_confusion_pairs" in result.invalid_artifacts["coco_error_report.json"]


def test_executor_completes_fixed_coco_evidence_and_recovery_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import yolo_agent.adapters.ultralytics.coco_post_eval as post_eval_mod
    import yolo_agent.core.executor as executor_mod
    from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter

    dataset_root = tmp_path / "coco"
    annotations_dir = dataset_root / "annotations"
    annotations_dir.mkdir(parents=True)
    annotations_path = annotations_dir / "instances_val2017.json"
    annotations_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 640, "height": 640}],
                "categories": [{"id": 1, "name": "bottle"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400}
                ],
            }
        ),
        encoding="utf-8",
    )
    data_yaml = dataset_root / "coco.yaml"
    data_yaml.write_text("path: .\ntrain: images/train2017\nval: images/val2017\nnames: {1: bottle}\n", encoding="utf-8")

    training_run_dir = tmp_path / "ultralytics" / "candidate"
    weights_dir = training_run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"weights")
    (training_run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50-95(B)\n2,0.31\n",
        encoding="utf-8",
    )
    (training_run_dir / "args.yaml").write_text(
        "imgsz: 640\nepochs: 3\nbatch: 48\nfraction: 0.1\n",
        encoding="utf-8",
    )
    protocol_hash = "protocol-pilot-3"
    node = ExperimentNode(
        node_id="node_candidate__pilot_3",
        candidate_config=CandidateConfig(
            candidate_id="candidate",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        seed=42,
    )
    metadata = {
        "run_protocol_hash": protocol_hash,
        "dataset_manifest_sha256": "dataset-sha",
        "subset_manifest_sha256": "subset-sha",
        "batch_policy_hash": "batch-sha",
        "eval_protocol_hash": "eval-sha",
        "ultralytics_version": "test-version",
        "round_stage": "pilot_3",
        "training_budget_profile": "pilot",
        "epochs": 3,
    }
    train_spec = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=data_yaml,
        project=training_run_dir.parent,
        name=training_run_dir.name,
        epochs=3,
        imgsz=640,
        batch=48,
    ).model_copy(update={"metadata": metadata})
    node = node.model_copy(update={"command_spec": train_spec, "command": train_spec.display()})
    config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=data_yaml,
        imgsz=640,
        budget_profile="pilot",
        batch_tuning=BatchTuningConfig(enabled=False),
        data_cache_policy=DataCachePolicyConfig(enabled=False),
        coco_post_eval=CocoPostEvalConfig(enabled=True),
    )

    calls = {"train": 0, "val": 0}

    class FakePopen:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            self.args = argv
            self.returncode = 0
            if "train" in argv:
                calls["train"] += 1
                output = f"Results saved to {training_run_dir}\n"
            elif "val" in argv:
                calls["val"] += 1
                post_eval_dir = training_run_dir / "coco_post_eval"
                post_eval_dir.mkdir(parents=True, exist_ok=True)
                (post_eval_dir / "predictions.json").write_text(
                    json.dumps([{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]),
                    encoding="utf-8",
                )
                output = "validation complete\n"
            else:
                self.returncode = 1
                output = ""
            self.stdout = StringIO(output)

        def poll(self) -> int:
            return self.returncode

        def __enter__(self) -> "FakePopen":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def communicate(self, input: object = None, timeout: int | float | None = None) -> tuple[str, str]:
            return self.stdout.read(), ""

        def wait(self, timeout: int | float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def fake_eval_report(*, annotations_path: Path, predictions_path: Path, output_path: Path) -> Path:
        output_path.write_text(
            json.dumps(
                {
                    "AP": 0.31,
                    "AP50": 0.50,
                    "AP_small": 0.20,
                    "AP_medium": 0.35,
                    "AP_large": 0.45,
                    "per_class_ap": {"bottle": 0.31},
                    "per_class_ar": {"bottle": 0.52},
                }
            ),
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(UltralyticsAdapter, "is_available", lambda self: True)
    monkeypatch.setattr(executor_mod, "_resolve_executable", lambda command: command)
    monkeypatch.setattr(executor_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(post_eval_mod, "write_coco_eval_report", fake_eval_report)

    store = EvidenceStore(tmp_path / "runs")
    executor = UltralyticsTrainExecutor(evidence_store=store, training_config=config, data_path=data_yaml)
    result = executor.execute(node, "run-1", train_spec)
    gate_result = PilotEvidenceCompletenessGate(store).evaluate(
        run_id="run-1",
        candidate_id="candidate",
        node_id=node.node_id,
        protocol_hash=protocol_hash,
    )

    assert result.status == "completed"
    assert result.metrics["coco_post_eval_complete"] is True
    assert gate_result.complete is True
    assert calls == {"train": 1, "val": 1}

    recovery_spec = CommandSpec(
        command_type="benchmark",
        command="yolo",
        argv=["yolo"],
        metadata={
            **metadata,
            "evidence_recovery_action": "coco_post_eval",
            "training_run_dir": training_run_dir.as_posix(),
            "data_yaml": data_yaml.as_posix(),
        },
    )
    recovery_node = node.model_copy(
        update={"command_spec": recovery_spec, "command": recovery_spec.display()}
    )
    recovery = executor.execute(recovery_node, "run-1", recovery_spec)

    assert recovery.status == "completed"
    assert calls == {"train": 1, "val": 1}
