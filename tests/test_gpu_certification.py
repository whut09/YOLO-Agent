"""Offline tests for the opt-in real GPU certification orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.certification.fixture import create_mini_coco_fixture
from yolo_agent.certification.runner import BackendEvaluation, BackendRun, RealGpuAcceptanceSuite


class MockGpuBackend:
    def __init__(self) -> None:
        self.train_calls: list[tuple[str, int]] = []

    def environment(self) -> dict[str, object]:
        return {"cuda_available": True, "gpu_name": "mock-gpu", "ultralytics_version": "8.4.mock"}

    def train_entrypoint(self, *, data_yaml: Path, model: str, workdir: Path, device: str) -> list[str]:
        return ["yolo-agent", "train", "--data", str(data_yaml), "--model", model, "--dry-run"]

    def train(
        self,
        *,
        candidate_id: str,
        node_id: str,
        data_yaml: Path,
        model: str,
        workdir: Path,
        device: str,
        epochs: int,
        seed: int,
        overrides: dict[str, object],
    ) -> BackendRun:
        self.train_calls.append((node_id, epochs))
        run_dir = workdir / "mock_runs" / node_id
        checkpoint = run_dir / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"mock checkpoint")
        return BackendRun(
            candidate_id=candidate_id,
            node_id=node_id,
            run_dir=run_dir,
            checkpoint=checkpoint,
            command=["mock-train", node_id, str(epochs)],
        )

    def evaluate(self, *, run: BackendRun, data_yaml: Path, workdir: Path, device: str) -> BackendEvaluation:
        output = workdir / "mock_eval" / run.node_id
        output.mkdir(parents=True, exist_ok=True)
        baseline = run.candidate_id.startswith("baseline")
        gain = 0.0 if baseline else {
            "reduce_mosaic": 0.03,
            "increase_box_gain": 0.02,
            "reduce_cls_gain": 0.01,
        }.get(run.candidate_id, 0.0)
        eval_path = output / "coco_eval.json"
        eval_path.write_text(json.dumps({
            "AP": 0.30 + gain,
            "AP50": 0.50 + gain,
            "AP75": 0.28 + gain,
            "AP_small": 0.20 + gain,
            "AP_medium": 0.32 + gain,
            "AP_large": 0.40 + gain,
            "AR_small": 0.25 + gain,
            "per_class_ap": {"object": 0.30 + gain},
            "per_class_ar": {"object": 0.40 + gain},
        }), encoding="utf-8")
        predictions = output / "predictions.json"
        predictions.write_text("[]", encoding="utf-8")
        error_path = output / "coco_error_report.json"
        error_path.write_text(json.dumps({
            "false_negative_top_classes": [
                {"category_id": 1, "name": "object", "false_negative": 2 if baseline else 1, "recall": 0.5 + gain}
            ],
            "background_false_positive_top_classes": [
                {"category_id": 1, "name": "object", "background_false_positive": 1, "precision": 0.7 + gain}
            ],
            "localization_error_top_classes": [
                {"category_id": 1, "name": "object", "localization_error": 1, "ap50": 0.5 + gain}
            ],
        }), encoding="utf-8")
        return BackendEvaluation(
            eval_path=eval_path,
            predictions_path=predictions,
            error_report_path=error_path,
            latency_ms=10.0 + (0.0 if baseline else 0.1),
            model_size_mb=5.0,
            command=["mock-eval", run.node_id],
        )


def test_mini_coco_fixture_is_valid_and_deterministic(tmp_path: Path) -> None:
    data_yaml = create_mini_coco_fixture(tmp_path / "mini")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    annotations = json.loads((data_yaml.parent / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))

    assert config["train"] == "images/train2017"
    assert config["val"] == "images/val2017"
    assert len(list((data_yaml.parent / "images" / "train2017").glob("*.png"))) == 6
    assert len(annotations["images"]) == 4
    assert len(annotations["annotations"]) == 4


def test_suite_is_safe_without_explicit_gpu_opt_in(tmp_path: Path) -> None:
    backend = MockGpuBackend()
    report = RealGpuAcceptanceSuite(backend).run(workdir=tmp_path)

    assert report.status == "skipped"
    assert backend.train_calls == []
    assert (tmp_path / "certification_report.yaml").is_file()


def test_mock_backend_certifies_complete_mini_pipeline(tmp_path: Path) -> None:
    backend = MockGpuBackend()
    report = RealGpuAcceptanceSuite(backend).run(workdir=tmp_path, execute_real_gpu=True)

    assert report.status == "passed", report.failures
    assert report.asha_survivor == "reduce_mosaic"
    assert {stage.stage_id for stage in report.stages} >= {
        "train_entrypoint", "debug", "pilot_3_control", "post_eval", "error_facts",
        "paired_delta", "asha_decision", "pilot_10",
    }
    assert any(epochs == 10 for _, epochs in backend.train_calls)
    assert len(report.paired_result_hashes) == 4
    assert report.report_hash
