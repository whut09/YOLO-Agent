from __future__ import annotations

import json
from pathlib import Path

from yolo_agent.components.adapters.inference.artifacts import write_slicing_artifacts
from yolo_agent.components.adapters.inference.sliced_coco import evaluate_sliced_coco
from yolo_agent.components.adapters.inference.slicing import (
    SlicingInferenceConfig,
    SlicingInferenceMetrics,
    SlicingInferenceResult,
    protocol_from_config,
)


def test_sliced_coco_evaluator_renames_accuracy_namespace(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    annotations = tmp_path / "annotations.json"
    predictions = tmp_path / "predictions.json"
    annotations.write_text("{}", encoding="utf-8")
    predictions.write_text("[]", encoding="utf-8")

    def fake_eval(*, annotations_path, predictions_path, output_path):  # type: ignore[no-untyped-def]
        assert annotations_path == annotations
        assert predictions_path == predictions
        output_path.write_text(json.dumps({"AP": 0.41, "AP_small": 0.23}), encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "yolo_agent.components.adapters.inference.sliced_coco.write_coco_eval_report",
        fake_eval,
    )
    metrics = evaluate_sliced_coco(
        annotations_path=annotations,
        predictions_path=predictions,
    )

    assert metrics.model_dump() == {
        "sliced_map50_95": 0.41,
        "sliced_ap_small": 0.23,
        "inference_policy_changed": True,
    }


def test_artifacts_keep_standard_metric_namespace_absent(tmp_path: Path) -> None:
    result = SlicingInferenceResult(
        status="completed",
        protocol=protocol_from_config(
            SlicingInferenceConfig(model_path="yolo26n.pt", slice_width=512)
        ),
        metrics=SlicingInferenceMetrics(
            sliced_map50_95=0.41,
            sliced_ap_small=0.23,
            sliced_latency_ms=32.0,
            sliced_throughput=31.25,
        ),
        predictions=[{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}],
    )
    paths = write_slicing_artifacts(result, tmp_path / "artifacts")
    metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))

    assert set(metrics) == {
        "sliced_map50_95",
        "sliced_ap_small",
        "sliced_latency_ms",
        "sliced_throughput",
        "inference_policy_changed",
    }
    assert not {"map50_95", "ap_small", "latency_ms"}.intersection(metrics)
    assert json.loads(paths.predictions.read_text(encoding="utf-8"))[0]["image_id"] == 1
