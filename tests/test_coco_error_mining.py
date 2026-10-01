"""COCO error mining tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.cli import main
from yolo_agent.tools.coco_error_mining import confidence_iou_correlation, mine_coco_errors


def _write_coco_gt(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 1}, {"id": 2}, {"id": 3}],
                "categories": [
                    {"id": 1, "name": "person"},
                    {"id": 2, "name": "dog"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400},
                    {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 100, 80, 80], "area": 6400},
                    {"id": 3, "image_id": 2, "category_id": 1, "bbox": [30, 30, 20, 20], "area": 400},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_predictions(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99},
                {"image_id": 1, "category_id": 1, "bbox": [100, 100, 80, 80], "score": 0.95},
                {"image_id": 2, "category_id": 1, "bbox": [40, 40, 20, 20], "score": 0.9},
                {"image_id": 3, "category_id": 2, "bbox": [5, 5, 20, 20], "score": 0.8},
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_mine_coco_errors_outputs_loop_observations(tmp_path: Path) -> None:
    """Miner should summarize FN, confusion, localization, and background errors."""
    gt_path = _write_coco_gt(tmp_path / "instances_val2017.json")
    predictions_path = _write_predictions(tmp_path / "predictions.json")

    report = mine_coco_errors(gt_path, predictions_path, tmp_path / "coco_errors")

    assert report.area_recall["small"] < 1.0
    assert report.false_negative_top_classes[0].false_negative >= 1
    assert report.class_confusion_pairs == {"dog->person": 1}
    assert report.background_false_positive_top_classes[0].background_false_positive >= 1
    assert {observation.error_type for observation in report.observations} >= {
        "small_object_miss",
        "background_confusion",
        "class_confusion",
    }
    errors_yaml = yaml.safe_load((tmp_path / "coco_errors_errors.yaml").read_text(encoding="utf-8"))
    assert errors_yaml["errors"]


def test_mine_coco_errors_cli_writes_reports(tmp_path: Path) -> None:
    """CLI should write report artifacts and diagnose-compatible error YAML."""
    gt_path = _write_coco_gt(tmp_path / "instances_val2017.json")
    predictions_path = _write_predictions(tmp_path / "predictions.json")
    out_prefix = tmp_path / "reports" / "coco_error_report"

    assert main(
        [
            "mine-coco-errors",
            "--gt",
            str(gt_path),
            "--predictions",
            str(predictions_path),
            "--out",
            str(out_prefix),
        ]
    ) == 0

    assert (tmp_path / "reports" / "coco_error_report.json").exists()
    assert (tmp_path / "reports" / "coco_error_report.md").exists()
    assert (tmp_path / "reports" / "coco_error_report_errors.yaml").exists()


def test_confidence_iou_correlation_tracks_score_iou_agreement(tmp_path: Path) -> None:
    """fix 22: the calibration metric separates aligned from anti-correlated detectors."""
    annotations = {
        "categories": [{"id": 1, "name": "person"}],
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
            {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
        ],
    }
    # Detector A: high score on the tight match, low score on the loose match.
    aligned = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 5], "score": 0.2},
    ]
    # Detector B: the reverse assignment.
    anti = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 5], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.2},
    ]
    annotations_path = tmp_path / "instances.json"
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    aligned_path = tmp_path / "aligned.json"
    aligned_path.write_text(json.dumps(aligned), encoding="utf-8")
    anti_path = tmp_path / "anti.json"
    anti_path.write_text(json.dumps(anti), encoding="utf-8")

    aligned_r = confidence_iou_correlation(annotations_path, aligned_path)
    anti_r = confidence_iou_correlation(annotations_path, anti_path)

    assert aligned_r is not None and aligned_r > 0.9
    assert anti_r is not None and anti_r < -0.9


def test_confidence_iou_correlation_returns_none_without_true_positives(tmp_path: Path) -> None:
    annotations = {
        "categories": [{"id": 1, "name": "person"}],
        "images": [{"id": 1}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0}],
    }
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [100, 100, 10, 10], "score": 0.9}]
    annotations_path = tmp_path / "instances.json"
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")

    assert confidence_iou_correlation(annotations_path, predictions_path) is None
