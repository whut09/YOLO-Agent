from __future__ import annotations

import json
from pathlib import Path

from yolo_agent.core.paired_bootstrap import (
    PairedBootstrapConfig,
    PairedBootstrapReport,
    paired_bootstrap_coco_predictions,
)


def _write_fixture(tmp_path: Path, *, images: int = 24) -> tuple[Path, Path, Path]:
    ground_truth = {
        "images": [{"id": image_id} for image_id in range(1, images + 1)],
        "categories": [{"id": 1, "name": "bottle"}, {"id": 2, "name": "person"}],
        "annotations": [],
    }
    baseline: list[dict[str, object]] = []
    candidate: list[dict[str, object]] = []
    annotation_id = 1
    for image_id in range(1, images + 1):
        for category_id, offset in ((1, 0), (2, 40)):
            bbox = [offset, 0, 20, 20]
            ground_truth["annotations"].append(
                {
                    "id": annotation_id, "image_id": image_id,
                    "category_id": category_id, "bbox": bbox,
                    "area": 400, "iscrowd": 0,
                }
            )
            annotation_id += 1
            if category_id == 1:
                if image_id <= 6:
                    baseline.append({"image_id": image_id, "category_id": 1, "bbox": bbox, "score": 0.9})
                candidate.append({"image_id": image_id, "category_id": 1, "bbox": bbox, "score": 0.9})
            else:
                baseline.append({"image_id": image_id, "category_id": 2, "bbox": bbox, "score": 0.9})
                if image_id <= 5:
                    candidate.append({"image_id": image_id, "category_id": 2, "bbox": bbox, "score": 0.9})
    gt_path = tmp_path / "instances_val.json"
    baseline_path = tmp_path / "baseline_predictions.json"
    candidate_path = tmp_path / "candidate_predictions.json"
    gt_path.write_text(json.dumps(ground_truth), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    return gt_path, baseline_path, candidate_path


def test_detects_stable_class_improvement_and_regression(tmp_path: Path) -> None:
    gt_path, baseline_path, candidate_path = _write_fixture(tmp_path)
    report = paired_bootstrap_coco_predictions(
        gt_path, baseline_path, candidate_path,
        config=PairedBootstrapConfig(iterations=400, minimum_images=20, random_seed=7),
    )
    assert report.status == "completed"
    assert report.matched_image_count == 24
    assert report.single_seed_only is True
    assert report.stable_improved_classes == ["bottle"]
    assert report.stable_regressed_classes == ["person"]
    by_name = {item.category_name: item for item in report.classes}
    assert by_name["bottle"].confidence_interval_low > 0
    assert by_name["person"].confidence_interval_high < 0
    assert report.overall is not None
    assert report.overall.direction == "inconclusive"


def test_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    gt_path, baseline_path, candidate_path = _write_fixture(tmp_path)
    config = PairedBootstrapConfig(iterations=200, minimum_images=20, random_seed=19)
    first = paired_bootstrap_coco_predictions(gt_path, baseline_path, candidate_path, config=config)
    second = paired_bootstrap_coco_predictions(gt_path, baseline_path, candidate_path, config=config)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    output = first.to_json(tmp_path / "paired_bootstrap.json")
    loaded = PairedBootstrapReport.model_validate_json(output.read_text(encoding="utf-8"))
    assert loaded.protocol_hash == first.protocol_hash


def test_blocks_predictions_outside_ground_truth(tmp_path: Path) -> None:
    gt_path, baseline_path, candidate_path = _write_fixture(tmp_path)
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload.append({"image_id": 999, "category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.9})
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")
    report = paired_bootstrap_coco_predictions(
        gt_path, baseline_path, candidate_path,
        config=PairedBootstrapConfig(iterations=100, minimum_images=20),
    )
    assert report.status == "blocked"
    assert "predictions_outside_ground_truth:baseline=0,candidate=1" in report.warnings


def test_blocks_too_small_image_set(tmp_path: Path) -> None:
    gt_path, baseline_path, candidate_path = _write_fixture(tmp_path, images=4)
    report = paired_bootstrap_coco_predictions(
        gt_path, baseline_path, candidate_path,
        config=PairedBootstrapConfig(iterations=100, minimum_images=20),
    )
    assert report.status == "blocked"
    assert "minimum_images_not_met:4/20" in report.warnings
