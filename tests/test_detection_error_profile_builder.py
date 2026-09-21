"""Prompt-18I: error profile builder tests on deterministic synthetic COCO.

Every number is hand-computed from the synthetic GT/predictions below.
No GPU, no training, no real dataset required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_agent.core.detection_error_profile import ErrorProfileSource
from yolo_agent.core.detection_error_profile_builder import (
    build_detection_error_profile,
)

GT = {
    "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
    "images": [{"id": 1}, {"id": 2}],
    "annotations": [
        # image 1: small person matched; large car missed
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 100, 100, 80], "area": 8000},
        # image 2: medium person matched only by a car prediction (confusion);
        # large car matched once then duplicated
        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [50, 50, 40, 40], "area": 1600},
        {"id": 4, "image_id": 2, "category_id": 2, "bbox": [200, 200, 90, 70], "area": 6300},
    ],
}

PREDICTIONS = [
    {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},   # TP GT1
    {"image_id": 1, "category_id": 2, "bbox": [500, 500, 50, 50], "score": 0.8},  # background FP
    {"image_id": 2, "category_id": 2, "bbox": [52, 52, 40, 40], "score": 0.7},   # class confusion FP (car on person GT3)
    {"image_id": 2, "category_id": 2, "bbox": [202, 202, 90, 70], "score": 0.6},  # TP GT4
    {"image_id": 2, "category_id": 2, "bbox": [201, 201, 90, 70], "score": 0.55},  # duplicate FP (GT4 taken)
]


@pytest.fixture()
def artifacts(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "gt.json").write_text(json.dumps(GT))
    (tmp_path / "preds.json").write_text(json.dumps(PREDICTIONS))
    return tmp_path / "gt.json", tmp_path / "preds.json"


def _build(artifacts: tuple[Path, Path], **kwargs: object):
    gt, preds = artifacts
    source = ErrorProfileSource(
        gt_json=gt, predictions_json=preds, run_id="run-1", candidate_id="cand-1", **kwargs  # type: ignore[arg-type]
    )
    return build_detection_error_profile(source)


def test_false_negative_decomposition(artifacts: tuple[Path, Path]) -> None:
    profile = _build(artifacts)
    fn = profile.false_negative
    assert fn.total == 2  # GT2 (large car, never predicted) + GT3 (person, only car prediction)
    assert fn.by_scale == {"medium": 2}  # both 1600-area boxes are medium
    assert fn.by_class == {"car": 1, "person": 1}
    # GT3 has an overlapping 0.7-score prediction; GT2 has nothing near it.
    assert fn.by_confidence.get("0.7-0.8") == 1
    assert fn.by_confidence.get("none") == 1


def test_false_positive_kinds(artifacts: tuple[Path, Path]) -> None:
    profile = _build(artifacts)
    fp = profile.false_positive
    assert fp.total == 3
    assert fp.background_fp == 1  # empty region prediction
    assert fp.class_confusion_fp == 1  # car box on the person GT3
    assert fp.duplicate_fp == 1  # second box on GT4
    assert fp.high_confidence_fp == 3  # all FPs score >= 0.5
    assert fp.by_class == {"car": 3}


def test_classification_confusion_pairs(artifacts: tuple[Path, Path]) -> None:
    profile = _build(artifacts)
    assert profile.classification.top_confusion_pairs == [("person->car", 1)]
    assert profile.classification.confusion_matrix == {"person->car": 1}


def test_scale_metrics_restrict_to_bucket_gt(artifacts: tuple[Path, Path]) -> None:
    profile = _build(artifacts)
    # Only GT1 is small and matched -> small AP = 1.0.
    assert profile.scale.ap_small == 1.0
    # Medium bucket has GT3 (person) unmatched -> per-class medium AP < 1.
    assert profile.scale.ap_medium is not None and profile.scale.ap_medium < 0.5


def test_localization_and_calibration(artifacts: tuple[Path, Path]) -> None:
    profile = _build(artifacts)
    assert profile.localization.matched_iou_distribution
    # Both TPs are near-perfect boxes (>= 0.75 IoU):
    assert profile.localization.localization_error_count == 0
    assert profile.confidence.tp_confidence_histogram
    assert profile.confidence.fp_confidence_histogram
    assert profile.confidence.expected_calibration_error is not None


def test_scene_slices_from_metadata_not_fabricated(tmp_path: Path) -> None:
    (tmp_path / "gt.json").write_text(json.dumps(GT))
    (tmp_path / "preds.json").write_text(json.dumps(PREDICTIONS))
    metadata = {
        "1": {"day_night": "day", "weather": "clear"},
        "2": {"day_night": "night", "weather": "rain"},
    }
    (tmp_path / "meta.json").write_text(json.dumps(metadata))
    profile = _build(
        (tmp_path / "gt.json", tmp_path / "preds.json"),
        image_metadata_json=tmp_path / "meta.json",
    )
    names = {s.slice_name for s in profile.scene_slices}
    assert "day_night=day" in names and "day_night=night" in names
    assert "weather=clear" in names and "weather=rain" in names
    night = next(s for s in profile.scene_slices if s.slice_name == "day_night=night")
    # Slice recall is geometric: both night GTs sit under a high-IoU box
    # (GT3's box is covered by the car prediction), so tp == 2 even though
    # one is a class-confusion error — the class-aware story lives in the
    # FN/FP sections, the slice tracks where the model looks at all.
    assert night.gt_count == 2 and night.true_positives == 2
    # FN slice attribution flows through: the missed person GT is on the
    # night image, the missed car GT on the day image.
    assert profile.false_negative.by_scene_slice.get("day_night=night") == 1
    assert profile.false_negative.by_scene_slice.get("day_night=day") == 1


def test_no_metadata_means_no_slices_and_recorded_absence(
    artifacts: tuple[Path, Path],
) -> None:
    profile = _build(artifacts)
    assert profile.scene_slices == []
    assert set(profile.unavailable_scene_slices) == {
        "day_night",
        "indoor_outdoor",
        "weather",
        "distance",
        "camera",
        "domain",
    }


def test_official_metrics_override_estimates(tmp_path: Path) -> None:
    (tmp_path / "gt.json").write_text(json.dumps(GT))
    (tmp_path / "preds.json").write_text(json.dumps(PREDICTIONS))
    (tmp_path / "official.json").write_text(
        json.dumps({"map50": 0.512, "map50_95": 0.334, "precision": 0.61, "recall": 0.58})
    )
    profile = _build(
        (tmp_path / "gt.json", tmp_path / "preds.json"),
        official_metrics_json=tmp_path / "official.json",
    )
    assert profile.global_.map50 == 0.512
    assert profile.global_.map50_95 == 0.334
    assert profile.global_.precision == 0.61
    assert profile.global_.recall == 0.58


def test_deterministic_bytes(artifacts: tuple[Path, Path]) -> None:
    first = _build(artifacts).to_yaml_dict()
    second = _build(artifacts).to_yaml_dict()
    assert first == second
