"""Prompt-18I-v2 §7: full synthetic error matrix, deterministic, no GPU.

One fixture scene exercises every error family the profile decomposes,
with hand-computed expectations:

* image 1: small GT (area 400) matched -> TP; medium GT (area 1600) missed -> FN
* image 2: medium-large GT (area 6300 < 96^2) matched once (TP) then duplicated -> duplicate FP
* image 3: person GT overlapped by a car prediction at IoU 0.6 -> class-confusion FP
* image 4: prediction over empty background -> background FP
* confidence bins follow the 0.1-wide histogram convention
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
    "images": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
    "annotations": [
        # image 1: small person (TP), medium person (FN)
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [60, 60, 40, 40], "area": 1600},
        # image 2: car matched then duplicated (area 6300 -> medium bucket)
        {"id": 3, "image_id": 2, "category_id": 2, "bbox": [200, 200, 90, 70], "area": 6300},
        # image 3: person missed but overlapped by a car prediction (confusion)
        {"id": 4, "image_id": 3, "category_id": 1, "bbox": [50, 50, 40, 40], "area": 1600},
        # image 4: car missed entirely (area 8000 -> still medium bucket)
        {"id": 5, "image_id": 4, "category_id": 2, "bbox": [300, 300, 100, 80], "area": 8000},
    ],
}

PREDICTIONS = [
    # image 1: small person TP at 0.9
    {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
    # image 2: car TP at 0.8 then duplicate at 0.55
    {"image_id": 2, "category_id": 2, "bbox": [202, 202, 90, 70], "score": 0.8},
    {"image_id": 2, "category_id": 2, "bbox": [201, 201, 90, 70], "score": 0.55},
    # image 3: car prediction on person GT (IoU 0.6 >= 0.5, wrong class) at 0.7
    {"image_id": 3, "category_id": 2, "bbox": [52, 52, 40, 40], "score": 0.7},
    # image 4: car prediction over background at 0.85
    {"image_id": 4, "category_id": 2, "bbox": [10, 10, 40, 40], "score": 0.85},
]


@pytest.fixture()
def profile(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "gt.json").write_text(json.dumps(GT))
    (data / "preds.json").write_text(json.dumps(PREDICTIONS))
    return build_detection_error_profile(
        ErrorProfileSource(
            gt_json=data / "gt.json",
            predictions_json=data / "preds.json",
            run_id="run-1",
            candidate_id="cand",
            split="val",
        )
    )


def test_false_negatives_decomposed_by_class_and_scale(profile) -> None:
    assert profile.false_negative.total == 3
    assert profile.false_negative.by_class == {"person": 2, "car": 1}
    # All missed areas sit below 96*96, so every FN is a medium-bucket miss.
    assert profile.false_negative.by_scale == {"medium": 3}
    # Best overlapping scores for the missed GTs: only image 3's person is
    # overlapped (by the 0.7 car prediction); the other two have no
    # overlapping prediction at all.
    assert profile.false_negative.by_confidence == {"0.7-0.8": 1, "none": 2}


def test_false_positive_kinds_are_separated(profile) -> None:
    fp = profile.false_positive
    assert fp.total == 3
    assert fp.background_fp == 1
    assert fp.duplicate_fp == 1
    assert fp.class_confusion_fp == 1
    # High confidence = score >= 0.5 here: all three qualify.
    assert fp.high_confidence_fp == 3


def test_classification_confusion_pair(profile) -> None:
    assert profile.classification.confusion_matrix == {"person->car": 1}
    assert profile.classification.top_confusion_pairs[0][0] == "person->car"


def test_localization_metrics(profile) -> None:
    loc = profile.localization
    # Two TPs: exact IoU 1.0 and IoU ~0.9997 both land in the 0.9-1.0 bin.
    assert sum(loc.matched_iou_distribution.values()) == 2
    assert "0.9-1.0" in loc.matched_iou_distribution
    assert loc.mean_matched_iou is not None and loc.mean_matched_iou > 0.9
    # No TP survives at IoU >= 0.75, so the strict error count is 0.
    assert loc.localization_error_count == 0


def test_scale_ap_and_recall(profile) -> None:
    scale = profile.scale
    # Small bucket: 1 GT, 1 TP -> AP 1.0, recall 1.0.
    assert scale.ap_small == pytest.approx(1.0)
    assert scale.recall_small == pytest.approx(1.0)
    # Medium bucket: 4 GTs (2 person + 2 car), 1 TP -> AP 0.125, recall 0.25.
    assert scale.ap_medium == pytest.approx(0.125)
    assert scale.recall_medium == pytest.approx(0.25)
    # Large bucket: no GT at all -> unmeasured.
    assert scale.ap_large is None
    assert scale.recall_large is None


def test_confidence_bins_separate_tp_from_fp(profile) -> None:
    conf = profile.confidence
    # TPs at 0.9 and 0.8 land in the 0.8-0.9 and 0.9-1.0 bins.
    assert conf.tp_confidence_histogram == {"0.8-0.9": 1, "0.9-1.0": 1}
    # FPs at 0.55, 0.7, 0.85.
    assert conf.fp_confidence_histogram == {"0.5-0.6": 1, "0.7-0.8": 1, "0.8-0.9": 1}
    assert conf.expected_calibration_error is not None
    assert len(conf.calibration_bins) >= 3


def test_deterministic_content(tmp_path: Path) -> None:
    """Same inputs -> byte-identical content (paths aside, which differ by dir)."""
    outputs = []
    for name in ("a", "b"):
        data = tmp_path / name
        data.mkdir()
        (data / "gt.json").write_text(json.dumps(GT))
        (data / "preds.json").write_text(json.dumps(PREDICTIONS))
        profile = build_detection_error_profile(
            ErrorProfileSource(
                gt_json=data / "gt.json",
                predictions_json=data / "preds.json",
                run_id="run-1",
                candidate_id="cand",
                split="val",
            )
        )
        payload = profile.model_dump(mode="json")
        payload.pop("gt_artifact")
        payload.pop("predictions_artifact")
        outputs.append(json.dumps(payload, sort_keys=True))
    assert outputs[0] == outputs[1]
