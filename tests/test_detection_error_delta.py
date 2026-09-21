"""Prompt-18I: error delta tests on deterministic synthetic profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_agent.core.detection_error_delta import (
    DetectionErrorDelta,
    build_detection_error_delta,
)
from yolo_agent.core.detection_error_profile import ErrorProfileSource
from yolo_agent.core.detection_error_profile_builder import (
    build_detection_error_profile,
)

GT = {
    "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
    "images": [{"id": 1}, {"id": 2}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 100, 100, 80], "area": 8000},
        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [50, 50, 40, 40], "area": 1600},
        {"id": 4, "image_id": 2, "category_id": 2, "bbox": [200, 200, 90, 70], "area": 6300},
    ],
}


def _profile(tmp_path: Path, name: str, predictions: list[dict], **source_kwargs: object):
    gt_path = tmp_path / "gt.json"  # shared GT: matched evaluation semantics
    if not gt_path.exists():
        gt_path.write_text(json.dumps(GT))
    (tmp_path / f"preds_{name}.json").write_text(json.dumps(predictions))
    source = ErrorProfileSource(
        gt_json=gt_path,
        predictions_json=tmp_path / f"preds_{name}.json",
        run_id="run-1",
        candidate_id=name,
        **source_kwargs,  # type: ignore[arg-type]
    )
    return build_detection_error_profile(source)


PARENT_PREDS = [
    {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
]

# Candidate finds the small person too (same TP) but adds two FPs: accuracy
# gain (more matched GT => higher recall) with a precision regression.
CANDIDATE_PREDS = [
    {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
    {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.85},  # duplicate
    {"image_id": 1, "category_id": 2, "bbox": [500, 500, 50, 50], "score": 0.8},  # background
]


@pytest.fixture()
def delta(tmp_path: Path) -> DetectionErrorDelta:
    parent = _profile(tmp_path, "parent", PARENT_PREDS)
    candidate = _profile(tmp_path, "cand", CANDIDATE_PREDS)
    return build_detection_error_delta(candidate, parent)


def test_pinned_fields(delta: DetectionErrorDelta) -> None:
    assert delta.schema_version == "detection_error_delta.v1"
    assert delta.parent_candidate_id == "parent"
    assert delta.candidate_id == "cand"
    assert delta.matched_evaluation is True


def test_precision_and_fp_regressions_are_named(delta: DetectionErrorDelta) -> None:
    regressions = delta.regressions()
    assert "global.precision" in regressions
    assert "false_positive.total" in regressions
    assert "false_positive.background_fp" in regressions
    assert "false_positive.duplicate_fp" in regressions
    assert delta.has_regressions() is True


def test_recall_and_fn_improvements_are_named(delta: DetectionErrorDelta) -> None:
    # Same single TP and same GT set => recall unchanged; FN total unchanged.
    assert delta.metric("global", "recall").verdict == "unchanged"
    assert delta.section("false_negative").counts[0].delta == 0


def test_mismatched_evaluation_is_flagged(tmp_path: Path) -> None:
    parent = _profile(tmp_path, "parent", PARENT_PREDS)
    candidate = _profile(
        tmp_path, "cand", CANDIDATE_PREDS, dataset_manifest_hash="other-dataset"
    )
    combined = build_detection_error_delta(candidate, parent)
    assert combined.matched_evaluation is False


def test_null_metrics_are_not_comparable(tmp_path: Path) -> None:
    profile = _profile(tmp_path, "solo", PARENT_PREDS)
    # No official metrics anywhere => map50_95 is None on both sides.
    single = build_detection_error_delta(profile, profile)
    assert single.metric("global", "map50_95").verdict == "not_comparable"


def test_delta_sections_cover_the_contract(delta: DetectionErrorDelta) -> None:
    assert [s.section for s in delta.sections] == [
        "global",
        "scale",
        "per_class",
        "false_negative",
        "false_positive",
        "localization",
        "confidence",
    ]
