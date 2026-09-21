"""Prompt-18I: DetectionErrorProfile schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yolo_agent.core.detection_error_profile import (
    ConfidenceFacts,
    DetectionErrorProfile,
    FalseNegativeFacts,
    FalsePositiveFacts,
    GlobalMetrics,
    LocalizationFacts,
    PerClassMetrics,
    PROFILE_SCHEMA_VERSION,
    ScaleMetrics,
)


def _profile(**overrides: object) -> DetectionErrorProfile:
    defaults: dict[str, object] = {
        "profile_id": "prof-1",
        "run_id": "run-1",
        "candidate_id": "baseline",
    }
    defaults.update(overrides)
    return DetectionErrorProfile(**defaults)  # type: ignore[arg-type]


def test_profile_covers_all_nine_sections() -> None:
    profile = _profile(
        global_=GlobalMetrics(map50=0.5, map50_95=0.35, precision=0.6, recall=0.55),
        scale=ScaleMetrics(ap_small=0.3, ap_medium=0.5, ap_large=0.6),
        per_class=[
            PerClassMetrics(category_id=1, name="person", ap=0.6, precision=0.7, recall=0.5, support=120)
        ],
        false_negative=FalseNegativeFacts(total=8, by_scale={"small": 6, "medium": 2}),
        false_positive=FalsePositiveFacts(total=10, background_fp=6, duplicate_fp=2, class_confusion_fp=2),
        localization=LocalizationFacts(localization_error_count=5, ap50_vs_ap75_gap=0.12),
        confidence=ConfidenceFacts(expected_calibration_error=0.08),
    )
    assert profile.global_.map50 == 0.5
    assert profile.scale.ap_small == 0.3
    assert profile.per_class[0].support == 120
    assert profile.false_negative.by_scale["small"] == 6
    assert profile.false_positive.background_fp == 6
    assert profile.localization.ap50_vs_ap75_gap == 0.12
    assert profile.confidence.expected_calibration_error == 0.08
    assert profile.schema_version == PROFILE_SCHEMA_VERSION


def test_yaml_round_trip_uses_global_alias() -> None:
    profile = _profile(global_=GlobalMetrics(map50=0.4))
    payload = profile.to_yaml_dict()
    assert "global" in payload and "global_" not in payload
    assert payload["schema_version"] == PROFILE_SCHEMA_VERSION


def test_scene_slices_default_empty_and_record_unavailable() -> None:
    profile = _profile(unavailable_scene_slices=["day_night", "weather"])
    assert profile.scene_slices == []
    assert profile.unavailable_scene_slices == ["day_night", "weather"]


def test_profile_requires_identity() -> None:
    with pytest.raises(ValidationError):
        DetectionErrorProfile()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        _profile(run_id="")  # type: ignore[dict-item]


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        _profile(some_unknown_field=1)  # type: ignore[call-arg]
