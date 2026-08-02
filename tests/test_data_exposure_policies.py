from __future__ import annotations

import pytest

from yolo_agent.components.adapters.data_pipeline import (
    DataSampleRecord,
    ExposureConfig,
    compute_exposure,
    compute_exposure_details,
)


def _records() -> list[DataSampleRecord]:
    return [
        DataSampleRecord(
            image_path="small-rare.jpg",
            normalized_areas=[0.002],
            class_ids=[2],
        ),
        DataSampleRecord(
            image_path="common.jpg",
            normalized_areas=[0.2, 0.3],
            class_ids=[1, 1],
        ),
        DataSampleRecord(
            image_path="hard-negative.jpg",
            is_hard_negative=True,
        ),
        DataSampleRecord(
            image_path="fn-class.jpg",
            class_ids=[3],
            false_negative_score=0.9,
        ),
    ]


@pytest.mark.parametrize(
    ("mechanism", "boosted_index", "options"),
    [
        ("small_object_weighted_sampling", 0, {}),
        ("class_balanced_sampling", 0, {}),
        ("repeat_factor_sampling", 0, {"repeat_threshold": 0.8}),
        ("hard_negative_replay", 2, {}),
        (
            "false_negative_class_boost",
            3,
            {"target_class_ids": [3]},
        ),
    ],
)
def test_exposure_mechanisms_change_the_intended_distribution(
    mechanism: str,
    boosted_index: int,
    options: dict[str, object],
) -> None:
    exposure, _ = compute_exposure(
        _records(),
        ExposureConfig(mechanism=mechanism, **options),  # type: ignore[arg-type]
    )

    assert exposure[boosted_index] > min(exposure)


@pytest.mark.parametrize(
    "mechanism",
    [
        "small_object_weighted_sampling",
        "class_balanced_sampling",
        "repeat_factor_sampling",
        "hard_negative_replay",
        "false_negative_class_boost",
    ],
)
def test_zero_effect_is_exact_native_exposure_equivalence(mechanism: str) -> None:
    exposure, statistics = compute_exposure(
        _records(),
        ExposureConfig(mechanism=mechanism, strength=0),  # type: ignore[arg-type]
    )

    assert exposure == [1.0] * len(_records())
    assert statistics["clipped_count"] == 0


def test_manifest_details_keep_raw_and_bounded_exposure_separate() -> None:
    raw, final, statistics = compute_exposure_details(
        _records(),
        ExposureConfig(
            mechanism="hard_negative_replay",
            strength=10,
            max_weight=2,
        ),
    )

    assert max(raw) > max(final)
    assert max(final) == 2
    assert statistics["clipped_count"] == 1
