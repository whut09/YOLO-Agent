"""Independent exposure policies for train-only paper mechanism adapters."""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.adapters.data_pipeline.contracts import DataSampleRecord
from yolo_agent.components.adapters.data_pipeline.sampling import bound_exposure


ExposureMechanism = Literal[
    "small_object_weighted_sampling",
    "class_balanced_sampling",
    "repeat_factor_sampling",
    "hard_negative_replay",
    "false_negative_class_boost",
]


class ExposureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mechanism: ExposureMechanism
    strength: float = Field(default=1.0, ge=0.0)
    max_weight: float = Field(default=3.0, ge=1.0)
    max_exposure_ratio: float = Field(default=3.0, ge=1.0)
    area_threshold: float = Field(default=0.01, gt=0.0, lt=1.0)
    repeat_threshold: float = Field(default=0.1, gt=0.0, le=1.0)
    target_class_ids: list[int] = Field(default_factory=list)
    sample_count: int | None = Field(default=None, ge=1)
    seed: int = Field(default=0, ge=0)
    imgsz: int = 640

    def model_post_init(self, __context: object) -> None:
        if self.imgsz != 640:
            raise ValueError("data exposure adapters require fixed imgsz=640")


def compute_exposure(
    records: list[DataSampleRecord],
    config: ExposureConfig,
) -> tuple[list[float], dict[str, int | float]]:
    """Compute one mechanism's bounded exposure without blending semantics."""
    _, final, statistics = compute_exposure_details(records, config)
    return final, statistics


def compute_exposure_details(
    records: list[DataSampleRecord],
    config: ExposureConfig,
) -> tuple[list[float], list[float], dict[str, int | float]]:
    """Return raw and bounded exposure for complete runtime manifests."""
    if not records:
        raise ValueError("data exposure requires at least one training record")
    if config.strength == 0:
        native = [1.0] * len(records)
        return native, native, _zero_statistics(len(records), config)
    counts = Counter(class_id for item in records for class_id in item.class_ids)
    image_frequency = {
        class_id: sum(class_id in set(item.class_ids) for item in records) / len(records)
        for class_id in counts
    }
    maximum = max(counts.values(), default=1)
    target_classes = set(config.target_class_ids)
    values: list[float] = []
    for record in records:
        signal = _signal(
            record,
            config=config,
            class_counts=counts,
            image_frequency=image_frequency,
            maximum_class_count=maximum,
            target_classes=target_classes,
        )
        values.append(1.0 + config.strength * max(signal, 0.0))
    final, statistics = bound_exposure(
        values,
        max_weight=config.max_weight,
        max_ratio=config.max_exposure_ratio,
    )
    return values, final, statistics


def _signal(
    record: DataSampleRecord,
    *,
    config: ExposureConfig,
    class_counts: Counter[int],
    image_frequency: dict[int, float],
    maximum_class_count: int,
    target_classes: set[int],
) -> float:
    mechanism = config.mechanism
    if mechanism == "small_object_weighted_sampling":
        areas = [area for area in record.normalized_areas if 0 < area <= 1]
        return (
            sum(area <= config.area_threshold for area in areas) / len(areas)
            if areas
            else 0.0
        )
    if mechanism == "class_balanced_sampling":
        if not record.class_ids:
            return 0.0
        rarest = min(class_counts[class_id] for class_id in record.class_ids)
        return math.sqrt(maximum_class_count / max(rarest, 1)) - 1.0
    if mechanism == "repeat_factor_sampling":
        factors = [
            math.sqrt(config.repeat_threshold / max(image_frequency[class_id], 1e-12))
            for class_id in set(record.class_ids)
        ]
        return max([1.0, *factors]) - 1.0
    if mechanism == "hard_negative_replay":
        return 1.0 if record.is_hard_negative else 0.0
    if mechanism == "false_negative_class_boost":
        overlap = target_classes.intersection(record.class_ids)
        return record.false_negative_score if overlap else 0.0
    raise AssertionError(f"unsupported exposure mechanism: {mechanism}")


def _zero_statistics(
    count: int,
    config: ExposureConfig,
) -> dict[str, int | float]:
    return {
        "clipped_count": 0,
        "clipped_fraction": 0.0,
        "raw_min": 1.0,
        "raw_max": 1.0,
        "final_min": 1.0,
        "final_max": 1.0,
        "max_weight": config.max_weight,
        "max_ratio": config.max_exposure_ratio,
        "zero_effect_count": count,
    }


__all__ = [
    "ExposureConfig",
    "ExposureMechanism",
    "compute_exposure",
    "compute_exposure_details",
]
