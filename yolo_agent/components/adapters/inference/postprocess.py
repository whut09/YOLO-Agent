"""Framework-independent inference score and cross-view merge policies."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


Prediction = dict[str, Any]


def calibrate_confidence(predictions: Iterable[Prediction], temperature: float) -> list[Prediction]:
    """Apply temperature scaling to confidence probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    output: list[Prediction] = []
    for item in predictions:
        score = min(max(float(item.get("score", 0.0)), 1e-7), 1.0 - 1e-7)
        logit = math.log(score / (1.0 - score)) / temperature
        calibrated = 1.0 / (1.0 + math.exp(-logit))
        output.append({**item, "score": calibrated})
    return output


def apply_class_thresholds(
    predictions: Iterable[Prediction],
    thresholds: dict[int, float],
    *,
    default_threshold: float,
) -> list[Prediction]:
    """Filter predictions with explicit per-class validation thresholds."""
    return [
        dict(item)
        for item in predictions
        if float(item.get("score", 0.0))
        >= float(thresholds.get(int(item.get("category_id", -1)), default_threshold))
    ]


def merge_predictions(
    predictions: Iterable[Prediction],
    *,
    policy: str,
    iou_threshold: float,
    max_detections: int,
) -> tuple[list[Prediction], dict[str, int | float | str | bool]]:
    """Merge only cross-view predictions, grouped by image and class."""
    records = [dict(item) for item in predictions]
    if policy == "none":
        merged = _limit(records, max_detections)
    elif policy == "nms":
        merged = _group_merge(records, iou_threshold, _nms_group)
        merged = _limit(merged, max_detections)
    elif policy == "nmm":
        merged = _group_merge(records, iou_threshold, _nmm_group)
        merged = _limit(merged, max_detections)
    elif policy == "weighted_box_fusion":
        merged = _group_merge(records, iou_threshold, _wbf_group)
        merged = _limit(merged, max_detections)
    else:
        raise ValueError(f"unsupported merge policy: {policy}")
    return merged, {
        "merge_policy": policy,
        "input_count": len(records),
        "output_count": len(merged),
        "merged_count": len(records) - len(merged),
        "extra_nms_applied": policy == "nms",
    }


def bbox_iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(lw, 0.0) * max(lh, 0.0) + max(rw, 0.0) * max(rh, 0.0) - intersection
    return intersection / union if union > 0.0 else 0.0


def _group_merge(
    predictions: list[Prediction],
    threshold: float,
    merger: Any,
) -> list[Prediction]:
    grouped: dict[tuple[int, int], list[Prediction]] = defaultdict(list)
    for item in predictions:
        grouped[(int(item.get("image_id", -1)), int(item.get("category_id", -1)))].append(item)
    return [item for group in grouped.values() for item in merger(group, threshold)]


def _nms_group(group: list[Prediction], threshold: float) -> list[Prediction]:
    remaining = sorted(group, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    kept: list[Prediction] = []
    while remaining:
        selected = remaining.pop(0)
        kept.append(selected)
        remaining = [
            item for item in remaining
            if bbox_iou(selected["bbox"], item["bbox"]) < threshold
        ]
    return kept


def _nmm_group(group: list[Prediction], threshold: float) -> list[Prediction]:
    return _fuse_groups(group, threshold, weighted=False)


def _wbf_group(group: list[Prediction], threshold: float) -> list[Prediction]:
    return _fuse_groups(group, threshold, weighted=True)


def _fuse_groups(
    group: list[Prediction], threshold: float, *, weighted: bool
) -> list[Prediction]:
    remaining = sorted(group, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    output: list[Prediction] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        unmatched: list[Prediction] = []
        for item in remaining:
            if bbox_iou(seed["bbox"], item["bbox"]) >= threshold:
                cluster.append(item)
            else:
                unmatched.append(item)
        remaining = unmatched
        weights = [float(item.get("score", 0.0)) if weighted else 1.0 for item in cluster]
        total = sum(weights) or float(len(cluster))
        bbox = [
            sum(float(item["bbox"][index]) * weight for item, weight in zip(cluster, weights, strict=True)) / total
            for index in range(4)
        ]
        output.append(
            {
                **seed,
                "bbox": bbox,
                "score": max(float(item.get("score", 0.0)) for item in cluster),
            }
        )
    return output


def _limit(predictions: list[Prediction], maximum: int) -> list[Prediction]:
    grouped: dict[int, list[Prediction]] = defaultdict(list)
    for item in predictions:
        grouped[int(item.get("image_id", -1))].append(item)
    return [
        item
        for group in grouped.values()
        for item in sorted(group, key=lambda row: float(row.get("score", 0.0)), reverse=True)[:maximum]
    ]


__all__ = [
    "Prediction",
    "apply_class_thresholds",
    "bbox_iou",
    "calibrate_confidence",
    "merge_predictions",
]
