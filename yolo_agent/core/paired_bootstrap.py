"""Paired image-level bootstrap for COCO prediction diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

BootstrapDirection = Literal["stable_improvement", "inconclusive", "stable_regression"]


class PairedBootstrapConfig(BaseModel):
    schema_version: str = "1.0"
    iterations: int = Field(default=1000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    random_seed: int = 20260716
    iou_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    score_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    minimum_images: int = Field(default=20, ge=1)
    minimum_effect: float = Field(default=0.0, ge=0.0)
    maximum_predictions_per_image: int = Field(default=100, ge=1)


class PairedBootstrapMetric(BaseModel):
    metric_name: str
    baseline_value: float
    candidate_value: float
    observed_delta: float
    confidence_interval_low: float
    confidence_interval_high: float
    probability_improvement: float = Field(ge=0.0, le=1.0)
    direction: BootstrapDirection


class PairedBootstrapClassResult(PairedBootstrapMetric):
    category_id: int
    category_name: str
    ground_truth_count: int = Field(ge=0)


class PairedBootstrapReport(BaseModel):
    schema_version: str = "1.0"
    status: Literal["completed", "blocked"]
    evidence_kind: Literal["diagnostic_paired_bootstrap_ap50"] = "diagnostic_paired_bootstrap_ap50"
    baseline_predictions: Path
    candidate_predictions: Path
    ground_truth: Path
    baseline_predictions_sha256: str
    candidate_predictions_sha256: str
    ground_truth_sha256: str
    matched_image_count: int = Field(ge=0)
    protocol_hash: str
    config: PairedBootstrapConfig
    overall: PairedBootstrapMetric | None = None
    classes: list[PairedBootstrapClassResult] = Field(default_factory=list)
    stable_improved_classes: list[str] = Field(default_factory=list)
    stable_regressed_classes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    single_seed_only: bool = True

    @model_validator(mode="after")
    def validate_completed_report(self) -> "PairedBootstrapReport":
        if self.status == "completed" and self.overall is None:
            raise ValueError("completed report requires overall metrics")
        return self

    def to_json(self, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(output)
        return output


class _GroundTruth(BaseModel):
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]


class _Prediction(BaseModel):
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float


class _Event(BaseModel):
    image_id: int
    score: float
    true_positive: int
    false_positive: int


class _PreparedClass(BaseModel):
    category_id: int
    category_name: str
    gt_count_by_image: dict[int, int]
    baseline_events: list[_Event]
    candidate_events: list[_Event]


def paired_bootstrap_coco_predictions(
    ground_truth: Path | str,
    baseline_predictions: Path | str,
    candidate_predictions: Path | str,
    *,
    config: PairedBootstrapConfig | None = None,
) -> PairedBootstrapReport:
    """Compare candidate and matched control on identical resampled images."""
    config = config or PairedBootstrapConfig()
    gt_path = Path(ground_truth)
    baseline_path = Path(baseline_predictions)
    candidate_path = Path(candidate_predictions)
    gt_sha = _sha256(gt_path)
    baseline_sha = _sha256(baseline_path)
    candidate_sha = _sha256(candidate_path)
    protocol_hash = _protocol_hash(config, gt_sha)
    categories, image_ids, gt_by_class = _load_ground_truth(gt_path)
    baseline = _load_predictions(
        baseline_path, config.score_threshold, config.maximum_predictions_per_image
    )
    candidate = _load_predictions(
        candidate_path, config.score_threshold, config.maximum_predictions_per_image
    )
    warnings = [
        "Diagnostic paired bootstrap uses lightweight AP@0.5, not official COCO AP50-95.",
        "A one-seed image bootstrap cannot confirm component contribution.",
    ]
    blocked_reason = _blocked_reason(image_ids, baseline, candidate, config.minimum_images)
    if blocked_reason:
        return _blocked_report(
            gt_path, baseline_path, candidate_path, gt_sha, baseline_sha, candidate_sha,
            image_ids, protocol_hash, config, [*warnings, blocked_reason]
        )
    prepared = _prepare_classes(categories, gt_by_class, baseline, candidate, config.iou_threshold)
    if not prepared:
        return _blocked_report(
            gt_path, baseline_path, candidate_path, gt_sha, baseline_sha, candidate_sha,
            image_ids, protocol_hash, config, [*warnings, "no_categories_with_ground_truth"]
        )

    unit_weights = {image_id: 1 for image_id in image_ids}
    observed = [_class_values(item, unit_weights) for item in prepared]
    rng = random.Random(config.random_seed)
    class_deltas: list[list[float]] = [[] for _ in prepared]
    overall_deltas: list[float] = []
    ordered_images = sorted(image_ids)
    for _ in range(config.iterations):
        weights = Counter(rng.choices(ordered_images, k=len(ordered_images)))
        values = [_class_values(item, weights) for item in prepared]
        deltas = [current - control for control, current in values]
        for index, delta in enumerate(deltas):
            class_deltas[index].append(delta)
        overall_deltas.append(sum(deltas) / len(deltas))

    classes = [
        PairedBootstrapClassResult(
            metric_name=f"class_ap50/{item.category_name}",
            category_id=item.category_id,
            category_name=item.category_name,
            ground_truth_count=sum(item.gt_count_by_image.values()),
            baseline_value=values[0], candidate_value=values[1],
            observed_delta=values[1] - values[0],
            **_interval_fields(samples, config),
        )
        for item, values, samples in zip(prepared, observed, class_deltas, strict=True)
    ]
    baseline_map = sum(item[0] for item in observed) / len(observed)
    candidate_map = sum(item[1] for item in observed) / len(observed)
    overall = PairedBootstrapMetric(
        metric_name="diagnostic_map50", baseline_value=baseline_map,
        candidate_value=candidate_map, observed_delta=candidate_map - baseline_map,
        **_interval_fields(overall_deltas, config),
    )
    return PairedBootstrapReport(
        status="completed", ground_truth=gt_path, baseline_predictions=baseline_path,
        candidate_predictions=candidate_path, ground_truth_sha256=gt_sha,
        baseline_predictions_sha256=baseline_sha, candidate_predictions_sha256=candidate_sha,
        matched_image_count=len(image_ids), protocol_hash=protocol_hash, config=config,
        overall=overall, classes=classes,
        stable_improved_classes=[item.category_name for item in classes if item.direction == "stable_improvement"],
        stable_regressed_classes=[item.category_name for item in classes if item.direction == "stable_regression"],
        warnings=warnings,
    )


def _blocked_report(
    gt: Path, baseline: Path, candidate: Path, gt_sha: str, baseline_sha: str,
    candidate_sha: str, image_ids: set[int], protocol_hash: str,
    config: PairedBootstrapConfig, warnings: list[str],
) -> PairedBootstrapReport:
    return PairedBootstrapReport(
        status="blocked", ground_truth=gt, baseline_predictions=baseline,
        candidate_predictions=candidate, ground_truth_sha256=gt_sha,
        baseline_predictions_sha256=baseline_sha, candidate_predictions_sha256=candidate_sha,
        matched_image_count=len(image_ids), protocol_hash=protocol_hash, config=config,
        warnings=warnings,
    )


def _blocked_reason(
    image_ids: set[int], baseline: list[_Prediction], candidate: list[_Prediction],
    minimum_images: int,
) -> str | None:
    if len(image_ids) < minimum_images:
        return f"minimum_images_not_met:{len(image_ids)}/{minimum_images}"
    invalid_baseline = {item.image_id for item in baseline} - image_ids
    invalid_candidate = {item.image_id for item in candidate} - image_ids
    if invalid_baseline or invalid_candidate:
        return (
            f"predictions_outside_ground_truth:baseline={len(invalid_baseline)},"
            f"candidate={len(invalid_candidate)}"
        )
    return None


def _load_ground_truth(path: Path) -> tuple[dict[int, str], set[int], dict[int, list[_GroundTruth]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("COCO ground truth must contain a JSON object")
    categories = {
        int(item["id"]): str(item.get("name", item["id"]))
        for item in payload.get("categories", []) if isinstance(item, dict) and "id" in item
    }
    image_ids = {
        int(item["id"]) for item in payload.get("images", [])
        if isinstance(item, dict) and "id" in item
    }
    by_class: dict[int, list[_GroundTruth]] = defaultdict(list)
    for item in payload.get("annotations", []):
        if not isinstance(item, dict) or item.get("iscrowd", 0):
            continue
        target = _GroundTruth(
            image_id=int(item["image_id"]), category_id=int(item["category_id"]),
            bbox=tuple(item["bbox"]),
        )
        image_ids.add(target.image_id)
        by_class[target.category_id].append(target)
    return categories, image_ids, by_class


def _load_predictions(path: Path, threshold: float, maximum_per_image: int) -> list[_Prediction]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("predictions", payload.get("annotations", []))
    if not isinstance(payload, list):
        raise ValueError("COCO predictions must contain a JSON list")
    predictions = [
        _Prediction(
            image_id=int(item["image_id"]), category_id=int(item["category_id"]),
            bbox=tuple(item["bbox"]), score=float(item.get("score", 1.0)),
        )
        for item in payload
        if isinstance(item, dict) and float(item.get("score", 1.0)) >= threshold
    ]
    by_image: dict[int, list[_Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_image[prediction.image_id].append(prediction)
    return [
        prediction
        for image_predictions in by_image.values()
        for prediction in sorted(image_predictions, key=lambda item: item.score, reverse=True)[:maximum_per_image]
    ]


def _prepare_classes(
    categories: dict[int, str], gt_by_class: dict[int, list[_GroundTruth]],
    baseline: list[_Prediction], candidate: list[_Prediction], iou_threshold: float,
) -> list[_PreparedClass]:
    grouped: dict[str, dict[int, list[_Prediction]]] = {
        "baseline": defaultdict(list), "candidate": defaultdict(list)
    }
    for item in baseline:
        grouped["baseline"][item.category_id].append(item)
    for item in candidate:
        grouped["candidate"][item.category_id].append(item)
    return [
        _PreparedClass(
            category_id=category_id,
            category_name=categories.get(category_id, str(category_id)),
            gt_count_by_image=dict(Counter(item.image_id for item in targets)),
            baseline_events=_match_events(targets, grouped["baseline"].get(category_id, []), iou_threshold),
            candidate_events=_match_events(targets, grouped["candidate"].get(category_id, []), iou_threshold),
        )
        for category_id, targets in sorted(gt_by_class.items()) if targets
    ]


def _match_events(
    ground_truth: list[_GroundTruth], predictions: list[_Prediction], iou_threshold: float,
) -> list[_Event]:
    gt_by_image: dict[int, list[_GroundTruth]] = defaultdict(list)
    for item in ground_truth:
        gt_by_image[item.image_id].append(item)
    matched: dict[int, set[int]] = defaultdict(set)
    events: list[_Event] = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(gt_by_image.get(prediction.image_id, [])):
            if index in matched[prediction.image_id]:
                continue
            overlap = _iou(prediction.bbox, target.bbox)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        true_positive = int(best_index >= 0 and best_iou >= iou_threshold)
        if true_positive:
            matched[prediction.image_id].add(best_index)
        events.append(
            _Event(
                image_id=prediction.image_id, score=prediction.score,
                true_positive=true_positive, false_positive=1 - true_positive,
            )
        )
    return events


def _class_values(item: _PreparedClass, weights: dict[int, int]) -> tuple[float, float]:
    gt_count = sum(count * weights.get(image_id, 0) for image_id, count in item.gt_count_by_image.items())
    return (
        _weighted_average_precision(item.baseline_events, weights, gt_count),
        _weighted_average_precision(item.candidate_events, weights, gt_count),
    )


def _weighted_average_precision(events: list[_Event], weights: dict[int, int], gt_count: int) -> float:
    if gt_count <= 0:
        return 0.0
    cumulative_tp = 0.0
    cumulative_fp = 0.0
    recalls = [0.0]
    precisions = [1.0]
    for event in events:
        weight = weights.get(event.image_id, 0)
        if weight <= 0:
            continue
        cumulative_tp += event.true_positive * weight
        cumulative_fp += event.false_positive * weight
        recalls.append(min(cumulative_tp / gt_count, 1.0))
        precisions.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1.0))
    recalls.append(1.0)
    precisions.append(0.0)
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    return sum(
        (recalls[index] - recalls[index - 1]) * precisions[index]
        for index in range(1, len(recalls)) if recalls[index] > recalls[index - 1]
    )


def _interval_fields(samples: list[float], config: PairedBootstrapConfig) -> dict[str, object]:
    ordered = sorted(samples)
    alpha = (1.0 - config.confidence_level) / 2.0
    low = _percentile(ordered, alpha)
    high = _percentile(ordered, 1.0 - alpha)
    probability = sum(value > config.minimum_effect for value in samples) / len(samples)
    if low > config.minimum_effect:
        direction: BootstrapDirection = "stable_improvement"
    elif high < -config.minimum_effect:
        direction = "stable_regression"
    else:
        direction = "inconclusive"
    return {
        "confidence_interval_low": low, "confidence_interval_high": high,
        "probability_improvement": probability, "direction": direction,
    }


def _percentile(ordered: list[float], probability: float) -> float:
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_hash(config: PairedBootstrapConfig, ground_truth_sha: str) -> str:
    payload = {"config": config.model_dump(mode="json"), "ground_truth_sha256": ground_truth_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
