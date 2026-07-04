"""COCO-style detection error mining for closed-loop optimization."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.error_to_action import DetectionErrorObservation


AreaBucket = Literal["small", "medium", "large"]


class ClassErrorSummary(BaseModel):
    """Per-class detection quality and error counts."""

    category_id: int
    name: str
    gt_count: int = 0
    prediction_count: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    class_confusion: int = 0
    localization_error: int = 0
    background_false_positive: int = 0
    precision: float = 0.0
    recall: float = 0.0
    ap50: float = 0.0


class CocoErrorReport(BaseModel):
    """Structured COCO validation error report."""

    gt_json: Path
    predictions_json: Path
    iou_threshold: float = 0.5
    score_threshold: float = 0.001
    class_summaries: list[ClassErrorSummary]
    area_ap50: dict[AreaBucket, float] = Field(default_factory=dict)
    area_recall: dict[AreaBucket, float] = Field(default_factory=dict)
    false_negative_top_classes: list[ClassErrorSummary] = Field(default_factory=list)
    localization_error_top_classes: list[ClassErrorSummary] = Field(default_factory=list)
    class_confusion_pairs: dict[str, int] = Field(default_factory=dict)
    background_false_positive_top_classes: list[ClassErrorSummary] = Field(default_factory=list)
    observations: list[DetectionErrorObservation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class _GtBox(BaseModel):
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    area_bucket: AreaBucket


class _Prediction(BaseModel):
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float


def mine_coco_errors(
    gt_json: Path | str,
    predictions_json: Path | str,
    out_prefix: Path | str | None = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001,
) -> CocoErrorReport:
    """Mine COCO detection error facts from GT annotations and predictions.

    The AP values are lightweight AP@0.5 estimates for loop guidance. They are
    not a replacement for official pycocotools COCO metrics.
    """
    gt_path = Path(gt_json)
    pred_path = Path(predictions_json)
    category_names, gt_by_image, gt_by_class = _load_coco_ground_truth(gt_path)
    predictions = _load_predictions(pred_path, score_threshold)
    predictions_by_class: dict[int, list[_Prediction]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_class[prediction.category_id].append(prediction)

    class_summaries: list[ClassErrorSummary] = []
    class_counts: dict[int, ClassErrorSummary] = {}
    area_stats: dict[AreaBucket, dict[str, float]] = {
        "small": {"tp": 0, "gt": 0, "ap_sum": 0, "classes": 0},
        "medium": {"tp": 0, "gt": 0, "ap_sum": 0, "classes": 0},
        "large": {"tp": 0, "gt": 0, "ap_sum": 0, "classes": 0},
    }
    confusion_pairs: dict[str, int] = defaultdict(int)

    for category_id, name in sorted(category_names.items()):
        gt_for_class = gt_by_class.get(category_id, [])
        preds_for_class = sorted(predictions_by_class.get(category_id, []), key=lambda item: item.score, reverse=True)
        summary, matched_gt_ids, matched_predictions = _evaluate_class(
            category_id=category_id,
            name=name,
            gt_for_class=gt_for_class,
            gt_by_image=gt_by_image,
            predictions=preds_for_class,
            iou_threshold=iou_threshold,
        )
        class_summaries.append(summary)
        class_counts[category_id] = summary

        for bucket in ("small", "medium", "large"):
            bucket_gt = [box for box in gt_for_class if box.area_bucket == bucket]
            if not bucket_gt:
                continue
            bucket_tp = sum(1 for index, box in enumerate(gt_for_class) if index in matched_gt_ids and box.area_bucket == bucket)
            area_stats[bucket]["tp"] += bucket_tp
            area_stats[bucket]["gt"] += len(bucket_gt)
            area_stats[bucket]["ap_sum"] += _safe_divide(bucket_tp, len(bucket_gt))
            area_stats[bucket]["classes"] += 1

        _collect_cross_class_errors(
            predictions=preds_for_class,
            matched_prediction_indices=matched_predictions,
            gt_by_image=gt_by_image,
            predicted_category_id=category_id,
            predicted_name=name,
            category_names=category_names,
            iou_threshold=iou_threshold,
            summary=summary,
            confusion_pairs=confusion_pairs,
        )

    area_ap50 = {
        bucket: round(_safe_divide(values["ap_sum"], values["classes"]), 6)
        for bucket, values in area_stats.items()
    }
    area_recall = {
        bucket: round(_safe_divide(values["tp"], values["gt"]), 6)
        for bucket, values in area_stats.items()
    }

    report = CocoErrorReport(
        gt_json=gt_path,
        predictions_json=pred_path,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_summaries=class_summaries,
        area_ap50=area_ap50,
        area_recall=area_recall,
        false_negative_top_classes=_top(class_summaries, "false_negative"),
        localization_error_top_classes=_top(class_summaries, "localization_error"),
        class_confusion_pairs=dict(sorted(confusion_pairs.items(), key=lambda item: item[1], reverse=True)[:20]),
        background_false_positive_top_classes=_top(class_summaries, "background_false_positive"),
        observations=_observations(class_summaries, area_recall),
        notes=[
            "AP values are lightweight AP@0.5 loop diagnostics, not official COCO AP50-95.",
            "Use official Ultralytics/COCO metrics as trusted performance evidence.",
        ],
    )
    if out_prefix is not None:
        write_coco_error_report(report, out_prefix)
    return report


def write_coco_error_report(report: CocoErrorReport, out_prefix: Path | str) -> tuple[Path, Path, Path]:
    """Write JSON, Markdown, and diagnose-compatible error YAML."""
    prefix = Path(out_prefix)
    json_path = prefix.with_suffix(".json") if prefix.suffix else Path(f"{prefix}.json")
    md_path = prefix.with_suffix(".md") if prefix.suffix else Path(f"{prefix}.md")
    errors_path = prefix.with_name(f"{prefix.name}_errors.yaml").with_suffix(".yaml")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    with errors_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {"errors": [observation.model_dump(mode="json") for observation in report.observations]},
            file,
            sort_keys=False,
        )
    return json_path, md_path, errors_path


def _load_coco_ground_truth(path: Path) -> tuple[dict[int, str], dict[int, list[_GtBox]], dict[int, list[_GtBox]]]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    categories = {int(item["id"]): str(item["name"]) for item in data.get("categories", [])}
    by_image: dict[int, list[_GtBox]] = defaultdict(list)
    by_class: dict[int, list[_GtBox]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        bbox = _bbox_tuple(ann["bbox"])
        gt = _GtBox(
            image_id=int(ann["image_id"]),
            category_id=int(ann["category_id"]),
            bbox=bbox,
            area_bucket=_area_bucket(float(ann.get("area", bbox[2] * bbox[3]))),
        )
        by_image[gt.image_id].append(gt)
        by_class[gt.category_id].append(gt)
    return categories, by_image, by_class


def _load_predictions(path: Path, score_threshold: float) -> list[_Prediction]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    items = data.get("predictions", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("COCO predictions must be a list or contain a 'predictions' list.")
    predictions: list[_Prediction] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        score = float(item.get("score", item.get("confidence", 0.0)))
        if score < score_threshold:
            continue
        predictions.append(
            _Prediction(
                image_id=int(item["image_id"]),
                category_id=int(item["category_id"]),
                bbox=_bbox_tuple(item["bbox"]),
                score=score,
            )
        )
    return predictions


def _evaluate_class(
    category_id: int,
    name: str,
    gt_for_class: list[_GtBox],
    gt_by_image: dict[int, list[_GtBox]],
    predictions: list[_Prediction],
    iou_threshold: float,
) -> tuple[ClassErrorSummary, set[int], set[int]]:
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    gt_index = {id(box): index for index, box in enumerate(gt_for_class)}

    gt_by_image_for_class: dict[int, list[_GtBox]] = defaultdict(list)
    for gt in gt_for_class:
        gt_by_image_for_class[gt.image_id].append(gt)

    for pred_index, prediction in enumerate(predictions):
        candidates = gt_by_image_for_class.get(prediction.image_id, [])
        best_gt, best_iou = _best_iou(prediction.bbox, candidates)
        if best_gt is not None and best_iou >= iou_threshold and gt_index[id(best_gt)] not in matched_gt:
            matched_gt.add(gt_index[id(best_gt)])
            matched_predictions.add(pred_index)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    true_positive = sum(tp_flags)
    false_positive = sum(fp_flags)
    false_negative = max(len(gt_for_class) - true_positive, 0)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, len(gt_for_class))
    ap50 = _average_precision(tp_flags, fp_flags, len(gt_for_class))
    summary = ClassErrorSummary(
        category_id=category_id,
        name=name,
        gt_count=len(gt_for_class),
        prediction_count=len(predictions),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=round(precision, 6),
        recall=round(recall, 6),
        ap50=round(ap50, 6),
    )
    return summary, matched_gt, matched_predictions


def _collect_cross_class_errors(
    predictions: list[_Prediction],
    matched_prediction_indices: set[int],
    gt_by_image: dict[int, list[_GtBox]],
    predicted_category_id: int,
    predicted_name: str,
    category_names: dict[int, str],
    iou_threshold: float,
    summary: ClassErrorSummary,
    confusion_pairs: dict[str, int],
) -> None:
    for index, prediction in enumerate(predictions):
        if index in matched_prediction_indices:
            continue
        all_gt = gt_by_image.get(prediction.image_id, [])
        best_gt, best_iou = _best_iou(prediction.bbox, all_gt)
        if best_gt is None or best_iou < 0.1:
            summary.background_false_positive += 1
            continue
        if best_gt.category_id != predicted_category_id and best_iou >= iou_threshold:
            summary.class_confusion += 1
            actual = category_names.get(best_gt.category_id, str(best_gt.category_id))
            confusion_pairs[f"{actual}->{predicted_name}"] += 1
            continue
        if best_gt.category_id == predicted_category_id and best_iou < iou_threshold:
            summary.localization_error += 1


def _observations(
    class_summaries: list[ClassErrorSummary],
    area_recall: dict[AreaBucket, float],
) -> list[DetectionErrorObservation]:
    observations: list[DetectionErrorObservation] = []
    total_fn = sum(item.false_negative for item in class_summaries)
    total_fp_bg = sum(item.background_false_positive for item in class_summaries)
    total_loc = sum(item.localization_error for item in class_summaries)
    total_conf = sum(item.class_confusion for item in class_summaries)
    if area_recall.get("small", 1.0) <= 0.5:
        observations.append(
            DetectionErrorObservation(
                error_type="small_object_miss",
                count=max(1, int(sum(item.false_negative for item in class_summaries))),
                severity="high" if area_recall.get("small", 1.0) < 0.35 else "medium",
                notes=[f"small_object_recall={area_recall.get('small', 0.0):.4f}"],
            )
        )
    if total_fn:
        observations.append(
            DetectionErrorObservation(
                error_type="out_of_distribution_miss",
                count=total_fn,
                severity=_severity(total_fn),
                notes=["False negatives require class/scene review before model changes."],
            )
        )
    if total_fp_bg:
        observations.append(
            DetectionErrorObservation(
                error_type="background_confusion",
                count=total_fp_bg,
                severity=_severity(total_fp_bg),
            )
        )
    if total_loc:
        observations.append(
            DetectionErrorObservation(
                error_type="shifted_box",
                count=total_loc,
                severity=_severity(total_loc),
            )
        )
    if total_conf:
        observations.append(
            DetectionErrorObservation(
                error_type="class_confusion",
                count=total_conf,
                severity=_severity(total_conf),
            )
        )
    return observations


def _markdown(report: CocoErrorReport) -> str:
    lines = [
        "# COCO Error Mining Report",
        "",
        f"- GT: `{report.gt_json}`",
        f"- Predictions: `{report.predictions_json}`",
        f"- IoU threshold: `{report.iou_threshold}`",
        "",
        "## Area Metrics",
        "",
        "| Area | AP50 approx | Recall |",
        "| --- | ---: | ---: |",
    ]
    for bucket in ("small", "medium", "large"):
        lines.append(f"| {bucket} | {report.area_ap50.get(bucket, 0):.4f} | {report.area_recall.get(bucket, 0):.4f} |")
    lines.extend(["", "## False Negative Top Classes", "", "| Class | FN | Recall | AP50 approx |", "| --- | ---: | ---: | ---: |"])
    for item in report.false_negative_top_classes:
        lines.append(f"| {item.name} | {item.false_negative} | {item.recall:.4f} | {item.ap50:.4f} |")
    lines.extend(["", "## Error Observations", ""])
    for observation in report.observations:
        lines.append(f"- `{observation.error_type}` count={observation.count} severity={observation.severity}")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report.notes)
    lines.append("")
    return "\n".join(lines)


def _best_iou(
    bbox: tuple[float, float, float, float],
    candidates: list[_GtBox],
) -> tuple[_GtBox | None, float]:
    best_gt: _GtBox | None = None
    best_iou = 0.0
    for gt in candidates:
        overlap = _iou_xywh(bbox, gt.bbox)
        if overlap > best_iou:
            best_gt = gt
            best_iou = overlap
    return best_gt, best_iou


def _iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return _safe_divide(intersection, union)


def _average_precision(tp_flags: list[int], fp_flags: list[int], gt_count: int) -> float:
    if gt_count <= 0:
        return 0.0
    tp_cum = 0
    fp_cum = 0
    points: list[tuple[float, float]] = []
    for tp, fp in zip(tp_flags, fp_flags, strict=False):
        tp_cum += tp
        fp_cum += fp
        recall = _safe_divide(tp_cum, gt_count)
        precision = _safe_divide(tp_cum, tp_cum + fp_cum)
        points.append((recall, precision))
    ap = 0.0
    previous_recall = 0.0
    for recall, precision in points:
        ap += max(recall - previous_recall, 0.0) * precision
        previous_recall = recall
    return ap


def _area_bucket(area: float) -> AreaBucket:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def _bbox_tuple(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"bbox must be [x, y, w, h], got: {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _top(items: list[ClassErrorSummary], field: str, limit: int = 10) -> list[ClassErrorSummary]:
    return sorted(items, key=lambda item: int(getattr(item, field)), reverse=True)[:limit]


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _severity(count: int) -> Literal["low", "medium", "high"]:
    if count >= 50:
        return "high"
    if count >= 10:
        return "medium"
    return "low"
