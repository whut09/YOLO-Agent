"""Build a :class:`DetectionErrorProfile` from real GT and predictions.

Prompt-18I step 2: attribution must come from real ground truth and real
predictions — never from a mAP drop alone.  The builder reuses the
existing mining primitives (:mod:`yolo_agent.tools.coco_error_mining`) and
extends them with the auditable decompositions the profile requires:
one-to-one greedy matching with duplicate/background/class-confusion/
high-confidence FP classification, FN breakdowns by class, scale,
confidence and scene slice, the matched-IoU distribution, the AP50-AP75
gap, the confusion matrix, TP/FP confidence histograms, and a binned
calibration summary.

Everything is deterministic: the same artifacts and metadata produce the
same profile bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from yolo_agent.core.detection_error_profile import (
    ClassificationFacts,
    ConfidenceFacts,
    DetectionErrorProfile,
    ErrorProfileSource,
    FalseNegativeFacts,
    FalsePositiveFacts,
    GlobalMetrics,
    LocalizationFacts,
    PerClassMetrics,
    ScaleMetrics,
    SceneSliceFacts,
)
from yolo_agent.tools.coco_error_mining import (
    _average_precision,
    _best_iou,
    _load_coco_ground_truth,
    _load_predictions,
    _safe_divide,
)

DEFAULT_IOU = 0.5
STRICT_IOU = 0.75
HIGH_CONFIDENCE_SCORE = 0.5
BIN_EDGES = [i / 10 for i in range(11)]

SCENE_SLICE_TAGS = ("day_night", "indoor_outdoor", "weather", "distance", "camera", "domain")


def _bin_label(value: float) -> str:
    value = max(0.0, min(0.999999, value))
    lower = int(value * 10) / 10
    return f"{lower:.1f}-{lower + 0.1:.1f}"


def _match_flags(
    predictions: list[Any],
    gt_by_image_for_class: dict[int, list[Any]],
    iou_threshold: float,
) -> tuple[list[int], list[int], dict[int, tuple[Any, float]]]:
    """Greedy one-to-one matching in score order (duplicates become FPs)."""
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    matched_gt: set[int] = set()
    match_info: dict[int, tuple[Any, float]] = {}
    for pred_index, prediction in enumerate(predictions):
        candidates = gt_by_image_for_class.get(prediction.image_id, [])
        best_gt, best_iou = _best_iou(prediction.bbox, candidates)
        gt_id = id(best_gt) if best_gt is not None else None
        if best_gt is not None and best_iou >= iou_threshold and gt_id not in matched_gt:
            matched_gt.add(gt_id)  # type: ignore[arg-type]
            tp_flags.append(1)
            fp_flags.append(0)
            match_info[pred_index] = (best_gt, best_iou)  # type: ignore[misc]
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    return tp_flags, fp_flags, match_info


def _ap_curve(
    predictions: list[Any],
    gt_for_class: list[Any],
    gt_by_image_for_class: dict[int, list[Any]],
    thresholds: list[float],
) -> dict[float, float]:
    curve: dict[float, float] = {}
    for threshold in thresholds:
        tp_flags, fp_flags, _ = _match_flags(
            predictions, gt_by_image_for_class, threshold
        )
        curve[round(threshold, 3)] = _average_precision(tp_flags, fp_flags, len(gt_for_class))
    return curve


def _load_image_metadata(path: Path | None) -> dict[int, dict[str, Any]]:
    """Load optional per-image metadata for scene slicing.

    Accepts ``{image_id: {tag: value}}`` or a COCO-style ``{"images":
    [{"id":..., "<tag>":...}, ...]}`` mapping.  A missing file yields an
    empty mapping — slices stay unpopulated (never fabricated).
    """
    if path is None or not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    mapping: dict[int, dict[str, Any]] = {}
    if isinstance(data, dict) and "images" in data:
        for image in data.get("images", []) or []:
            if isinstance(image, dict) and "id" in image:
                tags = {k: v for k, v in image.items() if k not in {"id", "file_name"}}
                if tags:
                    mapping[int(image["id"])] = tags
    elif isinstance(data, dict):
        for key, tags in data.items():
            if isinstance(tags, dict):
                mapping[int(key)] = dict(tags)
    return mapping


def _profile_id(source: ErrorProfileSource, gt_path: Path, pred_path: Path) -> str:
    digest = hashlib.sha256()
    for part in (
        source.run_id,
        source.candidate_id,
        source.node_id,
        source.protocol_hash,
        gt_path.name,
        pred_path.name,
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return f"dep-{digest.hexdigest()[:16]}"


def _read_official_metrics(path: Path | None) -> dict[str, float]:
    if path is None or not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        return {}
    result: dict[str, float] = {}
    for key in ("map50", "map50_95", "precision", "recall"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            result[key] = float(value)
    return result


def build_detection_error_profile(
    source: ErrorProfileSource,
    *,
    iou_threshold: float = DEFAULT_IOU,
    score_threshold: float = 0.001,
) -> DetectionErrorProfile:
    """Build the full error profile from real COCO GT and predictions."""
    gt_path = Path(source.gt_json)
    pred_path = Path(source.predictions_json)
    category_names, gt_by_image, gt_by_class = _load_coco_ground_truth(gt_path)
    predictions = _load_predictions(pred_path, score_threshold)
    predictions_by_class: dict[int, list[Any]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_class[prediction.category_id].append(prediction)
    image_metadata = _load_image_metadata(source.image_metadata_json)

    per_class: list[PerClassMetrics] = []
    fn_by_class: dict[str, int] = defaultdict(int)
    fn_by_scale: dict[str, int] = defaultdict(int)
    fn_by_confidence: dict[str, int] = defaultdict(int)
    fn_by_slice: dict[str, int] = defaultdict(int)
    fp_by_class: dict[str, int] = defaultdict(int)
    background_fp = 0
    duplicate_fp = 0
    class_confusion_fp = 0
    high_confidence_fp = 0
    matched_ious: list[float] = []
    confusion_counts: dict[str, int] = defaultdict(int)
    tp_scores: list[float] = []
    fp_scores: list[float] = []
    ap50_values: list[float] = []
    ap75_values: list[float] = []

    for category_id, name in sorted(category_names.items()):
        gt_for_class = gt_by_class.get(category_id, [])
        preds_for_class = sorted(
            predictions_by_class.get(category_id, []), key=lambda item: item.score, reverse=True
        )
        gt_by_image_for_class: dict[int, list[Any]] = defaultdict(list)
        for gt in gt_for_class:
            gt_by_image_for_class[gt.image_id].append(gt)

        tp_flags, fp_flags, match_info = _match_flags(
            preds_for_class, gt_by_image_for_class, iou_threshold
        )
        true_positives = sum(tp_flags)
        false_positives = sum(fp_flags)
        curve = _ap_curve(preds_for_class, gt_for_class, gt_by_image_for_class, [0.5, 0.75])
        ap50 = curve[0.5]
        ap75 = curve[0.75]
        if gt_for_class:
            ap50_values.append(ap50)
            ap75_values.append(ap75)

        per_class.append(
            PerClassMetrics(
                category_id=category_id,
                name=name,
                ap=round(ap50, 6),
                ap50=round(ap50, 6),
                precision=round(_safe_divide(true_positives, true_positives + false_positives), 6),
                recall=round(_safe_divide(true_positives, len(gt_for_class)), 6),
                support=len(gt_for_class),
            )
        )

        # --- FN decomposition ------------------------------------------------
        matched_gt_ids = {id(gt) for gt, _ in match_info.values()}
        for gt in gt_for_class:
            if id(gt) in matched_gt_ids:
                continue
            fn_by_class[name] += 1
            fn_by_scale[gt.area_bucket] += 1
            # The model's best look at this GT: the highest-scoring prediction
            # overlapping it at all, in any class.
            overlapping_scores = [
                pred.score
                for pred in predictions
                if pred.image_id == gt.image_id and _best_iou(gt.bbox, [pred])[0] is not None
            ]
            best_look = max(overlapping_scores) if overlapping_scores else None
            fn_by_confidence["none" if best_look is None else _bin_label(best_look)] += 1
            tags = image_metadata.get(gt.image_id, {})
            for tag_name in SCENE_SLICE_TAGS:
                if tag_name in tags:
                    fn_by_slice[f"{tag_name}={tags[tag_name]}"] += 1

        # --- FP decomposition ------------------------------------------------
        # Cross-class view: an FP is classified against ALL ground truth on
        # the image, because "which error kind is this?" is a question about
        # the scene, not about the predicted class's own GT subset.
        image_gt_all: dict[int, list[Any]] = defaultdict(list)
        for gt_list in gt_by_image.values():
            for gt in gt_list:
                image_gt_all[gt.image_id].append(gt)
        for pred_index, prediction in enumerate(preds_for_class):
            if fp_flags[pred_index] == 0:
                tp_scores.append(prediction.score)
                matched_ious.append(match_info[pred_index][1])
                continue
            fp_scores.append(prediction.score)
            fp_by_class[name] += 1
            if prediction.score >= HIGH_CONFIDENCE_SCORE:
                high_confidence_fp += 1
            best_gt, best_iou = _best_iou(
                prediction.bbox, image_gt_all.get(prediction.image_id, [])
            )
            if best_gt is None or best_iou <= 0.0:
                background_fp += 1
            elif best_iou >= iou_threshold:
                best_class_name = category_names.get(
                    best_gt.category_id, str(best_gt.category_id)
                )
                if best_class_name != name:
                    class_confusion_fp += 1
                    confusion_counts[f"{best_class_name}->{name}"] += 1
                else:
                    duplicate_fp += 1  # own-class GT already matched (or weakly matched)
            else:
                best_class_name = category_names.get(
                    best_gt.category_id, str(best_gt.category_id)
                )
                if best_class_name != name:
                    class_confusion_fp += 1
                    confusion_counts[f"{best_class_name}->{name}"] += 1
                else:
                    duplicate_fp += 1  # loose own-class overlap below threshold

    # --- localization -----------------------------------------------------------
    iou_histogram: dict[str, int] = defaultdict(int)
    for value in matched_ious:
        iou_histogram[_bin_label(value)] += 1
    localization_errors = sum(1 for value in matched_ious if value < STRICT_IOU)
    gap = (
        round(
            _safe_divide(sum(ap50_values), len(ap50_values))
            - _safe_divide(sum(ap75_values), len(ap75_values)),
            6,
        )
        if ap50_values
        else None
    )

    # --- confidence + calibration -------------------------------------------------
    tp_histogram: dict[str, int] = defaultdict(int)
    for score in tp_scores:
        tp_histogram[_bin_label(score)] += 1
    fp_histogram: dict[str, int] = defaultdict(int)
    for score in fp_scores:
        fp_histogram[_bin_label(score)] += 1
    calibration_bins: list[dict[str, float]] = []
    ece = 0.0
    total_scores = len(tp_scores) + len(fp_scores)
    if total_scores:
        for bin_index in range(10):
            lower = bin_index / 10
            upper = (bin_index + 1) / 10
            in_bin_tp = [s for s in tp_scores if lower <= s < upper]
            in_bin_fp = [s for s in fp_scores if lower <= s < upper]
            count = len(in_bin_tp) + len(in_bin_fp)
            if not count:
                continue
            mean_confidence = (sum(in_bin_tp) + sum(in_bin_fp)) / count
            tp_rate = len(in_bin_tp) / count
            calibration_bins.append(
                {
                    "bin": round(lower, 2),
                    "mean_confidence": round(mean_confidence, 6),
                    "tp_rate": round(tp_rate, 6),
                    "count": float(count),
                }
            )
            ece += (count / total_scores) * abs(tp_rate - mean_confidence)

    # --- global metrics -----------------------------------------------------------
    official = _read_official_metrics(source.official_metrics_json)
    if official:
        global_metrics = GlobalMetrics(
            map50=official.get("map50"),
            map50_95=official.get("map50_95"),
            precision=official.get("precision"),
            recall=official.get("recall"),
        )
    else:
        predicted_total = len(tp_scores) + len(fp_scores)
        gt_total = sum(len(boxes) for boxes in gt_by_class.values())
        global_metrics = GlobalMetrics(
            map50=round(_safe_divide(sum(ap50_values), len(ap50_values)), 6) if ap50_values else None,
            map50_95=None,  # official metrics fill this; estimates only at IoU=0.5
            precision=round(_safe_divide(len(tp_scores), predicted_total), 6) if predicted_total else None,
            recall=round(_safe_divide(sum(match_info.values().__len__() for _ in [0]) and sum(len(match_info) for _ in [0]), gt_total), 6) if gt_total else None,
        )

    profile = DetectionErrorProfile(
        profile_id=_profile_id(source, gt_path, pred_path),
        run_id=source.run_id,
        candidate_id=source.candidate_id,
        node_id=source.node_id,
        split=source.split,
        role=source.role,
        protocol_hash=source.protocol_hash,
        gt_artifact=str(gt_path),
        predictions_artifact=str(pred_path),
        dataset_manifest_hash=source.dataset_manifest_hash,
        global_=global_metrics,
        scale=_scale_ap(gt_by_class, predictions_by_class),
        per_class=per_class,
        false_negative=FalseNegativeFacts(
            total=sum(fn_by_class.values()),
            by_class=dict(fn_by_class),
            by_scale=dict(fn_by_scale),
            by_confidence=dict(fn_by_confidence),
            by_scene_slice=dict(fn_by_slice),
        ),
        false_positive=FalsePositiveFacts(
            total=len(fp_scores),
            by_class=dict(fp_by_class),
            background_fp=background_fp,
            duplicate_fp=duplicate_fp,
            class_confusion_fp=class_confusion_fp,
            high_confidence_fp=high_confidence_fp,
        ),
        localization=LocalizationFacts(
            matched_iou_distribution=dict(sorted(iou_histogram.items())),
            mean_matched_iou=round(sum(matched_ious) / len(matched_ious), 6) if matched_ious else None,
            localization_error_count=localization_errors,
            ap50_vs_ap75_gap=gap,
        ),
        classification=ClassificationFacts(
            confusion_matrix=dict(sorted(confusion_counts.items())),
            top_confusion_pairs=sorted(confusion_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
        ),
        confidence=ConfidenceFacts(
            tp_confidence_histogram=dict(sorted(tp_histogram.items())),
            fp_confidence_histogram=dict(sorted(fp_histogram.items())),
            calibration_bins=calibration_bins,
            expected_calibration_error=round(ece, 6) if total_scores else None,
        ),
        scene_slices=_scene_slices(gt_by_image, predictions, image_metadata, iou_threshold),
        unavailable_scene_slices=[
            tag
            for tag in SCENE_SLICE_TAGS
            if tag not in {key for tags in image_metadata.values() for key in tags}
        ],
    )
    if not image_metadata:
        profile.unavailable_scene_slices = list(SCENE_SLICE_TAGS)
    return profile


def _scale_ap(
    gt_by_class: dict[int, list[Any]],
    predictions_by_class: dict[int, list[Any]],
) -> ScaleMetrics:
    """AP@0.5 and recall per area bucket (bucket-restricted GT sets)."""
    bucket_ap: dict[str, list[float]] = {"small": [], "medium": [], "large": []}
    bucket_gt_totals: dict[str, int] = defaultdict(int)
    bucket_recall_counts: dict[str, int] = defaultdict(int)
    for category_id in sorted(set(gt_by_class) | set(predictions_by_class)):
        gt_for_class = gt_by_class.get(category_id, [])
        preds_for_class = sorted(
            predictions_by_class.get(category_id, []), key=lambda item: item.score, reverse=True
        )
        if not gt_for_class and not preds_for_class:
            continue
        for bucket in bucket_ap:
            bucket_gt = [box for box in gt_for_class if box.area_bucket == bucket]
            if not bucket_gt:
                continue
            bucket_by_image: dict[int, list[Any]] = defaultdict(list)
            for gt in bucket_gt:
                bucket_by_image[gt.image_id].append(gt)
            tp_flags, _, _ = _match_flags(preds_for_class, bucket_by_image, 0.5)
            curve = _ap_curve(preds_for_class, bucket_gt, bucket_by_image, [0.5])
            bucket_ap[bucket].append(curve[0.5])
            bucket_gt_totals[bucket] += len(bucket_gt)
            bucket_recall_counts[bucket] += sum(tp_flags)
    def _bucket_recall(bucket: str) -> float | None:
        recalled = bucket_recall_counts.get(bucket, 0)
        total = bucket_gt_totals.get(bucket, 0)
        return round(_safe_divide(recalled, total), 6) if total else None

    return ScaleMetrics(
        ap_small=round(_safe_divide(sum(bucket_ap["small"]), len(bucket_ap["small"])), 6) if bucket_ap["small"] else None,
        ap_medium=round(_safe_divide(sum(bucket_ap["medium"]), len(bucket_ap["medium"])), 6) if bucket_ap["medium"] else None,
        ap_large=round(_safe_divide(sum(bucket_ap["large"]), len(bucket_ap["large"])), 6) if bucket_ap["large"] else None,
        recall_small=_bucket_recall("small"),
        recall_medium=_bucket_recall("medium"),
        recall_large=_bucket_recall("large"),
    )


def _scene_slices(
    gt_by_image: dict[int, list[Any]],
    predictions: list[Any],
    image_metadata: dict[int, dict[str, Any]],
    iou_threshold: float,
) -> list[SceneSliceFacts]:
    if not image_metadata:
        return []
    grouped: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for image_id, tags in image_metadata.items():
        for tag_name in SCENE_SLICE_TAGS:
            if tag_name in tags:
                grouped[tag_name][str(tags[tag_name])].add(image_id)
    predictions_by_image: dict[int, list[Any]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_image[prediction.image_id].append(prediction)
    slices: list[SceneSliceFacts] = []
    for tag_name in sorted(grouped):
        for value, image_ids in sorted(grouped[tag_name].items()):
            gt_count = sum(len(gt_by_image.get(image_id, [])) for image_id in image_ids)
            tp = 0
            for image_id in image_ids:
                gts = gt_by_image.get(image_id, [])
                preds = predictions_by_image.get(image_id, [])
                matched: set[int] = set()
                for gt in gts:
                    best, best_iou = _best_iou(gt.bbox, preds)
                    if best is not None and best_iou >= iou_threshold and id(best) not in matched:
                        matched.add(id(best))
                        tp += 1
            slices.append(
                SceneSliceFacts(
                    slice_name=f"{tag_name}={value}",
                    gt_count=gt_count,
                    true_positives=tp,
                    recall=round(_safe_divide(tp, gt_count), 6) if gt_count else None,
                )
            )
    return slices
