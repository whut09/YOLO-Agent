import pytest

from yolo_agent.components.adapters.inference.postprocess import (
    apply_class_thresholds,
    bbox_iou,
    calibrate_confidence,
    merge_predictions,
)


def _prediction(score: float, bbox: list[float], category_id: int = 1) -> dict[str, object]:
    return {"image_id": 1, "category_id": category_id, "bbox": bbox, "score": score}


def test_temperature_scaling_is_monotonic_and_non_mutating() -> None:
    source = [_prediction(0.9, [0, 0, 10, 10]), _prediction(0.6, [20, 20, 5, 5])]
    calibrated = calibrate_confidence(source, 2.0)

    assert calibrated[0]["score"] > calibrated[1]["score"]
    assert calibrated[0]["score"] < 0.9
    assert source[0]["score"] == 0.9


def test_class_aware_thresholds_filter_independently() -> None:
    source = [
        _prediction(0.4, [0, 0, 10, 10], 1),
        _prediction(0.4, [20, 20, 10, 10], 2),
    ]
    output = apply_class_thresholds(source, {1: 0.5, 2: 0.3}, default_threshold=0.25)

    assert [item["category_id"] for item in output] == [2]


@pytest.mark.parametrize("policy", ["nms", "nmm", "weighted_box_fusion"])
def test_cross_view_merge_reduces_overlapping_predictions(policy: str) -> None:
    source = [
        _prediction(0.9, [0, 0, 10, 10]),
        _prediction(0.8, [1, 1, 10, 10]),
        _prediction(0.7, [30, 30, 5, 5]),
    ]

    output, stats = merge_predictions(
        source, policy=policy, iou_threshold=0.5, max_detections=300
    )

    assert len(output) == 2
    assert stats["merged_count"] == 1
    assert stats["extra_nms_applied"] is (policy == "nms")


def test_bbox_iou_uses_coco_xywh() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou([0, 0, 10, 10], [20, 20, 1, 1]) == 0.0
