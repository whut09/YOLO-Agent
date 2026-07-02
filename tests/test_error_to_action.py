"""Error-to-action mapper tests."""

from __future__ import annotations

from yolo_agent.agents.error_to_action import (
    DetectionErrorObservation,
    ErrorActionMapper,
)


def test_default_policy_loads_required_taxonomy() -> None:
    """Default policies should cover the first error taxonomy."""
    mapper = ErrorActionMapper.from_yaml()

    assert "small_object_miss" in mapper.policies
    assert "background_confusion" in mapper.policies
    assert "loose_box" in mapper.policies
    assert "class_confusion" in mapper.policies


def test_small_object_miss_maps_to_core_actions() -> None:
    """Small object misses should map to small-object recipe actions."""
    mapper = ErrorActionMapper.from_yaml()

    plan = mapper.map_errors(
        [
            DetectionErrorObservation(
                error_type="small_object_miss",
                count=5,
                severity="high",
            )
        ]
    )
    action_ids = {recommendation.action.id for recommendation in plan.recommendations}

    assert {
        "increase_imgsz",
        "add_p2_head",
        "enable_tiling_inference",
        "switch_to_nwd_loss",
        "increase_positive_assigner_weight",
    } <= action_ids
    assert plan.recommendations[0].priority == 15


def test_background_confusion_maps_to_false_positive_actions() -> None:
    """Background false positives should produce hard-negative actions."""
    mapper = ErrorActionMapper.from_yaml()

    plan = mapper.map_errors(
        [
            DetectionErrorObservation(
                error_type="background_confusion",
                count=3,
                severity="medium",
            )
        ]
    )
    action_ids = {recommendation.action.id for recommendation in plan.recommendations}

    assert "add_hard_negative_mining" in action_ids
    assert "reduce_mosaic_strength" in action_ids
    assert "add_background_only_sampling" in action_ids
    assert "increase_focal_loss_gamma" in action_ids
    assert {recommendation.error_category for recommendation in plan.recommendations} == {"false_positive"}


def test_mapper_prioritizes_by_count_and_severity() -> None:
    """Higher severity/count errors should sort first."""
    mapper = ErrorActionMapper.from_yaml()

    plan = mapper.map_errors(
        [
            DetectionErrorObservation(error_type="class_confusion", count=1, severity="low"),
            DetectionErrorObservation(error_type="small_object_miss", count=4, severity="high"),
        ]
    )

    assert plan.recommendations[0].error_type == "small_object_miss"
    assert plan.unresolved_errors == []

