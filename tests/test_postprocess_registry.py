"""Post-processing strategy registry tests."""

from __future__ import annotations

from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.components.postprocess import PostProcessRegistry
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def test_postprocess_registry_loads_strategy_families() -> None:
    """The bundled registry should expose NMS, threshold, TTA, and slicing strategies."""
    registry = PostProcessRegistry.from_yaml()

    assert registry.get("standard_nms").family == "nms"
    assert registry.get("soft_nms").family == "nms"
    assert registry.get("weighted_box_fusion").family == "fusion"
    assert registry.get("sahi_slicing").family == "slicing"
    assert registry.get("class_aware_nms").family == "nms"
    assert registry.get("recall_first_threshold").family == "threshold"
    assert registry.get("precision_first_threshold").family == "threshold"
    assert registry.get_by_family("threshold")


def test_crowded_scene_recommendation_matches_policy() -> None:
    """Crowded scenes should prefer softer suppression and TTA policies."""
    registry = PostProcessRegistry.from_yaml()

    recommendation = registry.recommend("crowded_scene")

    assert recommendation.ids == ["soft_nms", "class_aware_nms", "per_class_threshold", "multi_scale_tta"]
    assert recommendation.scenario == "crowded_scene"


def test_edge_realtime_recommendation_matches_policy() -> None:
    """Edge realtime should prefer simple low-latency inference policy."""
    registry = PostProcessRegistry.from_yaml()
    task = TaskSpec(
        task_type="detect",
        scene="traffic_edge",
        class_names=["person", "car"],
        primary_metric=MetricPriority(name="fps"),
        device_type="npu",
        target_fps=30,
        max_latency_ms=33,
    )

    recommendation = registry.recommend(task)

    assert recommendation.ids == ["standard_nms", "single_scale", "high_conf_threshold"]
    assert all(strategy.latency_cost == "low" for strategy in recommendation.recommended_postprocess)


def test_problem_lookup_finds_small_object_postprocess() -> None:
    """Problem lookup should find SAHI and multi-scale options for small-object misses."""
    registry = PostProcessRegistry.from_yaml()

    strategy_ids = {strategy.id for strategy in registry.get_by_problem("small-object-miss")}

    assert "sahi_slicing" in strategy_ids
    assert "multi_scale_inference" in strategy_ids


def test_error_driven_small_object_policy_adds_sahi_and_recall_thresholds() -> None:
    """Small-object misses should select SAHI and recall-oriented thresholds."""
    registry = PostProcessRegistry.from_yaml()

    recommendation = registry.recommend_for_errors(
        [DetectionErrorObservation(error_type="small_object_miss", count=5, severity="high")],
        "infrared_small_target",
    )

    assert "sahi_slicing" in recommendation.ids
    assert "low_conf_threshold" in recommendation.ids
    assert "recall_first_threshold" in recommendation.ids
    assert "consider_nwd_loss" in recommendation.companion_actions
    assert recommendation.warnings


def test_error_driven_background_policy_prefers_precision_and_class_aware_nms() -> None:
    """Background false positives should select precision-first inference policy."""
    registry = PostProcessRegistry.from_yaml()

    recommendation = registry.recommend_for_errors(
        [DetectionErrorObservation(error_type="background_confusion", count=3, severity="medium")],
        "edge_realtime",
    )

    assert "precision_first_threshold" in recommendation.ids
    assert "class_aware_nms" in recommendation.ids
    assert "add_hard_negative_mining" in recommendation.companion_actions
