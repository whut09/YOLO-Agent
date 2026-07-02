"""Post-processing strategy registry tests."""

from __future__ import annotations

from yolo_agent.components.postprocess import PostProcessRegistry
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def test_postprocess_registry_loads_strategy_families() -> None:
    """The bundled registry should expose NMS, threshold, TTA, and slicing strategies."""
    registry = PostProcessRegistry.from_yaml()

    assert registry.get("standard_nms").family == "nms"
    assert registry.get("soft_nms").family == "nms"
    assert registry.get("weighted_box_fusion").family == "fusion"
    assert registry.get("sahi_slicing").family == "slicing"
    assert registry.get_by_family("threshold")


def test_crowded_scene_recommendation_matches_policy() -> None:
    """Crowded scenes should prefer softer suppression and TTA policies."""
    registry = PostProcessRegistry.from_yaml()

    recommendation = registry.recommend("crowded_scene")

    assert recommendation.ids == ["soft_nms", "per_class_threshold", "multi_scale_tta"]
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

