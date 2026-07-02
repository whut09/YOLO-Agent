"""Optimization recipe engine tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.agents.optimization_recipe import OptimizationRecipeEngine
from yolo_agent.core.task_spec import MetricPriority, TaskSpec
from yolo_agent.tools.dataset_stats import DatasetHealth, DatasetReport


def _task(scene: str = "infrared_small_target", primary_metric: str = "recall") -> TaskSpec:
    return TaskSpec(
        task_type="detect",
        scene=scene,  # type: ignore[arg-type]
        class_names=["target"],
        primary_metric=MetricPriority(name=primary_metric),  # type: ignore[arg-type]
        secondary_metrics=[MetricPriority(name="map50_95")],
    )


def _dataset_report(problems: list[str] | None = None) -> DatasetReport:
    return DatasetReport(
        data_yaml=Path("data.yaml"),
        dataset_root=Path("."),
        scene="infrared_small_target",
        image_count=10,
        label_count=10,
        class_distribution={"target": 10},
        dataset_health=DatasetHealth(score=70, problems=problems or []),
    )


def test_small_object_localization_recipe_links_loss_head_assigner_and_imgsz() -> None:
    """Small-object localization errors should not recommend loss in isolation."""
    engine = OptimizationRecipeEngine.from_yaml()

    plan = engine.recommend(
        _task(),
        [DetectionErrorObservation(error_type="small_object_miss", count=3, severity="high")],
    )

    assert "small_object_localization" in {item.recipe_id for item in plan.recommendations}
    assert {"loss.bbox.nwd", "loss.bbox.mpdiou"} <= set(plan.component_candidates.bbox_loss)
    assert "head.p2_small_object" in plan.component_candidates.head
    assert "assigner.stal" in plan.component_candidates.assigner
    assert plan.train_overrides["imgsz"] == 960
    assert "sahi_slicing" in plan.postprocess
    assert "verify_small_object_labels" in plan.data_checks


def test_loose_box_recipe_pairs_losses_with_label_quality_check() -> None:
    """Loose boxes should trigger localization losses and stricter label checks."""
    engine = OptimizationRecipeEngine.from_yaml()

    plan = engine.recommend(
        _task(scene="generic", primary_metric="map50_95"),
        [DetectionErrorObservation(error_type="loose_box", count=2, severity="medium")],
    )

    assert "loose_box_localization" in {item.recipe_id for item in plan.recommendations}
    assert {"loss.bbox.wiou", "loss.bbox.mpdiou"} <= set(plan.component_candidates.bbox_loss)
    assert "stricter_label_quality_check" in plan.data_checks
    assert "redraw_loose_or_partial_boxes" in plan.data_checks
    assert "map50_95" in plan.evidence_required


def test_low_recall_recipe_adjusts_assigner_and_checks_missing_labels() -> None:
    """Low recall should combine assigner changes, thresholds, and label checks."""
    engine = OptimizationRecipeEngine.from_yaml()

    plan = engine.recommend(
        _task(scene="generic", primary_metric="recall"),
        [DetectionErrorObservation(error_type="occlusion_miss", count=1, severity="medium")],
    )

    assert "low_recall_assigner" in {item.recipe_id for item in plan.recommendations}
    assert "assigner.stal" in plan.component_candidates.assigner
    assert plan.train_overrides["positive_assigner_weight"] == "increase"
    assert "check_missing_labels" in plan.data_checks
    assert "recall_first_threshold" in plan.postprocess


def test_annotation_noise_recipe_blocks_blind_component_chasing() -> None:
    """Dataset health problems should produce guard recipes before component ablations."""
    engine = OptimizationRecipeEngine.from_yaml()

    plan = engine.recommend(
        _task(scene="industrial_defect"),
        dataset_report=_dataset_report(["annotation_noise"]),
    )

    assert "label_noise_guard" in {item.recipe_id for item in plan.recommendations}
    assert "label_audit_queue" in plan.data_checks
    assert "redraw_suspicious_boxes" in plan.data_checks
    assert "label_quality_report" in plan.evidence_required
