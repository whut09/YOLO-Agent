"""Augmentation policy engine tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.augmentation_policy import AugmentationPolicyEngine
from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.tools.dataset_stats import DatasetHealth, DatasetReport


def _report(
    scene: str = "generic",
    small_ratio: float = 0.0,
    health_problems: list[str] | None = None,
) -> DatasetReport:
    return DatasetReport(
        data_yaml=Path("data.yaml"),
        dataset_root=Path("."),
        scene=scene,
        image_count=10,
        label_count=10,
        class_distribution={"a": 10},
        object_size_ratio={"small": small_ratio, "medium": 1 - small_ratio, "large": 0.0},
        dataset_health=DatasetHealth(score=70, problems=health_problems or []),
    )


def test_small_object_ratio_enables_scale_policy() -> None:
    """Small-object heavy datasets should enable composition/scale augmentations."""
    engine = AugmentationPolicyEngine.from_yaml()

    result = engine.recommend(_report(small_ratio=0.8))

    assert "small_object_ratio_high" in result.matched_rules
    assert {"mosaic", "copy_paste", "random_crop"} <= set(result.actions.enable)


def test_false_positive_errors_reduce_color_blur_and_add_backgrounds() -> None:
    """High false-positive profiles should add negatives and reduce drift augmentations."""
    engine = AugmentationPolicyEngine.from_yaml()

    result = engine.recommend(
        _report(),
        [DetectionErrorObservation(error_type="background_confusion", count=4, severity="high")],
    )

    assert "false_positive_high" in result.matched_rules
    assert {"hsv", "blur"} <= set(result.actions.reduce)
    assert "background_only_images" in result.actions.add


def test_infrared_domain_disables_hsv_and_enables_sensor_augmentations() -> None:
    """Infrared domain should avoid RGB HSV augmentation."""
    engine = AugmentationPolicyEngine.from_yaml()

    result = engine.recommend(_report(scene="infrared_small_target", small_ratio=0.7))

    assert "infrared_domain" in result.matched_rules
    assert "hsv" in result.actions.disable
    assert {"noise", "blur", "contrast_shift"} <= set(result.actions.enable)


def test_health_problems_drive_sampling_and_audit_actions() -> None:
    """Dataset health problems should alter augmentation policy."""
    engine = AugmentationPolicyEngine.from_yaml()

    result = engine.recommend(
        _report(health_problems=["class_imbalance_long_tail", "annotation_noise"])
    )

    assert "long_tail_classes" in result.matched_rules
    assert "annotation_noise" in result.matched_rules
    assert "class_balanced_sampling" in result.actions.add
    assert "label_audit_queue" in result.actions.add
    assert "heavy_mosaic" in result.actions.reduce

