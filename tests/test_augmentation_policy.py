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
    assert {"mosaic", "copy_paste", "random_crop", "multi_scale_training", "tiling_aware_augmentation"} <= set(result.actions.enable)
    assert result.actions.set_params["random_crop"]["box_aware"] is True


def test_false_positive_errors_reduce_color_blur_and_add_backgrounds() -> None:
    """High false-positive profiles should add negatives and reduce drift augmentations."""
    engine = AugmentationPolicyEngine.from_yaml()

    result = engine.recommend(
        _report(),
        [DetectionErrorObservation(error_type="background_confusion", count=4, severity="high")],
    )

    assert "false_positive_high" in result.matched_rules
    assert {"hsv", "blur", "mosaic"} <= set(result.actions.reduce)
    assert {"hard_negative_mining", "background_only_sampling", "background_only_images"} <= set(result.actions.add)
    assert result.actions.set_params["mosaic"]["max_strength"] == 0.35


def test_infrared_domain_disables_hsv_and_enables_sensor_augmentations() -> None:
    """Infrared domain should avoid RGB HSV augmentation."""
    engine = AugmentationPolicyEngine.from_yaml()

    result = engine.recommend(_report(scene="infrared_small_target", small_ratio=0.7))

    assert "infrared_domain" in result.matched_rules
    assert "hsv" in result.actions.disable
    assert {"gaussian_noise", "bounded_blur", "contrast_shift"} <= set(result.actions.enable)
    assert result.actions.set_params["hsv"]["enabled"] is False


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


def test_industrial_defect_prefers_texture_preserving_policy() -> None:
    """Industrial defect scenarios should keep augmentation mild and texture-preserving."""
    engine = AugmentationPolicyEngine.from_yaml()

    result = engine.recommend(_report(scene="industrial_defect"))

    assert "industrial_defect_texture_preserving" in result.matched_rules
    assert "mild_affine" in result.actions.enable
    assert "texture_preserving_resize" in result.actions.enable
    assert {"cutout", "color_jitter", "heavy_mosaic"} <= set(result.actions.reduce)
    assert result.actions.set_params["affine"]["degrees"] == 3


def test_small_object_miss_error_adds_targeted_policy() -> None:
    """Small-object miss observations should trigger targeted small-object augmentation."""
    engine = AugmentationPolicyEngine.from_yaml()

    result = engine.recommend(
        _report(small_ratio=0.2),
        [DetectionErrorObservation(error_type="small_object_miss", count=2, severity="medium")],
    )

    assert "small_object_miss_errors" in result.matched_rules
    assert "small_object_oversampling" in result.actions.add
    assert result.actions.set_params["copy_paste"]["prefer_small_objects"] is True
