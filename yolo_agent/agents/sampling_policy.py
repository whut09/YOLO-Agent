"""Sampling and dataset reconstruction policy recommendations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.core.dataset_split import DatasetSplitPlan
from yolo_agent.tools.dataset_stats import DatasetReport


SamplingActionType = Literal[
    "long_tail_class_rebalancing",
    "hard_negative_sampling",
    "background_only_injection",
    "scene_balanced_split",
    "small_object_oversampling",
    "duplicate_frame_filtering",
    "train_val_leakage_fix",
]


class SamplingAction(BaseModel):
    """One sampling/reconstruction action."""

    action_type: SamplingActionType
    priority: float = Field(default=1.0, ge=0.0)
    target: str = ""
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    rationale: str
    risks: list[str] = Field(default_factory=list)


class SamplingPolicyPlan(BaseModel):
    """Sampling policy output for a dataset profile."""

    actions: list[SamplingAction] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)


class SamplingPolicyEngine:
    """Recommend sampling and split policies from dataset and error evidence."""

    def recommend(
        self,
        dataset_report: DatasetReport,
        error_profile: list[DetectionErrorObservation] | None = None,
        split_plan: DatasetSplitPlan | None = None,
    ) -> SamplingPolicyPlan:
        """Create a prioritized sampling policy plan."""
        observations = error_profile or []
        actions: list[SamplingAction] = []
        problems = set(dataset_report.dataset_health.problems)

        if "class_imbalance_long_tail" in problems or _is_long_tail(dataset_report.class_distribution):
            actions.append(
                SamplingAction(
                    action_type="long_tail_class_rebalancing",
                    priority=8.0,
                    target="minority_classes",
                    parameters={"max_oversample_factor": 3.0, "preserve_validation_distribution": True},
                    rationale="Class distribution is long-tailed; rebalance training without changing validation priors.",
                    risks=["Oversampling can overfit rare appearances."],
                )
            )

        if "missing_hard_backgrounds" in problems or _has_error(observations, {"background_confusion", "hard_negative"}):
            actions.append(
                SamplingAction(
                    action_type="hard_negative_sampling",
                    priority=7.0 + _error_count(observations, {"background_confusion", "hard_negative"}),
                    target="false_positive_backgrounds",
                    parameters={"max_negative_fraction": 0.25},
                    rationale="Background false positives need hard negatives in the training stream.",
                    risks=["Too many negatives can lower recall."],
                )
            )
            actions.append(
                SamplingAction(
                    action_type="background_only_injection",
                    priority=6.0,
                    target="empty_images",
                    parameters={"target_empty_image_ratio": 0.1},
                    rationale="Background-only images help calibrate precision and suppress clutter detections.",
                    risks=["Empty images should be intentional and have empty label files."],
                )
            )

        small_ratio = dataset_report.object_size_ratio.get("small", 0.0)
        if small_ratio > 0.5 or _has_error(observations, {"small_object_miss"}):
            actions.append(
                SamplingAction(
                    action_type="small_object_oversampling",
                    priority=7.5 + small_ratio * 2.0,
                    target="images_with_small_objects",
                    parameters={"max_oversample_factor": 2.5, "min_small_area": 0.01},
                    rationale="Small-object-heavy data benefits from retained tiny-instance exposure.",
                    risks=["Repeated tiny objects can inflate validation expectations if leakage exists."],
                )
            )

        if split_plan is not None and split_plan.duplicates:
            actions.append(
                SamplingAction(
                    action_type="duplicate_frame_filtering",
                    priority=9.0,
                    target="duplicate_fingerprints",
                    parameters={"duplicate_groups": len(split_plan.duplicates)},
                    rationale="Duplicate frames bias metrics and can over-weight near-identical scenes.",
                    risks=["Keep temporal diversity when filtering video frames."],
                )
            )
        elif "high_duplicate_frames" in problems:
            actions.append(
                SamplingAction(
                    action_type="duplicate_frame_filtering",
                    priority=8.5,
                    target="near_duplicate_frames",
                    parameters={"use_split_plan": True},
                    rationale="Dataset health score indicates duplicate frames.",
                    risks=["Fingerprint heuristics should be reviewed before deletion."],
                )
            )

        if split_plan is not None and split_plan.leakage:
            actions.append(
                SamplingAction(
                    action_type="train_val_leakage_fix",
                    priority=10.0,
                    target="leaked_fingerprints",
                    parameters={"leakage_pairs": len(split_plan.leakage)},
                    rationale="Duplicate fingerprints cross train/val/test boundaries.",
                    risks=["Do not compare experiments until leakage is fixed."],
                )
            )
        elif "train_val_leakage" in problems:
            actions.append(
                SamplingAction(
                    action_type="train_val_leakage_fix",
                    priority=10.0,
                    target="train_val_split",
                    parameters={"use_split_plan": True},
                    rationale="Dataset health score indicates train/val leakage.",
                    risks=["Existing metrics may be over-optimistic."],
                )
            )

        actions.append(
            SamplingAction(
                action_type="scene_balanced_split",
                priority=5.0,
                target="scene_groups",
                parameters={"train_ratio": 0.8, "val_ratio": 0.2},
                rationale="Scene-balanced splits make model comparisons less dependent on accidental scene skew.",
                risks=["Very small scene groups may need manual split constraints."],
            )
        )

        actions = sorted(_dedupe_actions(actions), key=lambda action: action.priority, reverse=True)
        return SamplingPolicyPlan(
            actions=actions,
            recommendations=[action.rationale for action in actions],
            required_checks=_required_checks(actions),
        )


def _is_long_tail(class_distribution: dict[str, int]) -> bool:
    nonzero = [count for count in class_distribution.values() if count > 0]
    return len(nonzero) > 1 and min(nonzero) / max(nonzero) < 0.2


def _has_error(observations: list[DetectionErrorObservation], error_types: set[str]) -> bool:
    return any(observation.error_type in error_types for observation in observations)


def _error_count(observations: list[DetectionErrorObservation], error_types: set[str]) -> int:
    return sum(observation.count for observation in observations if observation.error_type in error_types)


def _dedupe_actions(actions: list[SamplingAction]) -> list[SamplingAction]:
    by_type: dict[SamplingActionType, SamplingAction] = {}
    for action in actions:
        existing = by_type.get(action.action_type)
        if existing is None or action.priority > existing.priority:
            by_type[action.action_type] = action
    return list(by_type.values())


def _required_checks(actions: list[SamplingAction]) -> list[str]:
    checks: list[str] = []
    action_types = {action.action_type for action in actions}
    if "duplicate_frame_filtering" in action_types:
        checks.append("review_duplicate_groups")
    if "train_val_leakage_fix" in action_types:
        checks.append("block_experiment_comparison_until_leakage_fixed")
    if "background_only_injection" in action_types:
        checks.append("ensure_empty_label_files_for_background_images")
    if "long_tail_class_rebalancing" in action_types:
        checks.append("preserve_validation_class_distribution")
    return checks
