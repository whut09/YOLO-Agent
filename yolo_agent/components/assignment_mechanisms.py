"""Stable identities for reusable, point-based YOLO26 assignment mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AssignmentRuntimeMethod = Literal[
    "native",
    "task_aligned_weighting",
    "dynamic_topk",
    "quality_aware",
    "soft_label",
    "dual_path",
    "conflict_aware",
]
AssignmentTargetPath = Literal["one_to_many", "one_to_one"]


@dataclass(frozen=True)
class AssignmentMechanismSpec:
    component_id: str
    method: AssignmentRuntimeMethod
    display_name: str
    changed_variable: str
    supported_paths: tuple[AssignmentTargetPath, ...]
    shadow_required: bool = True
    replaces_head: bool = False
    replaces_loss: bool = False
    changes_inference_path: bool = False


ASSIGNMENT_MECHANISMS = {
    item.component_id: item
    for item in (
        AssignmentMechanismSpec(
            component_id="assigner.native_yolo26",
            method="native",
            display_name="Native YOLO26 assignment baseline",
            changed_variable="assignment.native.baseline",
            supported_paths=("one_to_many", "one_to_one"),
            shadow_required=False,
        ),
        AssignmentMechanismSpec(
            component_id="assigner.task_aligned_weighting",
            method="task_aligned_weighting",
            display_name="Task-aligned assignment weighting",
            changed_variable="assignment.one_to_many.task_aligned_weighting.mode",
            supported_paths=("one_to_many",),
        ),
        AssignmentMechanismSpec(
            component_id="assigner.dynamic_topk",
            method="dynamic_topk",
            display_name="Dynamic top-k matching",
            changed_variable="assignment.one_to_many.dynamic_topk.mode",
            supported_paths=("one_to_many",),
        ),
        AssignmentMechanismSpec(
            component_id="assigner.quality_aware",
            method="quality_aware",
            display_name="Quality-aware matching",
            changed_variable="assignment.one_to_many.quality_aware.mode",
            supported_paths=("one_to_many",),
        ),
        AssignmentMechanismSpec(
            component_id="assigner.soft_label",
            method="soft_label",
            display_name="Soft label assignment",
            changed_variable="assignment.one_to_many.soft_label.mode",
            supported_paths=("one_to_many",),
        ),
        AssignmentMechanismSpec(
            component_id="assigner.dual_path",
            method="dual_path",
            display_name="Dual-path assignment adaptation",
            changed_variable="assignment.dual_path.mode",
            supported_paths=("one_to_many", "one_to_one"),
        ),
        AssignmentMechanismSpec(
            component_id="assigner.conflict_aware",
            method="conflict_aware",
            display_name="Conflict-aware positive selection",
            changed_variable="assignment.one_to_many.conflict_aware.mode",
            supported_paths=("one_to_many",),
        ),
    )
}

ASSIGNMENT_METHODS = {
    item.method: item for item in ASSIGNMENT_MECHANISMS.values()
}


__all__ = [
    "ASSIGNMENT_MECHANISMS",
    "ASSIGNMENT_METHODS",
    "AssignmentMechanismSpec",
    "AssignmentRuntimeMethod",
    "AssignmentTargetPath",
]
