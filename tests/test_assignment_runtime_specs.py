from __future__ import annotations

from yolo_agent.components.adapters.assigners.yolo26_assignment import (
    ASSIGNMENT_SPECS,
)


def test_runtime_specs_cover_reusable_assignment_mechanisms() -> None:
    assert {
        "assigner.task_aligned_weighting",
        "assigner.dynamic_topk",
        "assigner.quality_aware",
        "assigner.soft_label",
        "assigner.dual_path",
        "assigner.conflict_aware",
    }.issubset(ASSIGNMENT_SPECS)
    assert all(
        item.changed_variable.startswith("assignment.")
        for item in ASSIGNMENT_SPECS.values()
    )


def test_only_dual_path_runtime_spec_can_change_both_training_paths() -> None:
    both = {
        item.component_id
        for item in ASSIGNMENT_SPECS.values()
        if set(item.supported_paths) == {"one_to_many", "one_to_one"}
    }

    assert both == {"assigner.dual_path"}
