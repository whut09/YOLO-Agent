from __future__ import annotations

import pytest

from yolo_agent.components.adapters.assigners.yolo26_assignment import (
    ASSIGNMENT_SPECS,
    AssignmentPaperPrior,
    AssignmentRuntimeConfig,
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


def test_runtime_config_requires_complete_dual_path_scope() -> None:
    values = {
        "component_id": "assigner.dual_path",
        "method": "dual_path",
        "changed_variable": "assignment.dual_path.mode",
        "paper_prior": AssignmentPaperPrior(
            paper_id="method-profile:dual-path-assignment",
            adaptation="test",
        ),
    }

    assert AssignmentRuntimeConfig(
        **values,
        assignment_path="both",
    ).assignment_path == "both"
    with pytest.raises(ValueError, match="requires assignment_path=both"):
        AssignmentRuntimeConfig(**values, assignment_path="one_to_many")


def test_single_path_runtime_rejects_both_scope() -> None:
    with pytest.raises(ValueError, match="does not support requested path scope"):
        AssignmentRuntimeConfig(
            component_id="assigner.dynamic_topk",
            method="dynamic_topk",
            changed_variable="assignment.one_to_many.dynamic_topk.mode",
            assignment_path="both",
            paper_prior=AssignmentPaperPrior(
                paper_id="method-profile:dynamic-topk",
                adaptation="test",
            ),
        )
