from __future__ import annotations

from yolo_agent.components.assignment_mechanisms import (
    ASSIGNMENT_MECHANISMS,
    ASSIGNMENT_METHODS,
)


def test_reusable_assignment_mechanisms_have_stable_independent_identities() -> None:
    assert set(ASSIGNMENT_MECHANISMS) == {
        "assigner.native_yolo26",
        "assigner.task_aligned_weighting",
        "assigner.dynamic_topk",
        "assigner.quality_aware",
        "assigner.soft_label",
        "assigner.dual_path",
        "assigner.conflict_aware",
    }
    assert len(ASSIGNMENT_METHODS) == len(ASSIGNMENT_MECHANISMS)
    assert all(not item.replaces_head for item in ASSIGNMENT_MECHANISMS.values())
    assert all(not item.replaces_loss for item in ASSIGNMENT_MECHANISMS.values())
    assert all(
        not item.changes_inference_path
        for item in ASSIGNMENT_MECHANISMS.values()
    )


def test_only_native_and_dual_path_mechanisms_target_both_assignment_paths() -> None:
    both = {
        item.component_id
        for item in ASSIGNMENT_MECHANISMS.values()
        if set(item.supported_paths) == {"one_to_many", "one_to_one"}
    }

    assert both == {"assigner.native_yolo26", "assigner.dual_path"}
    assert ASSIGNMENT_MECHANISMS["assigner.native_yolo26"].shadow_required is False
    assert ASSIGNMENT_MECHANISMS["assigner.dual_path"].shadow_required is True
