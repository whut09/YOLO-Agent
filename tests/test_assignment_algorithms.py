"""Independent assignment algorithm safety tests."""

from __future__ import annotations

import pytest

from tests.assignment_fixtures import assignment_inputs
from yolo_agent.components.assignment import (
    AssignerInputs,
    YOLO26AssignerPlugin,
    build_yolo26_assigner_plugin,
)


@pytest.mark.parametrize("method", ["tood_tal", "ota", "dsla"])
def test_assignment_methods_are_independent_point_based_implementations(method: str) -> None:
    plugin = build_yolo26_assigner_plugin(method)
    output = plugin.run(assignment_inputs())

    assert isinstance(plugin, YOLO26AssignerPlugin)
    assert output.target_labels.shape == (1, 20)
    assert output.target_boxes_xyxy.shape == (1, 20, 4)
    assert output.target_scores.shape == (1, 20, 2)
    assert output.foreground_mask.shape == (1, 20)
    assert output.foreground_mask.any()
    assert plugin.plugin_id in {
        "tood.task_aligned_learning",
        "ota.optimal_transport",
        "dsla.dynamic_smooth_label",
    }
    assert plugin.paper_id in {
        "arxiv:2108.07755",
        "arxiv:2103.14259",
        "arxiv:2208.00817",
    }
    assert plugin.exact_paper_reproduction is False
    assert plugin.mechanism_id in {
        "assigner.task_aligned",
        "assigner.optimal_transport",
        "assigner.dynamic_smooth_label",
    }
    assert plugin.replaces_head is False
    assert plugin.replaces_loss is False
    assert plugin.changes_inference_path is False


def test_anchor_based_and_one_to_one_paper_plugins_are_rejected() -> None:
    plugin = build_yolo26_assigner_plugin("tood_tal")

    with pytest.raises(ValueError, match="point anchors only"):
        plugin.run(assignment_inputs(anchor_representation="anchor_box"))

    one_to_one = AssignerInputs(
        **{**assignment_inputs().__dict__, "path": "one_to_one"}
    )
    with pytest.raises(ValueError, match="does not support one_to_one"):
        plugin.run(one_to_one)


def test_task_aligned_weighting_is_a_reusable_assignment_only_mechanism() -> None:
    plugin = build_yolo26_assigner_plugin(
        "task_aligned_weighting",
        topk=8,
        classification_weight=0.75,
        localization_weight=5.0,
    )

    output = plugin.run(assignment_inputs())

    assert plugin.mechanism_id == "assigner.task_aligned_weighting"
    assert plugin.paper_id is None
    assert output.foreground_mask.any()
    assert output.target_scores.max() <= 1
    assert plugin.replaces_head is False
    assert plugin.replaces_loss is False


def test_dynamic_topk_uses_point_quality_without_anchor_assumptions() -> None:
    plugin = build_yolo26_assigner_plugin("dynamic_topk", maximum_topk=8)

    output = plugin.run(assignment_inputs())

    assert plugin.mechanism_id == "assigner.dynamic_topk"
    assert 0 < int(output.foreground_mask.sum()) <= 8
    assert output.target_scores.max() <= 1
    assert plugin.anchor_representation == "point"


def test_quality_aware_matching_keeps_targets_bounded_and_assignment_only() -> None:
    plugin = build_yolo26_assigner_plugin(
        "quality_aware",
        topk=6,
        classification_power=1.0,
        iou_power=3.0,
    )

    output = plugin.run(assignment_inputs())

    assert plugin.mechanism_id == "assigner.quality_aware"
    assert 0 < int(output.foreground_mask.sum()) <= 6
    assert 0 <= output.target_scores.min() <= output.target_scores.max() <= 1
    assert plugin.replaces_head is False
    assert plugin.replaces_loss is False


def test_soft_label_assignment_changes_only_positive_quality_targets() -> None:
    plugin = build_yolo26_assigner_plugin(
        "soft_label",
        topk=8,
        minimum_positive_quality=0.1,
        temperature=2.0,
    )

    output = plugin.run(assignment_inputs())
    positive_scores = output.target_scores[output.target_scores > 0]

    assert plugin.mechanism_id == "assigner.soft_label"
    assert positive_scores.numel() > 0
    assert float(positive_scores.min()) >= 0.1
    assert float(positive_scores.max()) <= 1.0
    assert not bool(output.target_scores[~output.foreground_mask].any())


def test_conflict_aware_selection_can_reject_ambiguous_gt_claims() -> None:
    source = assignment_inputs()
    overlapping = AssignerInputs(
        **{
            **source.__dict__,
            "gt_labels": source.gt_labels.new_tensor([[[0.0], [0.0]]]),
            "gt_boxes_xyxy": source.gt_boxes_xyxy.new_tensor(
                [[[0.0, 0.0, 96.0, 96.0], [16.0, 16.0, 112.0, 112.0]]]
            ),
            "gt_mask": source.gt_mask.new_tensor([[[True], [True]]]),
        }
    )
    permissive = build_yolo26_assigner_plugin(
        "conflict_aware", topk=10, minimum_relative_margin=0.0
    )
    guarded = build_yolo26_assigner_plugin(
        "conflict_aware", topk=10, minimum_relative_margin=0.5
    )

    permissive_output = permissive.run(overlapping)
    guarded_output = guarded.run(overlapping)

    assert guarded.mechanism_id == "assigner.conflict_aware"
    assert guarded.last_conflict_candidates > 0
    assert guarded.last_rejected_conflicts > 0
    assert int(guarded_output.foreground_mask.sum()) <= int(
        permissive_output.foreground_mask.sum()
    )
