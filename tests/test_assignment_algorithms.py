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
