"""Explicit native assignment baseline tests."""

from __future__ import annotations

import torch

from tests.assignment_fixtures import assignment_inputs
from yolo_agent.components.assignment import (
    NativeYOLO26AssignerPlugin,
    build_yolo26_assigner_plugin,
    compare_assignments,
)


def test_native_assigner_is_an_explicit_baseline_plugin() -> None:
    inputs = assignment_inputs()

    def native(*_: object) -> tuple[torch.Tensor, ...]:
        foreground = torch.zeros((1, 20), dtype=torch.bool)
        foreground[:, :2] = True
        labels = torch.full((1, 20), 2, dtype=torch.long)
        labels[:, :2] = 0
        boxes = torch.zeros((1, 20, 4))
        scores = torch.zeros((1, 20, 2))
        scores[:, :2, 0] = 1.0
        indices = torch.zeros((1, 20), dtype=torch.long)
        return labels, boxes, scores, foreground, indices

    baseline = NativeYOLO26AssignerPlugin(native).run(inputs)
    candidate = build_yolo26_assigner_plugin("dsla").run(inputs)
    comparison = compare_assignments(baseline, candidate)

    assert NativeYOLO26AssignerPlugin.plugin_id == "yolo26.native_task_aligned"
    assert NativeYOLO26AssignerPlugin.mechanism_id == "assigner.native_yolo26"
    assert comparison.baseline_positive_count == 2
    assert comparison.candidate_positive_count > 0
    assert comparison.total_candidates == 20
    assert 0.0 <= comparison.conflict_rate <= 1.0
    assert 0.0 <= comparison.gt_conflict_rate <= 1.0
    assert comparison.matching_stability == 1.0 - comparison.conflict_rate


def test_native_baseline_explicitly_supports_both_yolo26_paths() -> None:
    assert NativeYOLO26AssignerPlugin.supported_paths == frozenset(
        {"one_to_many", "one_to_one"}
    )
