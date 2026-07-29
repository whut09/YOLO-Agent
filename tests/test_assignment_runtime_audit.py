"""Installed YOLO26 assignment contract audit tests."""

from __future__ import annotations

from tests.assignment_fixtures import native_model_and_criterion
from yolo_agent.components.adapters.assigners.yolo26_assignment import (
    audit_yolo26_assignment_runtime,
)


def test_installed_yolo26_assignment_audit_covers_both_paths_and_stal_behavior() -> None:
    model, criterion = native_model_and_criterion()

    audit = audit_yolo26_assignment_runtime(model, criterion)

    assert audit.ultralytics_version == "8.4.87"
    assert audit.criterion_class == "E2ELoss"
    assert audit.one_to_many.assigner_class == "TaskAlignedAssigner"
    assert audit.one_to_many.topk == audit.one_to_many.topk2 == 10
    assert audit.one_to_one.topk == 7
    assert audit.one_to_one.topk2 == 1
    assert audit.stal_spatial_behavior_verified is True
    assert "no distinct STAL class" in audit.stal_runtime_form
    assert audit.nms_free is True
    assert audit.dfl_free is True
    assert audit.one_to_many.use_dfl is False
    assert audit.one_to_one.use_dfl is False
    assert audit.native_loss_outputs == [
        "box_loss",
        "cls_loss",
        "dfl_slot_zero_when_reg_max_1",
    ]
    assert audit.verified is True
