from __future__ import annotations

import pytest
import torch

from yolo_agent.components.distillation.mechanism_losses import (
    DistillationInputs,
    build_distillation_mechanism_loss,
)
from yolo_agent.components.distillation.mechanisms import (
    DISTILLATION_COMPONENTS,
    DISTILLATION_MECHANISMS,
)


def test_distillation_mechanisms_have_independent_runtime_identities() -> None:
    assert set(DISTILLATION_MECHANISMS) == {
        "logits",
        "feature",
        "localization",
        "relation",
        "attention",
        "masked_feature",
        "quality_aware",
        "teacher_ensemble",
    }
    assert len(DISTILLATION_COMPONENTS) == 8
    assert len(
        {item.changed_variable for item in DISTILLATION_MECHANISMS.values()}
    ) == 8
    assert all(
        item.changed_variable == f"loss.distillation.{item.mechanism}.weight"
        for item in DISTILLATION_MECHANISMS.values()
    )


def test_distillation_mechanism_requirements_are_explicit() -> None:
    assert DISTILLATION_MECHANISMS["feature"].requires_features
    assert DISTILLATION_MECHANISMS["relation"].requires_features
    assert DISTILLATION_MECHANISMS["attention"].requires_features
    assert DISTILLATION_MECHANISMS["masked_feature"].requires_features
    assert DISTILLATION_MECHANISMS["localization"].requires_boxes
    assert DISTILLATION_MECHANISMS["teacher_ensemble"].requires_multiple_teachers
    assert not DISTILLATION_MECHANISMS["logits"].requires_features


@pytest.mark.parametrize("mechanism", ["logits", "localization"])
def test_base_distillation_losses_shape_backward_and_amp(mechanism: str) -> None:
    student_logits = torch.randn(2, 4, 7, requires_grad=True)
    student_boxes = torch.randn(2, 4, 7, requires_grad=True)
    inputs = DistillationInputs(
        student_logits=student_logits,
        teacher_logits=torch.randn(2, 4, 7, requires_grad=True),
        student_boxes=student_boxes,
        teacher_boxes=torch.randn(2, 4, 7, requires_grad=True),
    )
    options = {"class_dim": 1} if mechanism == "logits" else {}

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = build_distillation_mechanism_loss(mechanism, **options).compute(
            inputs
        )
    output.loss.backward()

    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    if mechanism == "logits":
        assert student_logits.grad is not None
        assert inputs.teacher_logits.grad is None
    else:
        assert student_boxes.grad is not None
        assert inputs.teacher_boxes.grad is None


def test_base_distillation_losses_reject_incompatible_inputs() -> None:
    inputs = DistillationInputs(
        student_logits=torch.randn(2, 4),
        teacher_logits=torch.randn(3, 4),
    )
    with pytest.raises(ValueError, match="identical shapes"):
        build_distillation_mechanism_loss("logits").compute(inputs)
    with pytest.raises(ValueError, match="requires student and teacher boxes"):
        build_distillation_mechanism_loss("localization").compute(inputs)
