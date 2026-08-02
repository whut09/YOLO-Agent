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


@pytest.mark.parametrize(
    "mechanism", ["logits", "feature", "localization", "relation"]
)
def test_base_distillation_losses_shape_backward_and_amp(mechanism: str) -> None:
    student_logits = torch.randn(2, 4, 7, requires_grad=True)
    student_boxes = torch.randn(2, 4, 7, requires_grad=True)
    student_features = [torch.randn(2, 5, 8, 8, requires_grad=True)]
    teacher_features = [torch.randn(2, 7, 8, 8, requires_grad=True)]
    inputs = DistillationInputs(
        student_logits=student_logits,
        teacher_logits=torch.randn(2, 4, 7, requires_grad=True),
        student_features=student_features,
        teacher_features=teacher_features,
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
    elif mechanism == "localization":
        assert student_boxes.grad is not None
        assert inputs.teacher_boxes.grad is None
    else:
        assert student_features[0].grad is not None
        assert teacher_features[0].grad is None


@pytest.mark.parametrize("mechanism", ["attention", "masked_feature"])
def test_attention_distillation_losses_backward_without_teacher_grad(
    mechanism: str,
) -> None:
    student = [torch.randn(2, 5, 8, 8, requires_grad=True)]
    teacher = [torch.randn(2, 9, 6, 6, requires_grad=True)]
    inputs = DistillationInputs(
        student_logits=torch.randn(2, 3, 4),
        teacher_logits=torch.randn(2, 3, 4),
        student_features=student,
        teacher_features=teacher,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = build_distillation_mechanism_loss(mechanism).compute(inputs)
    output.loss.backward()

    assert student[0].grad is not None
    assert teacher[0].grad is None
    assert output.metrics["feature_level_count"] == 1.0


def test_masked_feature_distillation_uses_bounded_teacher_mask() -> None:
    inputs = DistillationInputs(
        student_logits=torch.randn(1, 2, 4),
        teacher_logits=torch.randn(1, 2, 4),
        student_features=[torch.randn(1, 4, 4, 4)],
        teacher_features=[torch.randn(1, 4, 4, 4)],
    )

    output = build_distillation_mechanism_loss(
        "masked_feature", mask_ratio=0.25
    ).compute(inputs)

    assert output.metrics["masked_position_count"] == 4.0
    assert output.metrics["masked_position_fraction"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="ratio"):
        build_distillation_mechanism_loss("masked_feature", mask_ratio=0.0)


def test_quality_aware_distillation_weights_teacher_confidence() -> None:
    student = torch.randn(2, 3, 5, requires_grad=True)
    teacher = torch.randn(2, 3, 5, requires_grad=True)
    output = build_distillation_mechanism_loss(
        "quality_aware", class_dim=1
    ).compute(
        DistillationInputs(student_logits=student, teacher_logits=teacher)
    )

    output.loss.backward()

    assert student.grad is not None
    assert teacher.grad is None
    assert 0.0 < output.metrics["mean_teacher_quality"] <= 1.0


def test_teacher_ensemble_averages_multiple_frozen_teacher_profiles() -> None:
    student = torch.randn(2, 3, 5, requires_grad=True)
    teachers = [
        torch.randn(2, 3, 5, requires_grad=True),
        torch.randn(2, 3, 5, requires_grad=True),
    ]
    output = build_distillation_mechanism_loss(
        "teacher_ensemble", class_dim=1
    ).compute(
        DistillationInputs(student_logits=student, teacher_logits=teachers)
    )

    output.loss.backward()

    assert student.grad is not None
    assert all(teacher.grad is None for teacher in teachers)
    assert output.metrics["teacher_count"] == 2.0
    with pytest.raises(ValueError, match="at least two teachers"):
        build_distillation_mechanism_loss("teacher_ensemble").compute(
            DistillationInputs(
                student_logits=torch.randn(2, 3),
                teacher_logits=[torch.randn(2, 3)],
            )
        )


def test_relation_distillation_bounds_quadratic_spatial_matrix() -> None:
    inputs = DistillationInputs(
        student_logits=torch.randn(1, 2, 4),
        teacher_logits=torch.randn(1, 2, 4),
        student_features=[torch.randn(1, 4, 40, 40, requires_grad=True)],
        teacher_features=[torch.randn(1, 8, 40, 40)],
    )

    output = build_distillation_mechanism_loss(
        "relation", max_spatial_tokens=64
    ).compute(inputs)

    assert output.metrics["max_relation_tokens"] <= 64


def test_base_distillation_losses_reject_incompatible_inputs() -> None:
    inputs = DistillationInputs(
        student_logits=torch.randn(2, 4),
        teacher_logits=torch.randn(3, 4),
    )
    with pytest.raises(ValueError, match="identical shapes"):
        build_distillation_mechanism_loss("logits").compute(inputs)
    with pytest.raises(ValueError, match="requires student and teacher boxes"):
        build_distillation_mechanism_loss("localization").compute(inputs)
