"""Framework-neutral losses for independent distillation mechanisms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from yolo_agent.components.distillation.mechanisms import DistillationMechanism


@dataclass(frozen=True)
class DistillationInputs:
    student_logits: Any
    teacher_logits: Any
    student_features: Any | None = None
    teacher_features: Any | None = None
    student_boxes: Any | None = None
    teacher_boxes: Any | None = None


@dataclass(frozen=True)
class DistillationLossOutput:
    loss: Any
    metrics: dict[str, float] = field(default_factory=dict)


class DistillationMechanismLoss(ABC):
    mechanism: DistillationMechanism

    @abstractmethod
    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        """Return one unweighted training-only distillation scalar."""


class LogitsDistillationLoss(DistillationMechanismLoss):
    mechanism = "logits"

    def __init__(self, *, temperature: float = 2.0, class_dim: int = -1) -> None:
        if temperature <= 0.0:
            raise ValueError("distillation temperature must be positive")
        self.temperature = temperature
        self.class_dim = class_dim

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        student, teacher = _same_shape(inputs.student_logits, inputs.teacher_logits)
        temperature = self.temperature
        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(
                student.float() / temperature,
                dim=self.class_dim,
            ),
            torch.nn.functional.softmax(
                teacher.detach().float() / temperature,
                dim=self.class_dim,
            ),
            reduction="batchmean",
        ) * (temperature**2)
        return DistillationLossOutput(
            loss=loss,
            metrics={"temperature": temperature},
        )


class LocalizationDistillationLoss(DistillationMechanismLoss):
    mechanism = "localization"

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_boxes is None or inputs.teacher_boxes is None:
            raise ValueError("localization distillation requires student and teacher boxes")
        student, teacher = _same_shape(inputs.student_boxes, inputs.teacher_boxes)
        loss = torch.nn.functional.smooth_l1_loss(
            student.float(),
            teacher.detach().float(),
        )
        return DistillationLossOutput(
            loss=loss,
            metrics={"box_element_count": float(student.numel())},
        )


def build_distillation_mechanism_loss(
    mechanism: DistillationMechanism,
    **options: Any,
) -> DistillationMechanismLoss:
    implementations: dict[str, type[DistillationMechanismLoss]] = {
        LogitsDistillationLoss.mechanism: LogitsDistillationLoss,
        LocalizationDistillationLoss.mechanism: LocalizationDistillationLoss,
    }
    try:
        implementation = implementations[mechanism]
    except KeyError as exc:
        raise KeyError(f"distillation mechanism loss is not implemented: {mechanism}") from exc
    return implementation(**options)


def _same_shape(student: Any, teacher: Any) -> tuple[Any, Any]:
    if not hasattr(student, "shape") or not hasattr(teacher, "shape"):
        raise TypeError("distillation tensors must expose shape")
    if tuple(student.shape) != tuple(teacher.shape):
        raise ValueError(
            "distillation tensors require identical shapes: "
            f"{tuple(student.shape)} vs {tuple(teacher.shape)}"
        )
    return student, teacher


__all__ = [
    "DistillationInputs",
    "DistillationLossOutput",
    "DistillationMechanismLoss",
    "LocalizationDistillationLoss",
    "LogitsDistillationLoss",
    "build_distillation_mechanism_loss",
]
