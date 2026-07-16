"""Framework-agnostic YOLO26 teacher-student distillation losses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DistillationWeights(BaseModel):
    logits: float = Field(default=1.0, ge=0.0)
    feature: float = Field(default=1.0, ge=0.0)
    localization: float = Field(default=1.0, ge=0.0)
    temperature: float = Field(default=2.0, gt=0.0)


def _aligned(student: Any, teacher: Any) -> tuple[Any, Any]:
    import torch
    if not hasattr(student, "shape") or not hasattr(teacher, "shape"):
        raise TypeError("distillation tensors must expose shape")
    if student.ndim != teacher.ndim or student.shape[0] != teacher.shape[0]:
        raise ValueError(f"incompatible distillation shapes: {tuple(student.shape)} vs {tuple(teacher.shape)}")
    if student.shape[1:] != teacher.shape[1:]:
        if student.shape[1] != teacher.shape[1]:
            raise ValueError("channel dimensions require an explicit feature projector")
        teacher = torch.nn.functional.adaptive_avg_pool1d(teacher.reshape(teacher.shape[0], teacher.shape[1], -1), student.reshape(student.shape[0], student.shape[1], -1).shape[-1]).reshape_as(student)
    return student, teacher


def distillation_loss(student_logits: Any, teacher_logits: Any, *, student_features: Any | None = None, teacher_features: Any | None = None, student_boxes: Any | None = None, teacher_boxes: Any | None = None, weights: DistillationWeights | None = None) -> dict[str, Any]:
    import torch
    weights = weights or DistillationWeights()
    student_logits, teacher_logits = _aligned(student_logits, teacher_logits)
    temperature = weights.temperature
    logits = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(student_logits / temperature, dim=-1), torch.nn.functional.softmax(teacher_logits.detach() / temperature, dim=-1), reduction="batchmean") * (temperature**2)
    feature = student_logits.new_zeros(())
    if student_features is not None or teacher_features is not None:
        if student_features is None or teacher_features is None:
            raise ValueError("student_features and teacher_features must be provided together")
        student_features, teacher_features = _aligned(student_features, teacher_features)
        feature = torch.nn.functional.mse_loss(student_features, teacher_features.detach())
    localization = student_logits.new_zeros(())
    if student_boxes is not None or teacher_boxes is not None:
        if student_boxes is None or teacher_boxes is None:
            raise ValueError("student_boxes and teacher_boxes must be provided together")
        student_boxes, teacher_boxes = _aligned(student_boxes, teacher_boxes)
        localization = torch.nn.functional.smooth_l1_loss(student_boxes, teacher_boxes.detach())
    total = weights.logits * logits + weights.feature * feature + weights.localization * localization
    return {"total": total, "logits": logits, "feature": feature, "localization": localization}


class YOLO26DistillationLoss:
    name = "yolo26_distillation"

    def __init__(self, weights: DistillationWeights | None = None) -> None:
        self.weights = weights or DistillationWeights()

    def __call__(self, supervised_loss: Any, **outputs: Any) -> tuple[Any, dict[str, Any]]:
        terms = distillation_loss(weights=self.weights, **outputs)
        return supervised_loss + terms["total"], terms


__all__ = ["DistillationWeights", "YOLO26DistillationLoss", "distillation_loss"]
