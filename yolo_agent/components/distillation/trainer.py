"""Trainer hook boundary for explicit teacher-student loss injection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class DistillationBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    student_logits: Any
    teacher_logits: Any
    student_features: Any | None = None
    teacher_features: Any | None = None
    student_boxes: Any | None = None
    teacher_boxes: Any | None = None


class DistillationTrainerHook:
    """Add distillation loss while keeping teacher frozen and in eval mode."""

    def __init__(self, teacher: Any, loss_plugin: Any) -> None:
        self.teacher = teacher
        self.loss_plugin = loss_plugin
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def teacher_forward(self, inputs: Any) -> Any:
        import torch
        with torch.no_grad():
            return self.teacher(inputs)

    def add_loss(self, supervised_loss: Any, batch: DistillationBatch) -> tuple[Any, dict[str, Any]]:
        return self.loss_plugin(supervised_loss, **batch.model_dump())


class MockDistillationTrainer:
    def __init__(self, hook: DistillationTrainerHook) -> None:
        self.hook = hook
        self.last_terms: dict[str, Any] = {}

    def train_step(self, supervised_loss: Any, batch: DistillationBatch) -> Any:
        loss, self.last_terms = self.hook.add_loss(supervised_loss, batch)
        loss.backward()
        return loss


__all__ = ["DistillationBatch", "DistillationTrainerHook", "MockDistillationTrainer"]
