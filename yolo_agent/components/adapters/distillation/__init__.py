"""Distillation component adapters."""

from yolo_agent.components.adapters.distillation.method_registry import (
    DistillationMethodRegistry,
    DistillationTeacherMissingError,
    default_distillation_method_registry,
)
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    YOLO26DistillationAdapter,
    YOLO26DistillationRuntimePlugin,
)
from yolo_agent.components.adapters.distillation.teacher_evidence import (
    CheckpointIdentity,
    CheckpointMetadata,
    CheckpointResolution,
    resolve_checkpoint_identity,
    resolve_student_checkpoint,
    resolve_teacher_checkpoint,
)

__all__ = [
    "DistillationMethodRegistry",
    "DistillationTeacherMissingError",
    "YOLO26DistillationAdapter",
    "YOLO26DistillationRuntimePlugin",
    "CheckpointIdentity",
    "CheckpointMetadata",
    "CheckpointResolution",
    "default_distillation_method_registry",
    "resolve_checkpoint_identity",
    "resolve_student_checkpoint",
    "resolve_teacher_checkpoint",
]
