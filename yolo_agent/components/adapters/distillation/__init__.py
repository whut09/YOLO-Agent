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

__all__ = [
    "DistillationMethodRegistry",
    "DistillationTeacherMissingError",
    "YOLO26DistillationAdapter",
    "YOLO26DistillationRuntimePlugin",
    "default_distillation_method_registry",
]
