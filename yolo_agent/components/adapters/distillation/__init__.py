"""Distillation component adapters."""

from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    YOLO26DistillationAdapter,
    YOLO26DistillationRuntimePlugin,
)

__all__ = ["YOLO26DistillationAdapter", "YOLO26DistillationRuntimePlugin"]
