"""Independent YOLO26 detection-head graph adapters."""

from yolo_agent.components.adapters.heads.task_aligned import (
    TaskAlignedDetectionHead,
    TaskAlignedHeadAdapter,
    TaskAlignedHeadConfig,
    TaskAlignedHeadRuntimePlugin,
)

__all__ = [
    "TaskAlignedDetectionHead",
    "TaskAlignedHeadAdapter",
    "TaskAlignedHeadConfig",
    "TaskAlignedHeadRuntimePlugin",
]
