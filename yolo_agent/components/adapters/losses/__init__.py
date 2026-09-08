"""Executable auxiliary loss adapters."""

from yolo_agent.components.adapters.losses.quality_alignment import (
    QualityAlignmentAuxiliaryLossAdapter,
    QualityAlignmentRuntimePlugin,
)
from yolo_agent.components.adapters.losses.mutual_supervision import (
    MutualSupervisionAdapter,
    MutualSupervisionRuntimePlugin,
)

__all__ = [
    "MutualSupervisionAdapter",
    "MutualSupervisionRuntimePlugin",
    "QualityAlignmentAuxiliaryLossAdapter",
    "QualityAlignmentRuntimePlugin",
]
