"""Executable auxiliary loss adapters."""

from yolo_agent.components.adapters.losses.quality_alignment import (
    QualityAlignmentAuxiliaryLossAdapter,
    QualityAlignmentRuntimePlugin,
)

__all__ = ["QualityAlignmentAuxiliaryLossAdapter", "QualityAlignmentRuntimePlugin"]
