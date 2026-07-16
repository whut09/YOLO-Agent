"""Inference-only component adapters."""

from yolo_agent.components.adapters.inference.slicing import (
    SlicingInferenceAdapter,
    SlicingInferenceConfig,
    SlicingInferenceMetrics,
    SlicingInferenceProtocol,
    SlicingInferenceResult,
)

__all__ = [
    "SlicingInferenceAdapter",
    "SlicingInferenceConfig",
    "SlicingInferenceMetrics",
    "SlicingInferenceProtocol",
    "SlicingInferenceResult",
]
