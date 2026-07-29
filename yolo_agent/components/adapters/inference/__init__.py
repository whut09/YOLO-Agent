"""Inference-only component adapters."""

from yolo_agent.components.adapters.inference.artifacts import (
    SlicingArtifactPaths,
    write_slicing_artifacts,
)
from yolo_agent.components.adapters.inference.sahi_backend import (
    SahiSlicingBackend,
    SlicingImage,
)
from yolo_agent.components.adapters.inference.sliced_coco import (
    SlicedCocoMetrics,
    evaluate_sliced_coco,
)
from yolo_agent.components.adapters.inference.slicing import (
    SlicingInferenceAdapter,
    SlicingInferenceConfig,
    SlicingInferenceMetrics,
    SlicingInferenceProtocol,
    SlicingInferenceResult,
)

__all__ = [
    "SahiSlicingBackend",
    "SlicedCocoMetrics",
    "SlicingArtifactPaths",
    "SlicingInferenceAdapter",
    "SlicingInferenceConfig",
    "SlicingInferenceMetrics",
    "SlicingInferenceProtocol",
    "SlicingInferenceResult",
    "SlicingImage",
    "evaluate_sliced_coco",
    "write_slicing_artifacts",
]
