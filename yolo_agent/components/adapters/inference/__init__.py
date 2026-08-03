"""Inference-only component adapters."""

from yolo_agent.components.adapters.inference.artifacts import (
    SlicingArtifactPaths,
    write_slicing_artifacts,
)
from yolo_agent.components.adapters.inference.backend import (
    BackendResult,
    InferenceImage,
    UltralyticsInferenceBackend,
)
from yolo_agent.components.adapters.inference.plugin import (
    ClassAwareThresholdInferenceAdapter,
    ConfidenceCalibrationInferenceAdapter,
    InferencePolicyPlugin,
    MergePolicyInferenceAdapter,
    TestTimeAugmentationAdapter,
    TiledMultiScaleInferenceAdapter,
)
from yolo_agent.components.adapters.inference.policy import (
    InferencePolicyConfig,
    InferencePolicyMetrics,
    InferencePolicyProtocol,
    InferencePolicyResult,
    InferenceResourceMetrics,
    protocol_from_policy,
)
from yolo_agent.components.adapters.inference.policy_artifacts import (
    InferencePolicyArtifactPaths,
    write_inference_policy_artifacts,
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
    "BackendResult",
    "ClassAwareThresholdInferenceAdapter",
    "ConfidenceCalibrationInferenceAdapter",
    "InferenceImage",
    "InferencePolicyArtifactPaths",
    "InferencePolicyConfig",
    "InferencePolicyMetrics",
    "InferencePolicyPlugin",
    "InferencePolicyProtocol",
    "InferencePolicyResult",
    "InferenceResourceMetrics",
    "MergePolicyInferenceAdapter",
    "SahiSlicingBackend",
    "SlicedCocoMetrics",
    "SlicingArtifactPaths",
    "SlicingInferenceAdapter",
    "SlicingInferenceConfig",
    "SlicingInferenceMetrics",
    "SlicingInferenceProtocol",
    "SlicingInferenceResult",
    "SlicingImage",
    "TestTimeAugmentationAdapter",
    "TiledMultiScaleInferenceAdapter",
    "UltralyticsInferenceBackend",
    "evaluate_sliced_coco",
    "protocol_from_policy",
    "write_inference_policy_artifacts",
    "write_slicing_artifacts",
]
