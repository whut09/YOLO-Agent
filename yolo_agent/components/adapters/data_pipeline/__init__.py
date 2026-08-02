"""Reusable train-only data pipeline adapter primitives."""

from yolo_agent.components.adapters.data_pipeline.contracts import (
    DataMechanismKind,
    DataPipelineIdentity,
    DataPipelineManifest,
    DataSampleRecord,
)
from yolo_agent.components.adapters.data_pipeline.adapters import (
    ClassBalancedSamplingAdapter,
    FalseNegativeClassBoostAdapter,
    HardNegativeReplayAdapter,
    MultiImageSamplingScheduleAdapter,
    ObjectCentricCropAdapter,
    RareClassCopyPasteAdapter,
    RepeatFactorSamplingAdapter,
    ScaleAwareCropAdapter,
    SmallObjectWeightedSamplingAdapter,
)
from yolo_agent.components.adapters.data_pipeline.data_pipeline_plugin import (
    DataPipelinePlugin,
)
from yolo_agent.components.adapters.data_pipeline.sampling import (
    DistributedExposureSampler,
    bound_exposure,
)
from yolo_agent.components.adapters.data_pipeline.sampling_plugin import SamplingPlugin
from yolo_agent.components.adapters.data_pipeline.exposure import (
    ExposureConfig,
    ExposureMechanism,
    compute_exposure,
)

__all__ = [
    "DataMechanismKind",
    "ClassBalancedSamplingAdapter",
    "DataPipelineIdentity",
    "DataPipelineManifest",
    "DataPipelinePlugin",
    "DataSampleRecord",
    "DistributedExposureSampler",
    "ExposureConfig",
    "ExposureMechanism",
    "FalseNegativeClassBoostAdapter",
    "HardNegativeReplayAdapter",
    "MultiImageSamplingScheduleAdapter",
    "ObjectCentricCropAdapter",
    "RareClassCopyPasteAdapter",
    "RepeatFactorSamplingAdapter",
    "ScaleAwareCropAdapter",
    "SamplingPlugin",
    "SmallObjectWeightedSamplingAdapter",
    "bound_exposure",
    "compute_exposure",
]
