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
from yolo_agent.components.adapters.data_pipeline.hard_negative_evidence import (
    TrainHardNegativePrediction,
    TrainHardNegativePredictionBatch,
    TrainSampleIndex,
    TrainSampleIndexRecord,
    produce_train_hard_negative_manifest,
    train_sample_index_from_records,
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
    compute_exposure_details,
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
    "TrainHardNegativePrediction",
    "TrainHardNegativePredictionBatch",
    "TrainSampleIndex",
    "TrainSampleIndexRecord",
    "bound_exposure",
    "compute_exposure",
    "compute_exposure_details",
    "produce_train_hard_negative_manifest",
    "train_sample_index_from_records",
]
