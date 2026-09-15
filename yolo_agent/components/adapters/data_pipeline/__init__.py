"""Reusable train-only data pipeline adapter primitives."""

from yolo_agent.components.adapters.data_pipeline.contracts import (
    DataMechanismKind,
    DataPipelineIdentity,
    DataPipelineManifest,
    DataSampleRecord,
)
from yolo_agent.components.adapters.data_pipeline.adapters import (
    ActiveLearningAcquisitionAdapter,
    AnnotationQualityFilterAdapter,
    ClassBalancedSamplingAdapter,
    FalseNegativeClassBoostAdapter,
    HardNegativeReplayAdapter,
    MultiImageSamplingScheduleAdapter,
    NormalizationPreprocessingAdapter,
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
    train_sample_index_from_yolo_data,
    train_sample_index_from_records,
)
from yolo_agent.components.adapters.data_pipeline.data_pipeline_plugin import (
    DataPipelinePlugin,
)
from yolo_agent.components.adapters.data_pipeline.data_side_plugin import (
    ActiveLearningConfig,
    ActiveLearningPlugin,
    AnnotationFilterPlugin,
    PreprocessingPlugin,
)
from yolo_agent.components.adapters.data_pipeline.dataset import DataPipelineDataset
from yolo_agent.components.adapters.data_pipeline.transforms import (
    DataTransformConfig,
    TransformMechanism,
    blend_multi_image_samples,
    copy_paste_sample,
    crop_sample,
    zero_effect_sample,
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
    "DataPipelineDataset",
    "DataSampleRecord",
    "DataTransformConfig",
    "ActiveLearningAcquisitionAdapter",
    "ActiveLearningConfig",
    "ActiveLearningPlugin",
    "AnnotationFilterPlugin",
    "AnnotationQualityFilterAdapter",
    "DistributedExposureSampler",
    "ExposureConfig",
    "ExposureMechanism",
    "FalseNegativeClassBoostAdapter",
    "HardNegativeReplayAdapter",
    "MultiImageSamplingScheduleAdapter",
    "NormalizationPreprocessingAdapter",
    "ObjectCentricCropAdapter",
    "RareClassCopyPasteAdapter",
    "RepeatFactorSamplingAdapter",
    "ScaleAwareCropAdapter",
    "SamplingPlugin",
    "SmallObjectWeightedSamplingAdapter",
    "PreprocessingPlugin",
    "TransformMechanism",
    "TrainHardNegativePrediction",
    "TrainHardNegativePredictionBatch",
    "TrainSampleIndex",
    "TrainSampleIndexRecord",
    "bound_exposure",
    "compute_exposure",
    "compute_exposure_details",
    "produce_train_hard_negative_manifest",
    "train_sample_index_from_yolo_data",
    "train_sample_index_from_records",
    "blend_multi_image_samples",
    "copy_paste_sample",
    "crop_sample",
    "zero_effect_sample",
]
