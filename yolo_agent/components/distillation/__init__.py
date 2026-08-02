"""Teacher-student distillation primitives."""

from yolo_agent.components.distillation.losses import (
    DistillationWeights,
    YOLO26DistillationLoss,
    channel_agnostic_feature_loss,
    distillation_loss,
)
from yolo_agent.components.distillation.mechanisms import (
    DISTILLATION_COMPONENTS,
    DISTILLATION_MECHANISMS,
    DistillationMechanism,
    DistillationMechanismSpec,
)
from yolo_agent.components.distillation.mechanism_losses import (
    AttentionDistillationLoss,
    DistillationInputs,
    DistillationLossOutput,
    DistillationMechanismLoss,
    FeatureDistillationLoss,
    LocalizationDistillationLoss,
    LogitsDistillationLoss,
    MaskedFeatureDistillationLoss,
    QualityAwareDistillationLoss,
    RelationDistillationLoss,
    TeacherEnsembleDistillationLoss,
    build_distillation_mechanism_loss,
)
from yolo_agent.components.distillation.trainer import DistillationBatch, DistillationTrainerHook, MockDistillationTrainer

__all__ = [
    "DistillationBatch",
    "DistillationTrainerHook",
    "DistillationWeights",
    "MockDistillationTrainer",
    "YOLO26DistillationLoss",
    "channel_agnostic_feature_loss",
    "distillation_loss",
    "DISTILLATION_COMPONENTS",
    "DISTILLATION_MECHANISMS",
    "DistillationMechanism",
    "DistillationMechanismSpec",
    "DistillationInputs",
    "AttentionDistillationLoss",
    "DistillationLossOutput",
    "DistillationMechanismLoss",
    "FeatureDistillationLoss",
    "LocalizationDistillationLoss",
    "LogitsDistillationLoss",
    "MaskedFeatureDistillationLoss",
    "QualityAwareDistillationLoss",
    "RelationDistillationLoss",
    "TeacherEnsembleDistillationLoss",
    "build_distillation_mechanism_loss",
]
