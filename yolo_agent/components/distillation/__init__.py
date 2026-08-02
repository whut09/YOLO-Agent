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
    DistillationInputs,
    DistillationLossOutput,
    DistillationMechanismLoss,
    LocalizationDistillationLoss,
    LogitsDistillationLoss,
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
    "DistillationLossOutput",
    "DistillationMechanismLoss",
    "LocalizationDistillationLoss",
    "LogitsDistillationLoss",
    "build_distillation_mechanism_loss",
]
