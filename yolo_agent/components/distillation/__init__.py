"""Teacher-student distillation primitives."""

from yolo_agent.components.distillation.losses import DistillationWeights, YOLO26DistillationLoss, distillation_loss
from yolo_agent.components.distillation.trainer import DistillationBatch, DistillationTrainerHook, MockDistillationTrainer

__all__ = ["DistillationBatch", "DistillationTrainerHook", "DistillationWeights", "MockDistillationTrainer", "YOLO26DistillationLoss", "distillation_loss"]
