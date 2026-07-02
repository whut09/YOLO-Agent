"""Agent implementations that plan and coordinate experiments."""

from yolo_agent.agents.ablation_planner import AblationPlan, AblationPlanner
from yolo_agent.agents.active_learning import (
    ActiveLearningMiner,
    ActiveLearningPlan,
    LabelingManifest,
    MiningConfig,
    PredictionSummary,
)
from yolo_agent.agents.augmentation_policy import (
    AugmentationPolicyAction,
    AugmentationPolicyEngine,
    AugmentationPolicyResult,
)
from yolo_agent.agents.candidate_generator import CandidateConfig, CandidateGenerator, CandidatePlan
from yolo_agent.agents.error_to_action import (
    ActionPolicy,
    DetectionErrorObservation,
    ErrorActionMapper,
    ErrorActionPlan,
)

__all__ = [
    "AblationPlan",
    "AblationPlanner",
    "ActiveLearningMiner",
    "ActiveLearningPlan",
    "ActionPolicy",
    "AugmentationPolicyAction",
    "AugmentationPolicyEngine",
    "AugmentationPolicyResult",
    "CandidateConfig",
    "CandidateGenerator",
    "CandidatePlan",
    "DetectionErrorObservation",
    "ErrorActionMapper",
    "ErrorActionPlan",
    "LabelingManifest",
    "MiningConfig",
    "PredictionSummary",
]
