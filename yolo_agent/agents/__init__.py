"""Agent implementations that plan and coordinate experiments."""

from yolo_agent.agents.ablation_planner import AblationPlan, AblationPlanner
from yolo_agent.agents.active_learning import (
    ActiveLearningMiner,
    ActiveLearningPlan,
    LabelingManifest,
    MiningConfig,
    PredictionSummary,
)
from yolo_agent.agents.annotation_advisor import (
    AnnotationAdviceReport,
    AnnotationAdvisor,
    advise_annotations,
)
from yolo_agent.agents.augmentation_policy import (
    AugmentationPolicyAction,
    AugmentationPolicyEngine,
    AugmentationPolicyResult,
)
from yolo_agent.agents.candidate_generator import CandidateConfig, CandidateGenerator, CandidatePlan
from yolo_agent.agents.component_contribution import (
    AblationMatrix,
    ComponentContributionPlanner,
    ComponentContributionReport,
)
from yolo_agent.agents.error_to_action import (
    ActionPolicy,
    DetectionErrorObservation,
    ErrorActionMapper,
    ErrorActionPlan,
)
from yolo_agent.agents.pareto import CandidateMetrics, ParetoFront, ParetoPoint, ParetoSelector
from yolo_agent.agents.strategy_policy import (
    CandidatePolicy,
    PolicyConstraint,
    PolicyEvaluation,
    PolicyEvaluationReport,
    PolicyEvaluator,
)
from yolo_agent.agents.training_failure import (
    FailureDiagnosis,
    TrainingFailureDiagnoser,
    TrainingFailureReport,
    TrainingRunSignals,
)

__all__ = [
    "AblationPlan",
    "AblationPlanner",
    "ActiveLearningMiner",
    "ActiveLearningPlan",
    "ActionPolicy",
    "AnnotationAdviceReport",
    "AnnotationAdvisor",
    "AugmentationPolicyAction",
    "AugmentationPolicyEngine",
    "AugmentationPolicyResult",
    "CandidateConfig",
    "CandidateGenerator",
    "CandidateMetrics",
    "CandidatePlan",
    "CandidatePolicy",
    "AblationMatrix",
    "ComponentContributionPlanner",
    "ComponentContributionReport",
    "DetectionErrorObservation",
    "ErrorActionMapper",
    "ErrorActionPlan",
    "FailureDiagnosis",
    "LabelingManifest",
    "MiningConfig",
    "ParetoFront",
    "ParetoPoint",
    "ParetoSelector",
    "PolicyConstraint",
    "PolicyEvaluation",
    "PolicyEvaluationReport",
    "PolicyEvaluator",
    "PredictionSummary",
    "TrainingFailureDiagnoser",
    "TrainingFailureReport",
    "TrainingRunSignals",
    "advise_annotations",
]
