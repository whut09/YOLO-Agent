"""Agent implementations that plan and coordinate experiments."""

from yolo_agent.agents.ablation_planner import AblationPlan, AblationPlanner
from yolo_agent.agents.active_learning import (
    ActiveLearningMiner,
    ActiveLearningPlan,
    DatasetPromotionPlan,
    LabelHandoffResult,
    LabelingManifest,
    MiningConfig,
    PredictionSummary,
    load_prediction_summaries,
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
from yolo_agent.agents.budget_optimizer import (
    BudgetArm,
    BudgetArmSelection,
    BudgetOptimizationReport,
    BudgetOptimizer,
    BudgetOptimizerConfig,
)
from yolo_agent.agents.candidate_generator import CandidateConfig, CandidateGenerator, CandidatePlan
from yolo_agent.agents.component_contribution import (
    AblationMatrix,
    ComponentContributionPlanner,
    ComponentContributionReport,
)
from yolo_agent.agents.decision_bundle import DecisionContext, LLMDecisionBundle
from yolo_agent.agents.error_to_action import (
    ActionPolicy,
    DetectionErrorObservation,
    ErrorActionMapper,
    ErrorActionPlan,
)
from yolo_agent.agents.error_driven_loop import (
    ClosedLoopDiagnosis,
    ErrorDrivenLoopEngine,
    ErrorDrivenLoopReport,
    NextRoundPlan,
)
from yolo_agent.agents.optimization_recipe import (
    OptimizationRecipeEngine,
    OptimizationRecipePlan,
    OptimizationRecipeRecommendation,
    RecipeComponents,
)
from yolo_agent.agents.pareto import CandidateMetrics, ParetoFront, ParetoPoint, ParetoSelector
from yolo_agent.agents.sampling_policy import (
    SamplingAction,
    SamplingPolicyEngine,
    SamplingPolicyPlan,
)
from yolo_agent.agents.strategy_policy import (
    ActionDomain,
    CandidatePolicy,
    ExecutionAction,
    PolicyConstraint,
    PolicyEvaluation,
    PolicyEvaluationReport,
    PolicyEvaluator,
)
from yolo_agent.agents.successive_halving import (
    HalvingAssignment,
    HalvingCandidate,
    HalvingStage,
    SuccessiveHalvingPlan,
    SuccessiveHalvingPlanner,
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
    "ActionDomain",
    "ExecutionAction",
    "AnnotationAdviceReport",
    "AnnotationAdvisor",
    "AugmentationPolicyAction",
    "AugmentationPolicyEngine",
    "AugmentationPolicyResult",
    "BudgetArm",
    "BudgetArmSelection",
    "BudgetOptimizationReport",
    "BudgetOptimizer",
    "BudgetOptimizerConfig",
    "CandidateConfig",
    "CandidateGenerator",
    "CandidateMetrics",
    "CandidatePlan",
    "CandidatePolicy",
    "AblationMatrix",
    "ComponentContributionPlanner",
    "ComponentContributionReport",
    "DecisionContext",
    "ClosedLoopDiagnosis",
    "DetectionErrorObservation",
    "DatasetPromotionPlan",
    "ErrorActionMapper",
    "ErrorActionPlan",
    "ErrorDrivenLoopEngine",
    "ErrorDrivenLoopReport",
    "FailureDiagnosis",
    "LabelHandoffResult",
    "LabelingManifest",
    "LLMDecisionBundle",
    "MiningConfig",
    "NextRoundPlan",
    "OptimizationRecipeEngine",
    "OptimizationRecipePlan",
    "OptimizationRecipeRecommendation",
    "ParetoFront",
    "ParetoPoint",
    "ParetoSelector",
    "PolicyConstraint",
    "PolicyEvaluation",
    "PolicyEvaluationReport",
    "PolicyEvaluator",
    "PredictionSummary",
    "RecipeComponents",
    "SamplingAction",
    "SamplingPolicyEngine",
    "SamplingPolicyPlan",
    "HalvingAssignment",
    "HalvingCandidate",
    "HalvingStage",
    "SuccessiveHalvingPlan",
    "SuccessiveHalvingPlanner",
    "TrainingFailureDiagnoser",
    "TrainingFailureReport",
    "TrainingRunSignals",
    "advise_annotations",
    "load_prediction_summaries",
]
