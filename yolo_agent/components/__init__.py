"""Reusable YOLO component cards and registries."""

from yolo_agent.components.registry import ComponentRegistry, load_cards
from yolo_agent.components.postprocess import (
    PostProcessRecommendation,
    PostProcessRegistry,
    PostProcessStrategy,
)
from yolo_agent.components.schema import (
    Compatibility,
    ComponentCard,
    ComponentType,
    EvidenceRequirement,
    SearchSpace,
)
from yolo_agent.components.contracts import (
    ComponentContract,
    ComponentExecutionError,
    contract_from_card,
    load_contracts,
)
from yolo_agent.components.maturity import (
    ComponentMaturity,
    MaturityTransitionError,
    can_transition,
    transition_maturity,
)
from yolo_agent.components.compatibility import BaseModelSpec, CompatibilityChecker, CompatibilityResult
from yolo_agent.components.yolo26_compatibility import YOLO26CompatibilityChecker, YOLO26CompatibilityResult
from yolo_agent.components.auxiliary_losses import (
    AuxiliaryLossInputs,
    AuxiliaryLossOutput,
    AuxiliaryLossPlugin,
    BPCCalibrationAuxiliaryLoss,
    CorrelationAuxiliaryLoss,
    PseudoIoUQualityAuxiliaryLoss,
    build_auxiliary_loss,
)
from yolo_agent.components.assignment import (
    AssignerInputs,
    AssignerOutput,
    AssignmentComparison,
    DSLAAssignerPlugin,
    NativeYOLO26AssignerPlugin,
    OTAAssignerPlugin,
    TOODTaskAlignedAssignerPlugin,
    YOLO26AssignerPlugin,
    build_yolo26_assigner_plugin,
    compare_assignments,
)

__all__ = [
    "Compatibility",
    "ComponentCard",
    "ComponentRegistry",
    "ComponentType",
    "BaseModelSpec",
    "CompatibilityChecker",
    "CompatibilityResult",
    "YOLO26CompatibilityChecker",
    "YOLO26CompatibilityResult",
    "AuxiliaryLossInputs",
    "AuxiliaryLossOutput",
    "AuxiliaryLossPlugin",
    "BPCCalibrationAuxiliaryLoss",
    "CorrelationAuxiliaryLoss",
    "PseudoIoUQualityAuxiliaryLoss",
    "build_auxiliary_loss",
    "AssignerInputs",
    "AssignerOutput",
    "AssignmentComparison",
    "DSLAAssignerPlugin",
    "NativeYOLO26AssignerPlugin",
    "OTAAssignerPlugin",
    "TOODTaskAlignedAssignerPlugin",
    "YOLO26AssignerPlugin",
    "build_yolo26_assigner_plugin",
    "compare_assignments",
    "EvidenceRequirement",
    "PostProcessRecommendation",
    "PostProcessRegistry",
    "PostProcessStrategy",
    "SearchSpace",
    "load_cards",
    "ComponentContract",
    "ComponentExecutionError",
    "ComponentMaturity",
    "MaturityTransitionError",
    "can_transition",
    "contract_from_card",
    "load_contracts",
    "transition_maturity",
]
