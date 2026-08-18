"""Runtime adapters for explicitly supervised domain adaptation batches."""

from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
    DomainAdaptationBranchAdapter,
    DomainAdaptationBranchPlugin,
)
from yolo_agent.components.adapters.domain_adaptation.branches import (
    DomainAdaptationMethodRegistry,
    default_domain_adaptation_registry,
)
from yolo_agent.components.adapters.domain_adaptation.feature_alignment import (
    DomainFeatureAlignmentAdapter,
    DomainFeatureAlignmentRuntimePlugin,
)

__all__ = [
    "DomainAdaptationBranchAdapter",
    "DomainAdaptationBranchPlugin",
    "DomainAdaptationMethodRegistry",
    "DomainFeatureAlignmentAdapter",
    "DomainFeatureAlignmentRuntimePlugin",
    "default_domain_adaptation_registry",
]
