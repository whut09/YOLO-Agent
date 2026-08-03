"""Runtime adapters for explicitly supervised domain adaptation batches."""

from yolo_agent.components.adapters.domain_adaptation.feature_alignment import (
    DomainFeatureAlignmentAdapter,
    DomainFeatureAlignmentRuntimePlugin,
)

__all__ = [
    "DomainFeatureAlignmentAdapter",
    "DomainFeatureAlignmentRuntimePlugin",
]
