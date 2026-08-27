"""Runtime adapters for explicitly supervised domain adaptation batches."""

from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
    DomainAdaptationBranchAdapter,
    DomainAdaptationBranchPlugin,
)
from yolo_agent.components.adapters.domain_adaptation.branches import (
    DomainAdaptationMethodRegistry,
    default_domain_adaptation_registry,
)
from yolo_agent.components.adapters.domain_adaptation.domain_paper_routes import (
    DOMAIN_PAPER_ROUTE_ADAPTERS,
    DomainPaperRoute,
    DomainPaperRouteCoverage,
    DomainPaperRouteMissingError,
    DomainPaperRouteRegistry,
    build_all_domain_paper_route_adapters,
    build_domain_paper_route,
    build_domain_paper_routes,
    create_domain_paper_route_adapter,
    default_domain_paper_route_registry,
    domain_paper_route_adapter,
    domain_paper_route_coverage,
)
from yolo_agent.components.adapters.domain_adaptation.feature_alignment import (
    DomainFeatureAlignmentAdapter,
    DomainFeatureAlignmentRuntimePlugin,
)

__all__ = [
    "DOMAIN_PAPER_ROUTE_ADAPTERS",
    "DomainAdaptationBranchAdapter",
    "DomainAdaptationBranchPlugin",
    "DomainAdaptationMethodRegistry",
    "DomainFeatureAlignmentAdapter",
    "DomainFeatureAlignmentRuntimePlugin",
    "DomainPaperRoute",
    "DomainPaperRouteCoverage",
    "DomainPaperRouteMissingError",
    "DomainPaperRouteRegistry",
    "build_all_domain_paper_route_adapters",
    "build_domain_paper_route",
    "build_domain_paper_routes",
    "create_domain_paper_route_adapter",
    "default_domain_adaptation_registry",
    "default_domain_paper_route_registry",
    "domain_paper_route_adapter",
    "domain_paper_route_coverage",
]
