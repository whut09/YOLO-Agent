"""Distillation component adapters."""

from yolo_agent.components.adapters.distillation.branch_adapters import (
    BRANCH_ADAPTERS,
    REQUIRED_BRANCH_ADAPTERS,
    branch_adapter,
    branch_adapter_class_name,
    make_branch_adapter,
)
from yolo_agent.components.adapters.distillation.method_registry import (
    DistillationMethodRegistry,
    DistillationTeacherMissingError,
    default_distillation_method_registry,
)
from yolo_agent.components.adapters.distillation.paper_routes import (
    PAPER_ROUTE_ADAPTERS,
    DistillationPaperRoute,
    DistillationPaperRouteCoverage,
    DistillationPaperRouteMissingError,
    DistillationPaperRouteRegistry,
    build_all_paper_route_adapters,
    build_paper_route,
    build_paper_routes,
    create_paper_route_adapter,
    default_paper_route_registry,
    paper_route_adapter,
    paper_route_coverage,
)
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    YOLO26DistillationAdapter,
    YOLO26DistillationRuntimePlugin,
)
from yolo_agent.components.adapters.distillation.teacher_evidence import (
    CheckpointIdentity,
    CheckpointMetadata,
    CheckpointResolution,
    resolve_checkpoint_identity,
    resolve_student_checkpoint,
    resolve_teacher_checkpoint,
)

__all__ = [
    "BRANCH_ADAPTERS",
    "PAPER_ROUTE_ADAPTERS",
    "REQUIRED_BRANCH_ADAPTERS",
    "DistillationMethodRegistry",
    "DistillationPaperRoute",
    "DistillationPaperRouteCoverage",
    "DistillationPaperRouteMissingError",
    "DistillationPaperRouteRegistry",
    "DistillationTeacherMissingError",
    "YOLO26DistillationAdapter",
    "YOLO26DistillationRuntimePlugin",
    "CheckpointIdentity",
    "CheckpointMetadata",
    "CheckpointResolution",
    "branch_adapter",
    "branch_adapter_class_name",
    "build_all_paper_route_adapters",
    "build_paper_route",
    "build_paper_routes",
    "create_paper_route_adapter",
    "default_distillation_method_registry",
    "default_paper_route_registry",
    "make_branch_adapter",
    "paper_route_adapter",
    "paper_route_coverage",
    "resolve_checkpoint_identity",
    "resolve_student_checkpoint",
    "resolve_teacher_checkpoint",
]
