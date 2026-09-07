"""Distillation component adapters with lazy public exports.

The adapter modules depend on research and recipe registries.  Lazy exports
keep importing a leaf module such as the teacher asset resolver independent
from the order in which those registries are initialized.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "BRANCH_ADAPTERS": "branch_adapters",
    "PAPER_ROUTE_ADAPTERS": "paper_routes",
    "REQUIRED_BRANCH_ADAPTERS": "branch_adapters",
    "DistillationMethodRegistry": "method_registry",
    "DistillationPaperRoute": "paper_routes",
    "DistillationPaperRouteCoverage": "paper_routes",
    "DistillationPaperRouteMissingError": "paper_routes",
    "DistillationPaperRouteRegistry": "paper_routes",
    "DistillationTeacherMissingError": "method_registry",
    "YOLO26DistillationAdapter": "yolo26_distillation",
    "YOLO26DistillationRuntimePlugin": "yolo26_distillation",
    "CheckpointIdentity": "teacher_evidence",
    "CheckpointMetadata": "teacher_evidence",
    "CheckpointResolution": "teacher_evidence",
    "branch_adapter": "branch_adapters",
    "branch_adapter_class_name": "branch_adapters",
    "build_all_paper_route_adapters": "paper_routes",
    "build_paper_route": "paper_routes",
    "build_paper_routes": "paper_routes",
    "create_paper_route_adapter": "paper_routes",
    "default_distillation_method_registry": "method_registry",
    "default_paper_route_registry": "paper_routes",
    "make_branch_adapter": "branch_adapters",
    "paper_route_adapter": "paper_routes",
    "paper_route_coverage": "paper_routes",
    "resolve_checkpoint_identity": "teacher_evidence",
    "resolve_student_checkpoint": "teacher_evidence",
    "resolve_teacher_checkpoint": "teacher_evidence",
    "TeacherAssetRecord": "teacher_asset_resolver",
    "TeacherAssetReport": "teacher_asset_resolver",
    "TeacherAssetResolver": "teacher_asset_resolver",
    "verify_teacher_asset": "teacher_asset_resolver",
    "resolve_teacher_assets": "teacher_asset_resolver",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
