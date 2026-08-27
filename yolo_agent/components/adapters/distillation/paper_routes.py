"""Paper-specific runtime routes for the certified distillation papers.

A shared teacher-student runtime cannot certify a paper by itself.  Each
certified distillation paper therefore owns one route with its own adapter
class, changed variables, runtime payload schema, recipe identity, and
execution fingerprint.  The 18 named papers stay branch-bound; the 14 papers
without a certified branch keep an explicit identity-recovery route instead
of being collapsed into a generic adapter.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.distillation.method_registry import (
    BRANCH_TO_MECHANISM,
    CERTIFIED_DISTILLATION_PAPERS,
    DistillationBranchId,
    NAMED_PAPER_BRANCHES,
    build_branch,
)
from yolo_agent.components.distillation.mechanisms import DISTILLATION_MECHANISMS
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.recipes.paper_recipe_bindings import (
    NAMED_METHOD_SLUGS,
    _fallback_slug,
)


MethodIdentityStatus = Literal["branch_bound", "identity_recovery"]


class DistillationPaperRouteMissingError(LookupError):
    """Raised when a paper has no independent distillation route."""


class DistillationPaperRoute(BaseModel, YAMLModelMixin):
    """One paper's independent distillation runtime route."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "distillation_paper_route.v1"
    paper_id: str
    method_profile_id: str
    paper_specific_mechanism_id: str
    method_identity_status: MethodIdentityStatus
    branch_id: DistillationBranchId | None = None
    component_id: str
    branch_component_id: str | None = None
    recipe_id: str
    recipe_version: str = "v1.0.0"
    adapter_class: str
    adapter_version: str
    changed_variables: dict[str, Any]
    runtime_payload_schema: dict[str, str]
    teacher_protocol: dict[str, Any]
    student_protocol: dict[str, Any]
    matched_baseline_required: Literal[True] = True
    student_only_export: Literal[True] = True
    student_only_metrics: tuple[str, ...] = ("latency_ms", "model_size_mb")
    reason_codes: list[str] = Field(default_factory=list)
    execution_fingerprint: str = ""

    @model_validator(mode="after")
    def bind_route_identity(self) -> "DistillationPaperRoute":
        if not self.paper_id.strip():
            raise ValueError("paper route requires a paper id")
        if self.method_identity_status == "branch_bound" and self.branch_id is None:
            raise ValueError(
                f"branch-bound paper route requires a branch: {self.paper_id}"
            )
        if (
            self.method_identity_status == "branch_bound"
            and not self.branch_component_id
        ):
            raise ValueError(
                f"branch-bound paper route requires its branch component: {self.paper_id}"
            )
        if (
            self.method_identity_status == "identity_recovery"
            and self.branch_id is not None
        ):
            raise ValueError(
                f"identity-recovery routes must not declare a branch: {self.paper_id}"
            )
        if (
            self.method_identity_status == "identity_recovery"
            and not self.reason_codes
        ):
            raise ValueError(
                f"identity-recovery routes must record reason codes: {self.paper_id}"
            )
        if self.component_id != self.paper_specific_mechanism_id:
            raise ValueError(
                "paper route component identity must match the paper-specific "
                "mechanism id"
            )
        if not self.paper_specific_mechanism_id.startswith("distillation."):
            raise ValueError(
                f"paper route mechanism must be paper-specific: {self.paper_id}"
            )
        if not self.changed_variables:
            raise ValueError("paper route requires independent changed variables")
        if "distillation.method" not in self.changed_variables:
            raise ValueError(
                "paper route changed variables must name the paper method"
            )
        required_schema_keys = (
            "paper_id",
            "paper_route_fingerprint",
            "recipe_id",
            "teacher",
            "teacher_sha256",
            "teacher_architecture",
            "teacher_split",
            "student",
            "student_sha256",
            "student_architecture",
            "student_split",
            "dataset_hash",
            "matched_baseline",
        )
        for key in required_schema_keys:
            if key not in self.runtime_payload_schema:
                raise ValueError(
                    f"paper route payload schema misses {key}: {self.paper_id}"
                )
        if self.student_protocol.get("architecture") != "yolo26n":
            raise ValueError(
                f"paper routes fix the student to yolo26n: {self.paper_id}"
            )
        if self.student_protocol.get("export_and_measure") != "student_only":
            raise ValueError(
                f"paper routes must measure the student only: {self.paper_id}"
            )
        if not self.teacher_protocol.get("frozen"):
            raise ValueError(f"paper routes require a frozen teacher: {self.paper_id}")
        if not self.teacher_protocol.get("checkpoint_required"):
            raise ValueError(
                f"paper routes require a real teacher checkpoint: {self.paper_id}"
            )
        if self.teacher_protocol.get("mock_forbidden") is not True:
            raise ValueError(
                f"paper routes forbid mock teachers: {self.paper_id}"
            )
        expected = compute_paper_route_fingerprint(self)
        if self.execution_fingerprint and self.execution_fingerprint != expected:
            raise ValueError(f"paper route fingerprint mismatch: {self.paper_id}")
        self.execution_fingerprint = expected
        return self


def compute_paper_route_fingerprint(route: DistillationPaperRoute) -> str:
    """Hash the full paper-route identity so two papers never share it."""
    payload = {
        "paper_id": route.paper_id,
        "method_profile_id": route.method_profile_id,
        "paper_specific_mechanism_id": route.paper_specific_mechanism_id,
        "method_identity_status": route.method_identity_status,
        "branch_id": route.branch_id,
        "component_id": route.component_id,
        "branch_component_id": route.branch_component_id,
        "recipe_id": route.recipe_id,
        "recipe_version": route.recipe_version,
        "adapter_class": route.adapter_class,
        "adapter_version": route.adapter_version,
        "changed_variables": route.changed_variables,
        "runtime_payload_schema": route.runtime_payload_schema,
        "teacher_protocol": route.teacher_protocol,
        "student_protocol": route.student_protocol,
        "student_only_metrics": list(route.student_only_metrics),
        "reason_codes": list(route.reason_codes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def paper_route_method_slug(paper_id: str) -> str:
    """Return the paper method slug used by the recipe bindings."""
    return NAMED_METHOD_SLUGS.get(paper_id) or _fallback_slug(paper_id)


def paper_route_paper_slug(paper_id: str) -> str:
    """Return the per-paper slug used for recipe and profile identity."""
    return _fallback_slug(paper_id)


def paper_route_recipe_id(paper_id: str) -> str:
    return f"paper_{paper_route_paper_slug(paper_id)}"


def paper_route_adapter_class_name(paper_id: str) -> str:
    slug = paper_route_method_slug(paper_id)
    camel = "".join(part[:1].upper() + part[1:] for part in slug.split("_") if part)
    return f"Distillation{camel}Adapter"


def _teacher_protocol() -> dict[str, Any]:
    return {
        "frozen": True,
        "checkpoint_required": True,
        "sha256_required": True,
        "architecture_required": True,
        "dataset_hash_required": True,
        "split_required": True,
        "imgsz": 640,
        "allowed_names": ["yolo26s.pt", "yolo26m.pt"],
        "export_forbidden": True,
        "mock_forbidden": True,
    }


def _student_protocol() -> dict[str, Any]:
    return {
        "checkpoint": "yolo26n.pt",
        "architecture": "yolo26n",
        "imgsz": 640,
        "split_required": True,
        "dataset_hash_required": True,
        "native_dfl_free": True,
        "native_one_to_one_head": True,
        "export_and_measure": "student_only",
        "metrics": ["latency_ms", "model_size_mb"],
    }


def _paper_payload_schema(
    branch: Any | None,
) -> dict[str, str]:
    schema: dict[str, str] = {}
    if branch is not None:
        schema.update(branch.runtime_payload_schema)
    schema.update(
        {
            "paper_id": "str",
            "method_profile_id": "str",
            "paper_specific_mechanism_id": "str",
            "recipe_id": "str",
            "paper_route_fingerprint": "sha256",
            "teacher": "path",
            "teacher_sha256": "sha256",
            "teacher_architecture": "str",
            "teacher_split": "str",
            "teacher_imgsz": "640",
            "student": "path",
            "student_sha256": "sha256",
            "student_architecture": "yolo26n",
            "student_split": "str",
            "student_imgsz": "640",
            "dataset_hash": "sha256",
            "weight": "float",
            "matched_baseline": "object",
            "student_only_export": "true",
            "student_only_metrics": "latency_ms,model_size_mb",
        }
    )
    return schema


def build_paper_route(paper_id: str) -> DistillationPaperRoute:
    """Build the independent runtime route for one certified paper."""
    if not str(paper_id).strip():
        raise ValueError("paper id is required")
    branch_id = NAMED_PAPER_BRANCHES.get(paper_id)
    branch = build_branch(branch_id) if branch_id is not None else None
    branch_component_id = (
        DISTILLATION_MECHANISMS[BRANCH_TO_MECHANISM[branch_id]].component_id
        if branch_id is not None
        else None
    )
    method_slug = paper_route_method_slug(paper_id)
    mechanism_id = f"distillation.{method_slug}"
    return DistillationPaperRoute(
        paper_id=paper_id,
        method_profile_id=f"method-profile-{paper_route_paper_slug(paper_id)}",
        paper_specific_mechanism_id=mechanism_id,
        method_identity_status=(
            "branch_bound" if branch_id is not None else "identity_recovery"
        ),
        branch_id=branch_id,
        component_id=mechanism_id,
        branch_component_id=branch_component_id,
        recipe_id=paper_route_recipe_id(paper_id),
        adapter_class=paper_route_adapter_class_name(paper_id),
        adapter_version=f"paper_route.{mechanism_id}.v1",
        changed_variables={
            "distillation.method": mechanism_id,
            "distillation.loss": method_slug,
        },
        runtime_payload_schema=_paper_payload_schema(branch),
        teacher_protocol=_teacher_protocol(),
        student_protocol=_student_protocol(),
        reason_codes=(
            []
            if branch_id is not None
            else ["distillation_branch_unmapped", "paper_method_identity_missing"]
        ),
    )


def build_paper_routes(
    paper_ids: tuple[str, ...] = CERTIFIED_DISTILLATION_PAPERS,
) -> list[DistillationPaperRoute]:
    return [build_paper_route(paper_id) for paper_id in paper_ids]


class DistillationPaperRouteCoverage(BaseModel, YAMLModelMixin):
    """Coverage proof that every paper owns exactly one route."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "distillation_paper_route_coverage.v1"
    papers_total: int
    branch_bound: int
    identity_recovery: int
    routes: list[DistillationPaperRoute]
    silent_drops: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_silent_drop(self) -> "DistillationPaperRouteCoverage":
        if self.silent_drops:
            raise ValueError(
                f"paper route coverage silent drops: {self.silent_drops}"
            )
        paper_ids = [item.paper_id for item in self.routes]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("paper route coverage contains duplicate papers")
        if self.papers_total != len(paper_ids):
            raise ValueError("every paper must appear in exactly one route")
        if self.branch_bound + self.identity_recovery != self.papers_total:
            raise ValueError(
                "every paper route must be branch-bound or identity-recovery"
            )
        return self


def paper_route_coverage(
    paper_ids: tuple[str, ...] = CERTIFIED_DISTILLATION_PAPERS,
) -> DistillationPaperRouteCoverage:
    routes = build_paper_routes(paper_ids)
    return DistillationPaperRouteCoverage(
        papers_total=len(paper_ids),
        branch_bound=sum(
            item.method_identity_status == "branch_bound" for item in routes
        ),
        identity_recovery=sum(
            item.method_identity_status == "identity_recovery" for item in routes
        ),
        routes=routes,
        silent_drops=[],
    )


class DistillationPaperRouteRegistry:
    """Index of independent paper routes keyed by paper id."""

    def __init__(self, routes: list[DistillationPaperRoute] | None = None) -> None:
        self._routes = {
            item.paper_id: item for item in (routes or build_paper_routes())
        }

    def route(self, paper_id: str) -> DistillationPaperRoute:
        try:
            return self._routes[paper_id]
        except KeyError:
            raise DistillationPaperRouteMissingError(
                f"no paper-specific distillation route for {paper_id}"
            ) from None

    def routes(self) -> list[DistillationPaperRoute]:
        return [
            self._routes[item] for item in CERTIFIED_DISTILLATION_PAPERS if item in self._routes
        ]

    def coverage(
        self,
        paper_ids: tuple[str, ...] = CERTIFIED_DISTILLATION_PAPERS,
    ) -> DistillationPaperRouteCoverage:
        missing = [paper_id for paper_id in paper_ids if paper_id not in self._routes]
        routes = [self.route(paper_id) for paper_id in paper_ids if paper_id in self._routes]
        return DistillationPaperRouteCoverage(
            papers_total=len(paper_ids),
            branch_bound=sum(
                item.method_identity_status == "branch_bound" for item in routes
            ),
            identity_recovery=sum(
                item.method_identity_status == "identity_recovery" for item in routes
            ),
            routes=routes,
            silent_drops=missing,
        )

    def __len__(self) -> int:
        return len(self._routes)

    def __contains__(self, paper_id: str) -> bool:
        return paper_id in self._routes


def default_paper_route_registry() -> DistillationPaperRouteRegistry:
    return DistillationPaperRouteRegistry()


__all__ = [
    "CERTIFIED_DISTILLATION_PAPERS",
    "DistillationPaperRoute",
    "DistillationPaperRouteCoverage",
    "DistillationPaperRouteMissingError",
    "DistillationPaperRouteRegistry",
    "MethodIdentityStatus",
    "build_paper_route",
    "build_paper_routes",
    "compute_paper_route_fingerprint",
    "default_paper_route_registry",
    "paper_route_adapter_class_name",
    "paper_route_coverage",
    "paper_route_method_slug",
    "paper_route_paper_slug",
    "paper_route_recipe_id",
]
