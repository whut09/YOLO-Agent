"""Paper-specific source-target protocol routes for domain adaptation.

A shared source-target batch cannot certify a paper by itself.  Each of the
40 certified domain-adaptation papers therefore owns one route with its own
adapter class, changed variables, runtime payload schema, recipe identity,
protocol hash, and execution fingerprint.  Every route stays bound to one of
the eight canonical domain branches while keeping a paper-specific component
identity, so papers on the same branch never collapse into one generic
adapter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.domain_adaptation.branches import (
    BASE_PAYLOAD_SCHEMA,
    CANONICAL_DOMAIN_BRANCHES,
    DOMAIN_BRANCH_PROFILES,
    NAMED_PAPER_BRANCHES,
    DomainAdaptationBranchSpec,
    DomainProtocolError,
    build_branch,
    default_domain_adaptation_registry,
)
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.recipes.paper_recipe_bindings import (
    NAMED_METHOD_SLUGS,
    _fallback_slug,
)


DomainBranchId = Literal[
    "adversarial_alignment",
    "feature_alignment",
    "pseudo_label_adaptation",
    "domain_distillation",
    "source_free_adaptation",
    "cross_domain_teacher",
    "contrastive_domain_alignment",
    "active_domain_adaptation",
]


class DomainPaperRoute(BaseModel, YAMLModelMixin):
    """One paper's independent domain-adaptation source-target route."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "domain_paper_route.v1"
    paper_id: str
    method_profile_id: str
    paper_specific_mechanism_id: str
    branch_id: DomainBranchId
    component_id: str
    branch_component_id: str
    recipe_id: str
    recipe_version: str = "v1.0.0"
    adapter_class: str
    adapter_version: str
    adapter_hash: str
    branch_changed_variable: str
    changed_variables: dict[str, Any]
    runtime_strategy: str
    evidence_artifact: str
    adaptation_mode: Literal["unsupervised", "semi_supervised", "source_free", "active"]
    required_label_availability: str
    source_free: bool
    requires_source_domain: bool
    source_protocol: dict[str, Any]
    target_protocol: dict[str, Any]
    branch_payload_schema: dict[str, str]
    route_payload_schema: dict[str, str]
    protocol_hash: str
    reason_codes: list[str] = Field(default_factory=list)
    execution_fingerprint: str = ""

    @model_validator(mode="after")
    def bind_route_identity(self) -> "DomainPaperRoute":
        if not self.paper_id.strip():
            raise DomainProtocolError("domain paper route requires a paper id")
        if self.branch_id not in CANONICAL_DOMAIN_BRANCHES:
            raise DomainProtocolError(
                f"domain paper route branch must be canonical: {self.paper_id}"
            )
        if self.component_id != self.paper_specific_mechanism_id:
            raise DomainProtocolError(
                "domain paper route component identity must match the "
                "paper-specific mechanism id"
            )
        if not self.paper_specific_mechanism_id.startswith("domain_adaptation."):
            raise DomainProtocolError(
                f"domain paper route mechanism must be paper-specific: {self.paper_id}"
            )
        if self.branch_component_id != f"domain_adaptation.{self.branch_id}":
            raise DomainProtocolError(
                f"domain paper route branch component mismatch: {self.paper_id}"
            )
        if self.branch_component_id == self.component_id:
            raise DomainProtocolError(
                "domain paper route must not reuse the branch component: "
                + self.paper_id
            )
        if not self.changed_variables:
            raise DomainProtocolError(
                "domain paper route requires independent changed variables"
            )
        if "domain_adaptation.method" not in self.changed_variables:
            raise DomainProtocolError(
                "domain paper route changed variables must name the paper method"
            )
        required_schema_keys = (
            "paper_id",
            "paper_route_fingerprint",
            "recipe_id",
            "source_manifest",
            "source_manifest_sha256",
            "source_dataset_hash",
            "source_split",
            "target_manifest",
            "target_manifest_sha256",
            "target_dataset_hash",
            "target_split",
            "domain_pair_id",
            "label_availability",
            "adaptation_mode",
            "paper_changed_variable",
            "domain_protocol_hash",
            "evidence_artifact",
            "adapter_hash",
            "protocol_hash",
            "execution_fingerprint",
        )
        for key in required_schema_keys:
            if key not in self.route_payload_schema:
                raise DomainProtocolError(
                    f"domain paper route payload schema misses {key}: {self.paper_id}"
                )
        if self.source_free and self.requires_source_domain:
            raise DomainProtocolError(
                f"source-free routes must not require source data: {self.paper_id}"
            )
        if not self.source_free and not self.requires_source_domain:
            raise DomainProtocolError(
                f"non source-free routes require the source domain: {self.paper_id}"
            )
        if self.source_protocol.get("coco_supervised_forbidden") is not True:
            raise DomainProtocolError(
                f"domain paper routes forbid COCO as a domain: {self.paper_id}"
            )
        if self.target_protocol.get("coco_supervised_forbidden") is not True:
            raise DomainProtocolError(
                f"domain paper routes forbid COCO as a target: {self.paper_id}"
            )
        if self.target_protocol.get("mock_forbidden") is not True:
            raise DomainProtocolError(
                f"domain paper routes forbid mock target data: {self.paper_id}"
            )
        if self.source_free and self.source_protocol.get("manifest_required"):
            raise DomainProtocolError(
                f"source-free routes must not require source data: {self.paper_id}"
            )
        if not self.source_free and not self.source_protocol.get("manifest_required"):
            raise DomainProtocolError(
                f"source-bound routes require the source manifest: {self.paper_id}"
            )
        if self.source_protocol.get("imgsz") != 640:
            raise DomainProtocolError(f"domain paper routes fix imgsz=640: {self.paper_id}")
        if self.target_protocol.get("imgsz") != 640:
            raise DomainProtocolError(f"domain paper routes fix imgsz=640: {self.paper_id}")
        if not self.adapter_class.startswith("DomainAdaptation"):
            raise DomainProtocolError(
                f"domain paper route adapter class must be independent: {self.paper_id}"
            )
        if self.adapter_class == "DomainAdaptationBranchAdapter":
            raise DomainProtocolError(
                f"domain paper routes must not reuse the generic adapter: {self.paper_id}"
            )
        if not self.adapter_hash or len(self.adapter_hash) != 64:
            raise DomainProtocolError(
                f"domain paper route requires a sha256 adapter hash: {self.paper_id}"
            )
        expected_protocol = compute_domain_route_protocol_hash(self)
        if self.protocol_hash and self.protocol_hash != expected_protocol:
            raise DomainProtocolError(
                f"domain paper route protocol hash mismatch: {self.paper_id}"
            )
        self.protocol_hash = expected_protocol
        expected = compute_domain_route_fingerprint(self)
        if self.execution_fingerprint and self.execution_fingerprint != expected:
            raise DomainProtocolError(
                f"domain paper route fingerprint mismatch: {self.paper_id}"
            )
        self.execution_fingerprint = expected
        return self


def compute_domain_route_protocol_hash(route: DomainPaperRoute) -> str:
    """Hash the paper-level source-target protocol identity."""
    payload = {
        "paper_id": route.paper_id,
        "method_profile_id": route.method_profile_id,
        "paper_specific_mechanism_id": route.paper_specific_mechanism_id,
        "branch_id": route.branch_id,
        "component_id": route.component_id,
        "branch_component_id": route.branch_component_id,
        "recipe_id": route.recipe_id,
        "changed_variables": route.changed_variables,
        "branch_changed_variable": route.branch_changed_variable,
        "runtime_strategy": route.runtime_strategy,
        "evidence_artifact": route.evidence_artifact,
        "adaptation_mode": route.adaptation_mode,
        "required_label_availability": route.required_label_availability,
        "source_free": route.source_free,
        "requires_source_domain": route.requires_source_domain,
        "source_protocol": route.source_protocol,
        "target_protocol": route.target_protocol,
        "route_payload_schema": route.route_payload_schema,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_domain_route_fingerprint(route: DomainPaperRoute) -> str:
    """Hash the full paper-route identity so two papers never share it."""
    payload = {
        "protocol_hash": compute_domain_route_protocol_hash(route),
        "adapter_class": route.adapter_class,
        "adapter_version": route.adapter_version,
        "adapter_hash": route.adapter_hash,
        "recipe_version": route.recipe_version,
        "branch_payload_schema": route.branch_payload_schema,
        "reason_codes": list(route.reason_codes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _branch_runtime_sha256() -> str:
    path = Path(__file__).with_name("branch_runtime.py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_domain_route_adapter_hash(
    *,
    adapter_class: str,
    branch_id: str,
) -> str:
    """Bind the executing code (branch runtime) to one paper's adapter class."""
    payload = f"{adapter_class}\n{_branch_runtime_sha256()}\n{branch_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def domain_route_method_slug(paper_id: str) -> str:
    """Return the paper method slug used by the recipe bindings."""
    return NAMED_METHOD_SLUGS.get(paper_id) or _fallback_slug(paper_id)


def domain_route_paper_slug(paper_id: str) -> str:
    """Return the per-paper slug used for recipe and profile identity."""
    return _fallback_slug(paper_id)


def domain_route_recipe_id(paper_id: str) -> str:
    return f"paper_{domain_route_paper_slug(paper_id)}"


def domain_route_mechanism_id(paper_id: str) -> str:
    return f"domain_adaptation.{domain_route_method_slug(paper_id)}"


def domain_route_adapter_class_name(paper_id: str) -> str:
    slug = domain_route_method_slug(paper_id)
    camel = "".join(part[:1].upper() + part[1:] for part in slug.split("_") if part)
    return f"DomainAdaptation{camel}Adapter"


def _source_protocol(source_free: bool) -> dict[str, Any]:
    return {
        "manifest_required": not source_free,
        "sha256_required": True,
        "dataset_hash_required": True,
        "split_required": True,
        "label_availability": "labeled",
        "coco_supervised_forbidden": True,
        "mock_forbidden": True,
        "imgsz": 640,
    }


def _target_protocol() -> dict[str, Any]:
    return {
        "manifest_required": True,
        "sha256_required": True,
        "dataset_hash_required": True,
        "split_required": True,
        "coco_supervised_forbidden": True,
        "mock_forbidden": True,
        "imgsz": 640,
    }


def _route_payload_schema(branch: DomainAdaptationBranchSpec) -> dict[str, str]:
    schema: dict[str, str] = dict(branch.payload_schema)
    schema.update(
        {
            "paper_id": "str",
            "paper_route_fingerprint": "sha256",
            "recipe_id": "str",
            "source_label_availability": "str",
            "target_label_availability": "str",
            "label_availability": "str",
            "paper_changed_variable": "str",
            "domain_protocol_hash": "sha256",
            "evidence_artifact": "str",
            "adapter_hash": "sha256",
            "protocol_hash": "sha256",
            "execution_fingerprint": "sha256",
        }
    )
    return schema


def build_domain_paper_route(paper_id: str) -> DomainPaperRoute:
    """Build the independent source-target route for one certified paper."""
    if paper_id not in NAMED_PAPER_BRANCHES:
        raise DomainProtocolError(f"paper has no certified domain branch: {paper_id}")
    branch_id = NAMED_PAPER_BRANCHES[paper_id]
    branch = build_branch(branch_id)  # type: ignore[arg-type]
    profile = DOMAIN_BRANCH_PROFILES[branch_id]
    mechanism_id = domain_route_mechanism_id(paper_id)
    adapter_class = domain_route_adapter_class_name(paper_id)
    source_free = branch_id == "source_free_adaptation"
    route_protocol = {
        "paper_id": paper_id,
        "method_profile_id": f"method-profile-{domain_route_paper_slug(paper_id)}",
        "paper_specific_mechanism_id": mechanism_id,
        "branch_id": branch_id,
        "component_id": mechanism_id,
        "branch_component_id": branch.component_id,
        "recipe_id": domain_route_recipe_id(paper_id),
    }
    adapter_hash = compute_domain_route_adapter_hash(
        adapter_class=adapter_class,
        branch_id=branch_id,
    )
    route = DomainPaperRoute(
        **route_protocol,
        recipe_version="v1.0.0",
        adapter_class=adapter_class,
        adapter_version=f"domain_paper_route.{mechanism_id}.v1",
        adapter_hash=adapter_hash,
        branch_changed_variable=branch.changed_variable,
        changed_variables={
            "domain_adaptation.method": mechanism_id,
            "domain_alignment": domain_route_method_slug(paper_id),
        },
        runtime_strategy=branch.runtime_strategy,
        evidence_artifact=branch.evidence_artifact,
        adaptation_mode=branch.adaptation_mode,  # type: ignore[arg-type]
        required_label_availability=branch.required_label_availability,
        source_free=source_free,
        requires_source_domain=bool(profile["requires_source_domain"]),
        source_protocol=_source_protocol(source_free),
        target_protocol=_target_protocol(),
        branch_payload_schema=dict(branch.payload_schema),
        route_payload_schema=_route_payload_schema(branch),
        protocol_hash="",
        reason_codes=[],
        execution_fingerprint="",
    )
    return route


def build_domain_paper_routes(
    paper_ids: tuple[str, ...] | None = None,
) -> list[DomainPaperRoute]:
    """Build every certified paper route without silently dropping any."""
    ids = tuple(paper_ids or sorted(NAMED_PAPER_BRANCHES))
    routes = [build_domain_paper_route(paper_id) for paper_id in ids]
    seen: set[str] = set()
    for route in routes:
        if route.paper_id in seen:
            raise DomainProtocolError(f"duplicate domain paper route: {route.paper_id}")
        seen.add(route.paper_id)
    missing = [paper_id for paper_id in sorted(NAMED_PAPER_BRANCHES) if paper_id not in seen]
    if missing:
        raise DomainProtocolError(
            "domain paper routes dropped papers: " + ", ".join(missing)
        )
    return routes


__all__ = [
    "BASE_PAYLOAD_SCHEMA",
    "DomainBranchId",
    "DomainPaperRoute",
    "build_domain_paper_route",
    "build_domain_paper_routes",
    "compute_domain_route_adapter_hash",
    "compute_domain_route_fingerprint",
    "compute_domain_route_protocol_hash",
    "default_domain_adaptation_registry",
    "domain_route_adapter_class_name",
    "domain_route_mechanism_id",
    "domain_route_method_slug",
    "domain_route_paper_slug",
    "domain_route_recipe_id",
]
