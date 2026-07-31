"""Paper-method profiles and conservative paper-to-adapter decisions.

The profile is paper metadata.  The decision is an implementation routing result;
neither object is local metric evidence or an authorization to enqueue training.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.component_aliases import (
    ComponentAliasResolution,
    ComponentAliasResolver,
)
from yolo_agent.research.note_parser import PaperMethodClaim
from yolo_agent.research.schemas import PaperRecord


ImplementationDecisionKind = Literal[
    "reuse_existing_adapter",
    "new_method_profile",
    "new_component_adapter",
    "coupled_recipe",
    "separate_detector_family",
    "insufficient_information",
]


class PaperMethodProfile(BaseModel):
    """Frozen, paper-only description of one paper's method adaptation surface."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_method_profile.v1"
    profile_id: str
    paper_id: str
    paper_applicability: str = "insufficient_information"
    paper_detector_family: str | None = None
    method_name: str = "unknown"
    method_names: list[str] = Field(default_factory=list)
    paper_component_ids: list[str] = Field(default_factory=list)
    canonical_component_ids: list[str] = Field(default_factory=list)
    paper_parameters: dict[str, Any] = Field(default_factory=dict)
    protocol_constraints: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    source_locations: list[str] = Field(default_factory=list)
    exact_reproduction_claim: bool = False
    component_adaptation: bool = True
    evidence_level: Literal["paper_claim", "paper_prior"] = "paper_prior"

    @model_validator(mode="after")
    def validate_profile(self) -> "PaperMethodProfile":
        if not self.paper_id.strip():
            raise ValueError("paper method profile requires paper_id")
        if not self.source_locations:
            raise ValueError("paper method profile requires source_locations")
        if self.exact_reproduction_claim and self.component_adaptation:
            raise ValueError(
                "exact reproduction claims must remain separate from component adaptation"
            )
        return self


class PaperImplementationDecision(BaseModel):
    """One deterministic implementation route for one paper method."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_implementation_decision.v1"
    paper_id: str
    profile_id: str
    decision: ImplementationDecisionKind
    canonical_component_ids: list[str] = Field(default_factory=list)
    reusable_adapter_ids: list[str] = Field(default_factory=list)
    required_adapter_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    unimplemented_reasons: dict[str, list[str]] = Field(default_factory=dict)
    source_locations: list[str] = Field(default_factory=list)
    exact_reproduction_claim: bool = False
    component_adaptation: bool = True
    decision_hash: str = ""

    @model_validator(mode="after")
    def validate_decision(self) -> "PaperImplementationDecision":
        if self.decision == "reuse_existing_adapter" and not self.reusable_adapter_ids:
            raise ValueError("reuse_existing_adapter requires reusable_adapter_ids")
        if self.decision == "new_component_adapter" and not self.required_adapter_ids:
            raise ValueError("new_component_adapter requires required_adapter_ids")
        if self.decision in {"new_method_profile", "insufficient_information"} and not self.reasons:
            raise ValueError("decision requires an auditable reason")
        if self.exact_reproduction_claim and self.component_adaptation:
            raise ValueError(
                "exact reproduction decisions must remain separate from component adaptation"
            )
        return self

    def with_hash(self) -> "PaperImplementationDecision":
        payload = self.model_dump(mode="json", exclude={"decision_hash"})
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self.model_copy(update={"decision_hash": digest})


class PaperMethodCoverageReport(BaseModel, YAMLModelMixin):
    """Paper-to-adapter coverage, including explicit reasons for every gap."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_method_coverage.v1"
    snapshot_hash: str | None = None
    paper_count: int = Field(default=0, ge=0)
    profile_count: int = Field(default=0, ge=0)
    decision_counts: dict[ImplementationDecisionKind, int] = Field(default_factory=dict)
    profiles: list[PaperMethodProfile] = Field(default_factory=list)
    decisions: list[PaperImplementationDecision] = Field(default_factory=list)
    adapter_to_papers: dict[str, list[str]] = Field(default_factory=dict)
    unimplemented_reasons: dict[str, list[str]] = Field(default_factory=dict)


class PaperMethodProfileBuilder:
    """Build profiles and decisions without creating adapters or training recipes."""

    def __init__(self, resolver: ComponentAliasResolver) -> None:
        self.resolver = resolver

    def build(
        self,
        papers: list[PaperRecord],
        *,
        evidence_summaries: dict[str, Any] | None = None,
    ) -> PaperMethodCoverageReport:
        summaries = evidence_summaries or {}
        profiles: list[PaperMethodProfile] = []
        decisions: list[PaperImplementationDecision] = []
        adapter_to_papers: dict[str, set[str]] = {}
        unimplemented: dict[str, set[str]] = {}
        for paper in sorted(papers, key=lambda item: item.paper_id):
            claims = _claims_for(paper, summaries.get(paper.paper_id))
            profile = _profile_for(paper, claims, self.resolver)
            resolution = _resolutions_for(profile, self.resolver)
            decision = _decide(profile, resolution)
            profiles.append(profile)
            decisions.append(decision)
            for adapter_id in decision.reusable_adapter_ids:
                adapter_to_papers.setdefault(adapter_id, set()).add(paper.paper_id)
            for component_id, reasons in decision.unimplemented_reasons.items():
                unimplemented.setdefault(component_id, set()).update(reasons)
        counts = Counter(item.decision for item in decisions)
        return PaperMethodCoverageReport(
            paper_count=len(papers),
            profile_count=len(profiles),
            decision_counts={key: counts.get(key, 0) for key in _DECISIONS},
            profiles=profiles,
            decisions=decisions,
            adapter_to_papers={key: sorted(value) for key, value in sorted(adapter_to_papers.items())},
            unimplemented_reasons={key: sorted(value) for key, value in sorted(unimplemented.items())},
        )


_DECISIONS: tuple[ImplementationDecisionKind, ...] = (
    "reuse_existing_adapter",
    "new_method_profile",
    "new_component_adapter",
    "coupled_recipe",
    "separate_detector_family",
    "insufficient_information",
)


def _claims_for(paper: PaperRecord, summary: Any) -> list[PaperMethodClaim]:
    claims = list(getattr(summary, "method_claims", []) or [])
    if claims:
        return claims
    if paper.claimed_effects:
        return [
            PaperMethodClaim(
                method_name=claim.claimed_effect,
                component_ids=[claim.component_id],
                changed_variables=[],
                limitation="; ".join(claim.limitations) or "unknown",
                source_location=f"paper_record.claimed_effects:{index}",
            )
            for index, claim in enumerate(paper.claimed_effects)
        ]
    return [
        PaperMethodClaim(
            method_name="unknown",
            component_ids=list(paper.component_ids),
            source_location=(
                paper.provenance.source_path
                if paper.provenance is not None
                else "paper_record.component_ids"
            ),
        )
    ]


def _profile_for(
    paper: PaperRecord,
    claims: list[PaperMethodClaim],
    resolver: ComponentAliasResolver,
) -> PaperMethodProfile:
    paper_component_ids = sorted({item for claim in claims for item in claim.component_ids})
    canonical_ids: set[str] = set()
    for component_id in paper_component_ids:
        canonical_ids.update(
            item.canonical_component_id
            for item in resolver.resolve(component_id, source_paper_ids=[paper.paper_id]).mappings
        )
    method_names = sorted({claim.method_name for claim in claims if claim.method_name != "unknown"})
    source_locations = sorted({claim.source_location for claim in claims if claim.source_location})
    if not source_locations:
        source_locations = ["paper_record"]
    changed_variables = sorted({
        variable
        for claim in claims
        for variable in claim.changed_variables
        if variable
    })
    limitations = sorted({
        claim.limitation
        for claim in claims
        if claim.limitation and claim.limitation != "unknown"
    })
    return PaperMethodProfile(
        profile_id=_profile_id(paper.paper_id, method_names, source_locations),
        paper_id=paper.paper_id,
        paper_applicability=paper.applicability,
        paper_detector_family=paper.detector_family,
        method_name=method_names[0] if method_names else "unknown",
        method_names=method_names,
        paper_component_ids=paper_component_ids,
        canonical_component_ids=sorted(canonical_ids),
        paper_parameters={
            "changed_variables": changed_variables,
            "reported_deltas": [claim.reported_delta for claim in claims if claim.reported_delta],
            "training_costs": [claim.training_cost for claim in claims if claim.training_cost != "unknown"],
            "inference_costs": [claim.inference_cost for claim in claims if claim.inference_cost != "unknown"],
        },
        protocol_constraints={
            "baselines": [claim.baseline_description for claim in claims if claim.baseline_description != "unknown"],
            "datasets": sorted({claim.dataset for claim in claims if claim.dataset != "unknown"}),
            "model_families": sorted({claim.model_family for claim in claims if claim.model_family != "unknown"}),
            "insertion_points": sorted({claim.insertion_point for claim in claims if claim.insertion_point != "unknown"}),
        },
        limitations=limitations,
        source_locations=source_locations,
        component_adaptation=bool(
            canonical_ids
            and paper.applicability not in {"separate_detector_family", "incompatible"}
        ),
    )


def _resolutions_for(
    profile: PaperMethodProfile,
    resolver: ComponentAliasResolver,
) -> list[ComponentAliasResolution]:
    return [
        resolver.resolve(component_id, source_paper_ids=[profile.paper_id])
        for component_id in profile.paper_component_ids
    ]


def _decide(
    profile: PaperMethodProfile,
    resolutions: list[ComponentAliasResolution],
) -> PaperImplementationDecision:
    mappings = [mapping for item in resolutions for mapping in item.mappings]
    canonical_ids = sorted({item.canonical_component_id for item in mappings})
    reusable = sorted({
        item.canonical_component_id for item in mappings if item.adapter_verified
    })
    required = sorted(set(canonical_ids) - set(reusable))
    reasons: list[str] = []
    unimplemented: dict[str, list[str]] = {}
    unresolved = sorted(item.paper_component_id for item in resolutions if not item.resolved)
    if profile.paper_applicability in {"separate_detector_family", "incompatible"}:
        decision: ImplementationDecisionKind = "separate_detector_family"
        reasons.append("paper_applicability_routes_method_outside_yolo26")
    elif not profile.paper_component_ids:
        decision = "insufficient_information"
        reasons.append("paper_does_not_identify_a_component_or_method")
    elif unresolved:
        decision = "insufficient_information"
        reasons.append("unresolved_paper_component_alias")
        for item in unresolved:
            unimplemented[item] = ["canonical_component_mapping_required"]
    elif any(item.yolo26_compatibility == "incompatible" for item in mappings):
        decision = "separate_detector_family"
        reasons.append("component_contract_or_taxonomy_rejects_yolo26")
    elif len(canonical_ids) > 1:
        decision = "coupled_recipe"
        reasons.append("method_maps_to_multiple_canonical_components")
    elif reusable:
        decision = "reuse_existing_adapter"
        reasons.append("one_canonical_mechanism_has_a_verified_local_adapter")
    elif _has_method_detail(profile):
        decision = "new_component_adapter"
        reasons.append("canonical_component_is_known_but_no_verified_adapter_exists")
        for component_id in required:
            unimplemented[component_id] = [
                "adapter_not_verified",
                "runtime_and_smoke_artifacts_required",
            ]
    else:
        decision = "new_method_profile"
        reasons.append("paper_method_is_descriptive_but_runtime_parameters_are_incomplete")
        for component_id in required:
            unimplemented[component_id] = ["method_profile_requires_explicit_runtime_contract"]
    result = PaperImplementationDecision(
        paper_id=profile.paper_id,
        profile_id=profile.profile_id,
        decision=decision,
        canonical_component_ids=canonical_ids,
        reusable_adapter_ids=reusable,
        required_adapter_ids=required,
        reasons=reasons,
        unimplemented_reasons=unimplemented,
        source_locations=profile.source_locations,
        exact_reproduction_claim=profile.exact_reproduction_claim,
        component_adaptation=profile.component_adaptation,
    )
    return result.with_hash()


def _has_method_detail(profile: PaperMethodProfile) -> bool:
    parameters = profile.paper_parameters
    protocol = profile.protocol_constraints
    return bool(
        parameters.get("changed_variables")
        or protocol.get("insertion_points")
        or profile.method_name != "unknown"
    )


def _profile_id(paper_id: str, method_names: list[str], locations: list[str]) -> str:
    payload = json.dumps(
        {"paper_id": paper_id, "methods": method_names, "locations": locations},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "method-profile-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


__all__ = [
    "ImplementationDecisionKind",
    "PaperImplementationDecision",
    "PaperMethodCoverageReport",
    "PaperMethodProfile",
    "PaperMethodProfileBuilder",
]
