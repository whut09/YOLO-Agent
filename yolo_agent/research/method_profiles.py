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
from yolo_agent.research.code_metadata import (
    OfficialCodeMetadata,
    parse_official_code_metadata,
)
from yolo_agent.research.component_aliases import (
    ComponentAliasResolution,
    ComponentAliasResolver,
)
from yolo_agent.research.mechanism_evidence import (
    MechanismEvidenceExtractor,
    PaperMechanismEvidence,
)
from yolo_agent.research.mechanism_priority import MechanismPriorityConfig
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
AdaptationGapSeverity = Literal["blocking", "non_blocking"]
PaperAdaptationMode = Literal[
    "exact_reproduction",
    "component_adaptation",
    "separate_detector_family",
    "insufficient_information",
]
MechanismMappingSource = Literal[
    "catalog_component_id",
    "summary",
    "note",
    "harness_hint",
    "official_code_metadata",
]


class PaperAdaptationGap(BaseModel):
    """One field-level reason that limits a paper method adaptation."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    reason_code: str
    severity: AdaptationGapSeverity
    observed_value: Any | None = None
    paper_component_id: str | None = None
    source_locations: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)


class PaperEvidenceInventory(BaseModel):
    """Availability of local, offline paper metadata used for profiling."""

    model_config = ConfigDict(extra="forbid")

    summary_available: bool = False
    summary_source: str = "unknown"
    note_available: bool = False
    note_path: str | None = None
    harness_hint_count: int = Field(default=0, ge=0)
    official_code_available: bool = False
    code_license_known: bool = False
    framework_known: bool = False
    source_locations: list[str] = Field(default_factory=list)


class PaperMechanismMapping(BaseModel):
    """Auditable paper -> profile -> mechanism -> adapter chain."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    profile_id: str
    source_term: str
    source: MechanismMappingSource
    source_location: str
    canonical_component_id: str
    alias_match_type: str
    yolo26_compatibility: str
    implementation_status: str
    reusable_adapter_id: str | None = None
    adapter_verified: bool = False
    runtime_execution_ready: bool = False


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
    adaptation_mode: PaperAdaptationMode = "component_adaptation"
    evidence_level: Literal["paper_claim", "paper_prior"] = "paper_prior"
    evidence_inventory: PaperEvidenceInventory = Field(
        default_factory=PaperEvidenceInventory
    )
    official_code_metadata: OfficialCodeMetadata = Field(
        default_factory=OfficialCodeMetadata
    )
    mechanism_evidence: list[PaperMechanismEvidence] = Field(default_factory=list)

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
        if self.adaptation_mode == "exact_reproduction" and not (
            self.exact_reproduction_claim and not self.component_adaptation
        ):
            raise ValueError("exact_reproduction mode requires an explicit exclusive claim")
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
    adaptation_gaps: list[PaperAdaptationGap] = Field(default_factory=list)
    mechanism_mappings: list[PaperMechanismMapping] = Field(default_factory=list)
    source_locations: list[str] = Field(default_factory=list)
    exact_reproduction_claim: bool = False
    component_adaptation: bool = True
    adaptation_mode: PaperAdaptationMode = "component_adaptation"
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
        expected_mode = _adaptation_mode(
            self.decision,
            exact_reproduction=self.exact_reproduction_claim,
        )
        if self.adaptation_mode != expected_mode:
            raise ValueError(
                f"adaptation_mode {self.adaptation_mode!r} does not match "
                f"decision {self.decision!r}"
            )
        return self

    def with_hash(self) -> "PaperImplementationDecision":
        payload = self.model_dump(mode="json", exclude={"decision_hash"})
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self.model_copy(update={"decision_hash": digest})


class CanonicalMechanismCoverage(BaseModel):
    """Coverage state for one unique canonical mechanism, not one paper."""

    model_config = ConfigDict(extra="forbid")

    canonical_component_id: str
    paper_ids: list[str] = Field(default_factory=list)
    reference_count: int = Field(default=0, ge=0)
    yolo26_compatibility: str = "unknown"
    implementation_status: str = "metadata_only"
    reusable_adapter: bool = False
    runtime_execution_ready: bool = False
    priority_family: str = "other"
    priority_rank: int = Field(default=1000, ge=1)


class CompatibleMechanismCoverage(BaseModel):
    """Mechanism-level coverage with explicit, non-paper denominators."""

    model_config = ConfigDict(extra="forbid")

    referenced_mechanism_count: int = Field(default=0, ge=0)
    compatible_mechanism_count: int = Field(default=0, ge=0)
    potentially_adaptable_mechanism_count: int = Field(default=0, ge=0)
    reusable_adapter_mechanism_count: int = Field(default=0, ge=0)
    runtime_ready_mechanism_count: int = Field(default=0, ge=0)
    compatible_adapter_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    runtime_ready_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    priority_family_mechanism_counts: dict[str, int] = Field(default_factory=dict)
    mechanisms: list[CanonicalMechanismCoverage] = Field(default_factory=list)


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
    compatible_mechanism_coverage: CompatibleMechanismCoverage = Field(
        default_factory=CompatibleMechanismCoverage
    )


class PaperMethodProfileBuilder:
    """Build profiles and decisions without creating adapters or training recipes."""

    def __init__(self, resolver: ComponentAliasResolver) -> None:
        self.resolver = resolver
        self.mechanism_priorities = MechanismPriorityConfig.from_yaml()

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
            evidence_summary = summaries.get(paper.paper_id)
            claims = _claims_for(paper, evidence_summary)
            mechanism_evidence = MechanismEvidenceExtractor(
                self.resolver
            ).extract(paper, evidence_summary=evidence_summary)
            profile = _profile_for(
                paper,
                claims,
                self.resolver,
                evidence_summary=evidence_summary,
                mechanism_evidence=mechanism_evidence,
            )
            resolution = _resolutions_for(profile, self.resolver)
            mechanism_mappings = _mechanism_mapping_chain(
                profile,
                resolution,
                self.resolver,
            )
            decision = _decide(
                profile,
                resolution,
                mechanism_mappings=mechanism_mappings,
                priorities=self.mechanism_priorities,
            )
            profile = profile.model_copy(update={
                "adaptation_mode": decision.adaptation_mode,
                "component_adaptation": (
                    decision.adaptation_mode == "component_adaptation"
                ),
            })
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
            compatible_mechanism_coverage=_mechanism_coverage(
                decisions,
                priorities=self.mechanism_priorities,
            ),
        )


_DECISIONS: tuple[ImplementationDecisionKind, ...] = (
    "reuse_existing_adapter",
    "new_method_profile",
    "new_component_adapter",
    "coupled_recipe",
    "separate_detector_family",
    "insufficient_information",
)

_IMPLEMENTATION_STATUS_ORDER = {
    name: index
    for index, name in enumerate((
        "metadata_only",
        "recipe_idea_only",
        "adapter_required",
        "adapter_implemented",
        "runtime_integrated",
        "unit_tested",
        "smoke_passed",
        "gpu_certified",
        "pilot_reproduced",
        "full_reproduced",
        "confirmed_multi_seed",
    ))
}


def _mechanism_coverage(
    decisions: list[PaperImplementationDecision],
    *,
    priorities: MechanismPriorityConfig,
) -> CompatibleMechanismCoverage:
    grouped: dict[str, list[PaperMechanismMapping]] = {}
    for decision in decisions:
        for mapping in decision.mechanism_mappings:
            grouped.setdefault(mapping.canonical_component_id, []).append(mapping)
    mechanisms: list[CanonicalMechanismCoverage] = []
    for component_id, mappings in sorted(grouped.items()):
        paper_ids = sorted({item.paper_id for item in mappings})
        compatibilities = {item.yolo26_compatibility for item in mappings}
        compatibility = next(
            (
                value
                for value in ("compatible", "adapter_required", "incompatible", "unknown")
                if value in compatibilities
            ),
            "unknown",
        )
        status = max(
            (item.implementation_status for item in mappings),
            key=lambda value: _IMPLEMENTATION_STATUS_ORDER.get(value, -1),
        )
        priority = priorities.priority_for(component_id)
        mechanisms.append(CanonicalMechanismCoverage(
            canonical_component_id=component_id,
            paper_ids=paper_ids,
            reference_count=len(paper_ids),
            yolo26_compatibility=compatibility,
            implementation_status=status,
            reusable_adapter=any(item.adapter_verified for item in mappings),
            runtime_execution_ready=any(
                item.runtime_execution_ready for item in mappings
            ),
            priority_family=priority.family_id if priority else "other",
            priority_rank=priority.priority_rank if priority else 1000,
        ))
    mechanisms.sort(
        key=lambda item: (item.priority_rank, item.canonical_component_id)
    )
    compatible = [
        item for item in mechanisms if item.yolo26_compatibility == "compatible"
    ]
    adaptable = [
        item
        for item in mechanisms
        if item.yolo26_compatibility in {"compatible", "adapter_required"}
    ]
    reusable = [item for item in adaptable if item.reusable_adapter]
    runtime_ready = [item for item in adaptable if item.runtime_execution_ready]
    denominator = len(adaptable)
    return CompatibleMechanismCoverage(
        referenced_mechanism_count=len(mechanisms),
        compatible_mechanism_count=len(compatible),
        potentially_adaptable_mechanism_count=denominator,
        reusable_adapter_mechanism_count=len(reusable),
        runtime_ready_mechanism_count=len(runtime_ready),
        compatible_adapter_coverage_ratio=(len(reusable) / denominator if denominator else 0.0),
        runtime_ready_coverage_ratio=(
            len(runtime_ready) / denominator if denominator else 0.0
        ),
        priority_family_mechanism_counts=dict(sorted(Counter(
            item.priority_family for item in mechanisms if item.priority_family != "other"
        ).items())),
        mechanisms=mechanisms,
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
    *,
    evidence_summary: Any | None = None,
    mechanism_evidence: list[PaperMechanismEvidence] | None = None,
) -> PaperMethodProfile:
    explicit_mechanisms = mechanism_evidence or []
    paper_component_ids = sorted({item for claim in claims for item in claim.component_ids})
    canonical_ids: set[str] = set()
    for component_id in paper_component_ids:
        canonical_ids.update(
            item.canonical_component_id
            for item in resolver.resolve(component_id, source_paper_ids=[paper.paper_id]).mappings
        )
    canonical_ids.update(
        item.canonical_component_id for item in explicit_mechanisms
    )
    method_names = sorted({claim.method_name for claim in claims if claim.method_name != "unknown"})
    source_locations = sorted({claim.source_location for claim in claims if claim.source_location})
    if not source_locations:
        source_locations = ["paper_record"]
    source_locations = sorted({
        *source_locations,
        *(item.source_location for item in explicit_mechanisms),
    })
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
        adaptation_mode=(
            "separate_detector_family"
            if paper.applicability in {"separate_detector_family", "incompatible"}
            else "component_adaptation"
            if canonical_ids
            else "insufficient_information"
        ),
        evidence_inventory=_evidence_inventory(paper, evidence_summary),
        official_code_metadata=parse_official_code_metadata(paper),
        mechanism_evidence=explicit_mechanisms,
    )


def _evidence_inventory(
    paper: PaperRecord,
    evidence_summary: Any | None,
) -> PaperEvidenceInventory:
    parsed_locations = sorted(
        set(getattr(evidence_summary, "source_locations", []) or [])
    )
    provenance = paper.provenance
    code = parse_official_code_metadata(paper)
    locations = set(parsed_locations)
    locations.update(code.source_locations)
    return PaperEvidenceInventory(
        summary_available=bool(paper.abstract),
        summary_source=(
            provenance.abstract_source if provenance is not None else "unknown"
        ),
        note_available="note" in parsed_locations,
        note_path=(
            provenance.original_note_path if provenance is not None else None
        ),
        harness_hint_count=(
            len(provenance.original_harness_hints) if provenance is not None else 0
        ),
        official_code_available=code.available,
        code_license_known=code.license != "unknown",
        framework_known=code.framework != "unknown",
        source_locations=sorted(locations),
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
    *,
    mechanism_mappings: list[PaperMechanismMapping] | None = None,
    priorities: MechanismPriorityConfig,
) -> PaperImplementationDecision:
    chain = mechanism_mappings or []
    mappings = [mapping for item in resolutions for mapping in item.mappings]
    canonical_ids = sorted({item.canonical_component_id for item in chain})
    if not canonical_ids:
        canonical_ids = sorted({item.canonical_component_id for item in mappings})
    reusable = sorted({
        item.canonical_component_id for item in chain if item.adapter_verified
    })
    if not chain:
        reusable = sorted({
            item.canonical_component_id for item in mappings if item.adapter_verified
        })
    required = sorted(set(canonical_ids) - set(reusable))
    reasons: list[str] = []
    unimplemented: dict[str, list[str]] = {}
    unresolved = sorted(item.paper_component_id for item in resolutions if not item.resolved)
    for item in unresolved:
        unimplemented[item] = [priorities.unresolved_reason(item)]
    if (
        profile.paper_applicability in {"separate_detector_family", "incompatible"}
        or priorities.is_separate_detector_family(profile.paper_detector_family)
    ):
        decision: ImplementationDecisionKind = "separate_detector_family"
        reasons.append("paper_detector_family_routes_method_outside_yolo26")
        reusable = []
    elif not profile.paper_component_ids:
        decision = "insufficient_information"
        reasons.append("paper_does_not_identify_a_component_or_method")
    elif unresolved and not canonical_ids:
        decision = "insufficient_information"
        reasons.append("unresolved_paper_component_alias")
    elif _paper_specific_separate_track(chain):
        decision = "separate_detector_family"
        reasons.append("paper_specific_distillation_requires_separate_detector_family")
        reusable = []
    elif chain and canonical_ids and all(
        item.yolo26_compatibility == "incompatible" for item in chain
    ):
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
        component_adaptation=(
            profile.component_adaptation
            and decision not in {"separate_detector_family", "insufficient_information"}
        ),
        adaptation_mode=_adaptation_mode(
            decision,
            exact_reproduction=profile.exact_reproduction_claim,
        ),
        adaptation_gaps=_adaptation_gaps(
            profile,
            resolutions,
            mechanism_mappings=chain,
            decision=decision,
            required_adapter_ids=required,
            priorities=priorities,
        ),
        mechanism_mappings=chain,
    )
    return result.with_hash()


def _paper_specific_separate_track(
    mappings: list[PaperMechanismMapping],
) -> bool:
    separate_components = {
        "distillation.cross_modal",
        "distillation.vision_language",
    }
    return any(
        item.canonical_component_id in separate_components for item in mappings
    )


def _adaptation_mode(
    decision: ImplementationDecisionKind,
    *,
    exact_reproduction: bool,
) -> PaperAdaptationMode:
    if exact_reproduction:
        return "exact_reproduction"
    if decision == "separate_detector_family":
        return "separate_detector_family"
    if decision == "insufficient_information":
        return "insufficient_information"
    return "component_adaptation"


def _adaptation_gaps(
    profile: PaperMethodProfile,
    resolutions: list[ComponentAliasResolution],
    *,
    mechanism_mappings: list[PaperMechanismMapping],
    decision: ImplementationDecisionKind,
    required_adapter_ids: list[str],
    priorities: MechanismPriorityConfig,
) -> list[PaperAdaptationGap]:
    gaps: list[PaperAdaptationGap] = []
    has_mechanism = bool(mechanism_mappings)
    for resolution in resolutions:
        if resolution.resolved:
            continue
        reason_code = priorities.unresolved_reason(resolution.paper_component_id)
        required_evidence = [
            "explicit method mechanism in summary, note, harness hint, or official code metadata",
            "curated canonical mechanism alias",
        ]
        if reason_code != "canonical_component_mapping_required":
            required_evidence = [
                "paper-specific mechanism beyond the task or detector-family label"
            ]
        gaps.append(PaperAdaptationGap(
            field_name="canonical_component_ids",
            reason_code=reason_code,
            severity="non_blocking" if has_mechanism else "blocking",
            observed_value=resolution.paper_component_id,
            paper_component_id=resolution.paper_component_id,
            source_locations=["paper_record.component_ids"],
            required_evidence=required_evidence,
        ))
    if not profile.paper_component_ids:
        gaps.append(PaperAdaptationGap(
            field_name="paper_component_ids",
            reason_code="paper_does_not_identify_a_component_or_method",
            severity="blocking",
            observed_value=[],
            source_locations=profile.source_locations,
            required_evidence=["explicit paper component or method name"],
        ))
    if profile.method_name == "unknown":
        gaps.append(PaperAdaptationGap(
            field_name="method_name",
            reason_code="method_name_not_explicit",
            severity=(
                "blocking" if decision == "insufficient_information" else "non_blocking"
            ),
            observed_value="unknown",
            source_locations=profile.source_locations,
            required_evidence=["method name with a source location"],
        ))
    if not profile.paper_parameters.get("changed_variables"):
        gaps.append(PaperAdaptationGap(
            field_name="paper_parameters.changed_variables",
            reason_code="changed_variables_not_explicit",
            severity=(
                "blocking"
                if decision in {"new_component_adapter", "new_method_profile"}
                else "non_blocking"
            ),
            observed_value=[],
            source_locations=profile.source_locations,
            required_evidence=["paper-specific changed variable"],
        ))
    if not profile.protocol_constraints.get("insertion_points"):
        gaps.append(PaperAdaptationGap(
            field_name="protocol_constraints.insertion_points",
            reason_code="insertion_point_not_explicit",
            severity=(
                "blocking"
                if decision in {"new_component_adapter", "new_method_profile"}
                else "non_blocking"
            ),
            observed_value=[],
            source_locations=profile.source_locations,
            required_evidence=["paper-specific insertion point"],
        ))
    if not profile.official_code_metadata.available:
        gaps.append(PaperAdaptationGap(
            field_name="official_code_metadata.repository_url",
            reason_code="official_code_metadata_missing",
            severity="non_blocking",
            observed_value=None,
            source_locations=profile.source_locations,
            required_evidence=["offline official code URL metadata"],
        ))
    for component_id in required_adapter_ids:
        gaps.append(PaperAdaptationGap(
            field_name="reusable_adapter_ids",
            reason_code="adapter_not_verified",
            severity="blocking",
            observed_value=component_id,
            paper_component_id=component_id,
            source_locations=profile.source_locations,
            required_evidence=[
                "verified ComponentAdapter implementation",
                "runtime and smoke artifacts bound to adapter hash",
            ],
        ))
    if decision == "separate_detector_family":
        gaps.append(PaperAdaptationGap(
            field_name="paper_detector_family",
            reason_code="yolo26_incompatible_detector_family",
            severity="blocking",
            observed_value=profile.paper_detector_family or profile.paper_applicability,
            source_locations=profile.source_locations,
            required_evidence=["separate detector-family execution protocol"],
        ))
    return sorted(
        gaps,
        key=lambda item: (
            item.severity != "blocking",
            item.field_name,
            item.reason_code,
            item.paper_component_id or "",
        ),
    )


def _mechanism_mapping_chain(
    profile: PaperMethodProfile,
    resolutions: list[ComponentAliasResolution],
    resolver: ComponentAliasResolver,
) -> list[PaperMechanismMapping]:
    chain: list[PaperMechanismMapping] = []
    for resolution in resolutions:
        chain.extend(_mapping_records(
            profile,
            resolution,
            source="catalog_component_id",
            source_location="paper_record.component_ids",
        ))
    for evidence in profile.mechanism_evidence:
        resolution = resolver.resolve(
            evidence.source_term,
            source_paper_ids=[profile.paper_id],
        )
        chain.extend(_mapping_records(
            profile,
            resolution,
            source=evidence.source,
            source_location=evidence.source_location,
        ))
    unique = {
        (
            item.source_term,
            item.source,
            item.source_location,
            item.canonical_component_id,
        ): item
        for item in chain
    }
    return sorted(
        unique.values(),
        key=lambda item: (
            item.canonical_component_id,
            item.source,
            item.source_location,
            item.source_term,
        ),
    )


def _mapping_records(
    profile: PaperMethodProfile,
    resolution: ComponentAliasResolution,
    *,
    source: MechanismMappingSource,
    source_location: str,
) -> list[PaperMechanismMapping]:
    return [
        PaperMechanismMapping(
            paper_id=profile.paper_id,
            profile_id=profile.profile_id,
            source_term=resolution.paper_component_id,
            source=source,
            source_location=source_location,
            canonical_component_id=mapping.canonical_component_id,
            alias_match_type=resolution.match_type,
            yolo26_compatibility=mapping.yolo26_compatibility,
            implementation_status=mapping.implementation_status,
            reusable_adapter_id=(
                mapping.canonical_component_id if mapping.adapter_verified else None
            ),
            adapter_verified=mapping.adapter_verified,
            runtime_execution_ready=mapping.artifact_execution_ready,
        )
        for mapping in resolution.mappings
    ]


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
    "AdaptationGapSeverity",
    "CanonicalMechanismCoverage",
    "CompatibleMechanismCoverage",
    "ImplementationDecisionKind",
    "MechanismMappingSource",
    "PaperAdaptationMode",
    "PaperAdaptationGap",
    "PaperEvidenceInventory",
    "PaperMechanismMapping",
    "PaperImplementationDecision",
    "PaperMethodCoverageReport",
    "PaperMethodProfile",
    "PaperMethodProfileBuilder",
]
