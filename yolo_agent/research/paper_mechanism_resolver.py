"""Resolve paper-specific mechanisms before canonical runtime routing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.research.component_aliases import (
    ComponentAliasConfig,
    PaperMechanismDefinition,
    YOLO26Compatibility,
    normalize_component_id,
)


GENERIC_MECHANISM_IDS = frozenset(
    {
        "distillation.yolo26_teacher_student",
        "domain_adaptation.general",
        "quality_alignment.general",
    }
)


class PaperMechanismResolution(BaseModel):
    """One paper mechanism identity and its runtime boundary."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    original_method_name: str
    paper_specific_mechanism_id: str | None = None
    canonical_component_id: str | None = None
    implementation_family: str | None = None
    paper_config_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility: YOLO26Compatibility
    required_adapter: str | None = None
    required_evidence: list[str] = Field(default_factory=list)
    unresolved_reason: str | None = None
    changed_variables: list[str] = Field(default_factory=list)
    runtime_payload_schema: dict[str, Any] = Field(default_factory=dict)
    evidence_protocol: list[str] = Field(default_factory=list)
    execution_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_terms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resolution(self) -> "PaperMechanismResolution":
        resolved = bool(
            self.paper_specific_mechanism_id and self.canonical_component_id
        )
        if resolved and self.unresolved_reason:
            raise ValueError("resolved mechanism cannot retain unresolved_reason")
        if not resolved and not self.unresolved_reason:
            raise ValueError("unresolved mechanism requires unresolved_reason")
        if (
            self.paper_specific_mechanism_id in GENERIC_MECHANISM_IDS
            or self.paper_specific_mechanism_id is not None
            and self.paper_specific_mechanism_id.endswith(".general")
        ):
            raise ValueError("generic alias cannot become paper-specific")
        return self

    @property
    def resolved(self) -> bool:
        return bool(
            self.paper_specific_mechanism_id and self.canonical_component_id
        )

    @property
    def executable_candidate(self) -> bool:
        return bool(
            self.resolved
            and self.compatibility == "compatible"
            and self.required_adapter
            and not self.unresolved_reason
        )


class PaperMechanismResolutionSet(BaseModel):
    """All distinct mechanism implementations found for one paper."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    resolutions: list[PaperMechanismResolution] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_membership(self) -> "PaperMechanismResolutionSet":
        if any(item.paper_id != self.paper_id for item in self.resolutions):
            raise ValueError("resolution set paper_id mismatch")
        fingerprints = [item.execution_fingerprint for item in self.resolutions]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("paper contains duplicate mechanism implementations")
        return self


class PaperMechanismExecutionGroup(BaseModel):
    """Multiple papers that resolve to one identical execution."""

    model_config = ConfigDict(extra="forbid")

    execution_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    paper_ids: list[str] = Field(min_length=1)
    paper_specific_mechanism_id: str
    canonical_component_id: str
    implementation_family: str
    paper_config_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_variables: list[str]
    runtime_payload_schema: dict[str, Any]
    evidence_protocol: list[str]
    required_adapter: str


class PaperMechanismResolver:
    """Resolve only explicit mechanism evidence; unknowns stay unresolved."""

    def __init__(
        self,
        config: ComponentAliasConfig,
        *,
        contracts: Mapping[str, ComponentContract] | None = None,
    ) -> None:
        self.config = config
        self.contracts = dict(contracts or {})
        self._definitions = list(config.paper_mechanisms)
        self._canonical_definitions = {
            item.canonical_component_id: item
            for item in config.canonical_components
        }
        self._definition_terms = {
            item.paper_specific_mechanism_id: {
                normalize_component_id(term)
                for term in (
                    item.paper_specific_mechanism_id,
                    item.canonical_component_id,
                    *item.aliases,
                    *item.changed_variables,
                )
            }
            for item in self._definitions
        }

    @classmethod
    def from_alias_config(
        cls,
        config: ComponentAliasConfig | None = None,
        *,
        contracts: Iterable[ComponentContract] = (),
    ) -> "PaperMechanismResolver":
        return cls(
            config or ComponentAliasConfig.from_yaml(),
            contracts={item.component_id: item for item in contracts},
        )

    def resolve_profile(
        self,
        profile: Any,
        decision: Any,
    ) -> PaperMechanismResolutionSet:
        terms = _profile_evidence_terms(profile, decision)
        normalized_terms = {normalize_component_id(item) for item in terms}
        matched = [
            definition
            for definition in self._definitions
            if normalized_terms & self._definition_terms[
                definition.paper_specific_mechanism_id
            ]
        ]
        resolutions = [
            self._resolved(profile, definition, terms) for definition in matched
        ]
        if not resolutions:
            resolutions.extend(
                self._fallback_specific_resolutions(profile, decision, terms)
            )
        if not resolutions:
            resolutions = [self._unresolved(profile, decision, terms)]
        return PaperMechanismResolutionSet(
            paper_id=profile.paper_id,
            resolutions=sorted(
                resolutions,
                key=lambda item: item.execution_fingerprint,
            ),
        )

    def _resolved(
        self,
        profile: Any,
        definition: PaperMechanismDefinition,
        terms: list[str],
    ) -> PaperMechanismResolution:
        contract = self.contracts.get(definition.required_adapter)
        changed = sorted(
            set(definition.changed_variables)
            | set(profile.paper_parameters.get("changed_variables", []))
        )
        payload = dict(definition.runtime_payload_schema)
        protocol = sorted(set(definition.evidence_protocol))
        compatibility = _effective_compatibility(
            definition.compatibility, contract
        )
        config_signature = _config_signature(
            changed,
            payload,
            protocol,
            profile.protocol_constraints,
        )
        fingerprint = _execution_fingerprint(
            definition.paper_specific_mechanism_id,
            definition.canonical_component_id,
            definition.implementation_family,
            config_signature,
            changed,
            payload,
            protocol,
            definition.required_adapter,
        )
        return PaperMechanismResolution(
            paper_id=profile.paper_id,
            original_method_name=_original_method_name(
                profile, definition.paper_specific_mechanism_id
            ),
            paper_specific_mechanism_id=definition.paper_specific_mechanism_id,
            canonical_component_id=definition.canonical_component_id,
            implementation_family=definition.implementation_family,
            paper_config_signature=config_signature,
            compatibility=compatibility,
            required_adapter=definition.required_adapter,
            required_evidence=sorted(set(definition.required_evidence)),
            changed_variables=changed,
            runtime_payload_schema=payload,
            evidence_protocol=protocol,
            execution_fingerprint=fingerprint,
            evidence_terms=terms,
        )

    def _fallback_specific_resolutions(
        self,
        profile: Any,
        decision: Any,
        terms: list[str],
    ) -> list[PaperMechanismResolution]:
        canonical_ids = sorted(
            set(getattr(decision, "canonical_component_ids", []))
            - GENERIC_MECHANISM_IDS
        )
        results: list[PaperMechanismResolution] = []
        for component_id in canonical_ids:
            contract = self.contracts.get(component_id)
            canonical_definition = self._canonical_definitions.get(component_id)
            changed = sorted(
                set(profile.paper_parameters.get("changed_variables", []))
            )
            payload = _contract_payload_schema(contract)
            protocol = sorted(
                set(contract.evidence_protocol if contract else [])
                or {"matched_control"}
            )
            family = (
                contract.implementation_family
                if contract and contract.implementation_family
                else component_id
            )
            adapter = component_id
            config_signature = _config_signature(
                changed, payload, protocol, profile.protocol_constraints
            )
            fingerprint = _execution_fingerprint(
                component_id,
                component_id,
                family,
                config_signature,
                changed,
                payload,
                protocol,
                adapter,
            )
            results.append(PaperMechanismResolution(
                paper_id=profile.paper_id,
                original_method_name=_original_method_name(profile, component_id),
                paper_specific_mechanism_id=component_id,
                canonical_component_id=component_id,
                implementation_family=family,
                paper_config_signature=config_signature,
                compatibility=(
                    "incompatible"
                    if canonical_definition is not None
                    and canonical_definition.yolo26_compatibility
                    == "incompatible"
                    else (
                        "compatible"
                        if contract is not None and contract.can_execute
                        else "adapter_required"
                    )
                ),
                required_adapter=adapter,
                required_evidence=["paper_specific_mechanism_evidence", "matched_control"],
                changed_variables=changed,
                runtime_payload_schema=payload,
                evidence_protocol=protocol,
                execution_fingerprint=fingerprint,
                evidence_terms=terms,
            ))
        return results

    def _unresolved(
        self,
        profile: Any,
        decision: Any,
        terms: list[str],
    ) -> PaperMechanismResolution:
        generic = sorted(
            set(getattr(decision, "canonical_component_ids", []))
            & GENERIC_MECHANISM_IDS
        )
        reason = (
            "generic mechanism lacks paper-specific changed variables and runtime payload"
            if generic
            else "no explicit paper-specific mechanism matched"
        )
        changed = sorted(
            set(profile.paper_parameters.get("changed_variables", []))
        )
        config_signature = _config_signature(
            changed, {}, [], profile.protocol_constraints
        )
        fingerprint = _execution_fingerprint(
            "unresolved",
            generic[0] if generic else "unresolved",
            "implementation_request",
            config_signature,
            changed,
            {},
            [],
            "unresolved",
        )
        return PaperMechanismResolution(
            paper_id=profile.paper_id,
            original_method_name=_original_method_name(profile, "unknown"),
            paper_config_signature=config_signature,
            compatibility="unknown",
            required_evidence=[
                "paper_specific_mechanism",
                "changed_variables",
                "runtime_payload_schema",
                "evidence_protocol",
            ],
            unresolved_reason=reason,
            changed_variables=changed,
            execution_fingerprint=fingerprint,
            evidence_terms=terms,
        )


def merge_paper_mechanism_resolutions(
    resolutions: Iterable[PaperMechanismResolution],
) -> list[PaperMechanismExecutionGroup]:
    """Merge papers only when every execution-defining field is identical."""
    grouped: dict[str, list[PaperMechanismResolution]] = {}
    for item in resolutions:
        if not item.resolved:
            continue
        grouped.setdefault(item.execution_fingerprint, []).append(item)
    output: list[PaperMechanismExecutionGroup] = []
    for fingerprint, items in sorted(grouped.items()):
        first = items[0]
        identity = _merge_identity(first)
        if any(_merge_identity(item) != identity for item in items[1:]):
            raise ValueError("execution fingerprint collision across mechanisms")
        output.append(PaperMechanismExecutionGroup(
            execution_fingerprint=fingerprint,
            paper_ids=sorted({item.paper_id for item in items}),
            paper_specific_mechanism_id=first.paper_specific_mechanism_id or "",
            canonical_component_id=first.canonical_component_id or "",
            implementation_family=first.implementation_family or "",
            paper_config_signature=first.paper_config_signature,
            changed_variables=first.changed_variables,
            runtime_payload_schema=first.runtime_payload_schema,
            evidence_protocol=first.evidence_protocol,
            required_adapter=first.required_adapter or "",
        ))
    return output


def _profile_evidence_terms(profile: Any, decision: Any) -> list[str]:
    terms: set[str] = set(profile.paper_component_ids)
    if profile.method_name != "unknown":
        terms.add(profile.method_name)
    terms.update(profile.method_names)
    terms.update(profile.paper_parameters.get("changed_variables", []))
    terms.update(profile.protocol_constraints.get("insertion_points", []))
    for item in profile.mechanism_evidence:
        if item.source != "title":
            terms.update({item.source_term, item.canonical_component_id})
    structured = profile.structured_method_evidence
    if structured is not None:
        for item in structured.observations:
            if item.source != "title" and isinstance(item.value, str):
                terms.add(item.value)
    for item in getattr(decision, "mechanism_mappings", []):
        if item.source != "title" and item.authorizes_method_profile:
            terms.update({item.source_term, item.canonical_component_id})
    return sorted(item for item in terms if item)


def _effective_compatibility(
    configured: YOLO26Compatibility,
    contract: ComponentContract | None,
) -> YOLO26Compatibility:
    if configured == "incompatible":
        return configured
    if contract is None:
        return "adapter_required"
    if contract.can_execute:
        return "compatible"
    return "adapter_required"


def _contract_payload_schema(
    contract: ComponentContract | None,
) -> dict[str, Any]:
    if contract is None:
        return {}
    return contract.runtime_payload_schema or {
        "input": contract.tensor_input_contract,
        "output": contract.tensor_output_contract,
    }


def _config_signature(
    changed_variables: list[str],
    runtime_payload_schema: dict[str, Any],
    evidence_protocol: list[str],
    protocol_constraints: dict[str, Any],
) -> str:
    return _hash({
        "changed_variables": sorted(changed_variables),
        "runtime_payload_schema": runtime_payload_schema,
        "evidence_protocol": sorted(evidence_protocol),
        "protocol_constraints": protocol_constraints,
    })


def _execution_fingerprint(
    mechanism_id: str,
    canonical_component_id: str,
    implementation_family: str,
    config_signature: str,
    changed_variables: list[str],
    runtime_payload_schema: dict[str, Any],
    evidence_protocol: list[str],
    required_adapter: str,
) -> str:
    return _hash({
        "paper_specific_mechanism_id": mechanism_id,
        "canonical_component_id": canonical_component_id,
        "implementation_family": implementation_family,
        "paper_config_signature": config_signature,
        "changed_variables": sorted(changed_variables),
        "runtime_payload_schema": runtime_payload_schema,
        "evidence_protocol": sorted(evidence_protocol),
        "required_adapter": required_adapter,
    })


def _merge_identity(item: PaperMechanismResolution) -> tuple[Any, ...]:
    return (
        item.paper_specific_mechanism_id,
        tuple(item.changed_variables),
        json.dumps(item.runtime_payload_schema, sort_keys=True),
        tuple(item.evidence_protocol),
        item.execution_fingerprint,
    )


def _original_method_name(profile: Any, fallback: str) -> str:
    if profile.method_name != "unknown":
        return profile.method_name
    if profile.method_names:
        return profile.method_names[0]
    return fallback


def _hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "GENERIC_MECHANISM_IDS",
    "PaperMechanismExecutionGroup",
    "PaperMechanismResolution",
    "PaperMechanismResolutionSet",
    "PaperMechanismResolver",
    "merge_paper_mechanism_resolutions",
]
