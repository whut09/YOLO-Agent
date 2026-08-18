"""Conservative paper-component alias resolution against local contracts."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.base import ComponentAdapter
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.maturity import MaturityName, maturity_rank
from yolo_agent.research.schemas import ComponentCategory
from yolo_agent.resources import ResourcePaths


AliasMatchType = Literal["exact_match", "normalized_match", "semantic_match", "unresolved"]
YOLO26Compatibility = Literal["compatible", "adapter_required", "incompatible", "unknown"]
ImplementationStatus = Literal[
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
]


class CanonicalComponentDefinition(BaseModel):
    """One canonical taxonomy entry and its curated aliases."""

    model_config = ConfigDict(extra="forbid")

    canonical_component_id: str
    category: ComponentCategory
    aliases: list[str] = Field(default_factory=list)
    semantic_aliases: list[str] = Field(default_factory=list)
    detector_families: list[str] = Field(default_factory=lambda: ["generic"])
    target_error_types: list[str] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)
    insertion_point: str = "unknown"
    yolo26_compatibility: YOLO26Compatibility = "unknown"
    reusable_adapter_ids: list[str] = Field(default_factory=list)
    mapping_reason: str


class CompoundAliasDefinition(BaseModel):
    """An intentionally split paper concept mapped to multiple components."""

    model_config = ConfigDict(extra="forbid")

    aliases: list[str]
    semantic_aliases: list[str] = Field(default_factory=list)
    canonical_component_ids: list[str] = Field(min_length=2)
    split_reason: str


class PaperMechanismDefinition(BaseModel):
    """Explicit paper mechanism identity used before canonical routing."""

    model_config = ConfigDict(extra="forbid")

    paper_specific_mechanism_id: str
    aliases: list[str] = Field(default_factory=list)
    canonical_component_id: str
    implementation_family: str
    changed_variables: list[str] = Field(default_factory=list)
    runtime_payload_schema: dict[str, object] = Field(default_factory=dict)
    evidence_protocol: list[str] = Field(default_factory=list)
    required_adapter: str
    required_evidence: list[str] = Field(default_factory=list)
    compatibility: YOLO26Compatibility = "unknown"
    generic_parent_component_id: str | None = None


class ComponentAliasConfig(BaseModel):
    """Validated alias catalog; accidental overlaps are configuration errors."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_aliases.v1"
    canonical_components: list[CanonicalComponentDefinition]
    compound_aliases: list[CompoundAliasDefinition] = Field(default_factory=list)
    paper_mechanisms: list[PaperMechanismDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_aliases(self) -> "ComponentAliasConfig":
        canonical_ids = [item.canonical_component_id for item in self.canonical_components]
        if len(canonical_ids) != len(set(canonical_ids)):
            raise ValueError("canonical_component_id values must be unique")
        known = set(canonical_ids)
        owners: dict[str, str] = {}
        for item in self.canonical_components:
            terms = [item.canonical_component_id, *item.aliases, *item.semantic_aliases]
            _claim_terms(owners, terms, item.canonical_component_id)
        for index, item in enumerate(self.compound_aliases):
            missing = sorted(set(item.canonical_component_ids) - known)
            if missing:
                raise ValueError(f"compound alias references unknown canonical components: {missing}")
            _claim_terms(owners, [*item.aliases, *item.semantic_aliases], f"compound:{index}")
        paper_ids = [item.paper_specific_mechanism_id for item in self.paper_mechanisms]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("paper_specific_mechanism_id values must be unique")
        paper_owners: dict[str, str] = {}
        for item in self.paper_mechanisms:
            if item.canonical_component_id not in known:
                raise ValueError(
                    "paper mechanism references unknown canonical component: "
                    + item.canonical_component_id
                )
            _claim_terms(
                paper_owners,
                [item.paper_specific_mechanism_id, *item.aliases],
                item.paper_specific_mechanism_id,
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path | str = ResourcePaths.COMPONENT_ALIASES) -> "ComponentAliasConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        return cls.model_validate(payload)


class ResolvedComponentAlias(BaseModel):
    """One canonical mapping with audited local implementation status."""

    model_config = ConfigDict(extra="forbid")

    canonical_component_id: str
    category: ComponentCategory
    detector_families: list[str]
    target_error_types: list[str]
    target_metrics: list[str]
    insertion_point: str
    yolo26_compatibility: YOLO26Compatibility
    maturity: MaturityName
    implementation_status: ImplementationStatus
    alias_confidence: float = Field(ge=0.0, le=1.0)
    mapping_reason: str
    reusable_adapter_ids: list[str] = Field(default_factory=list)
    verified_adapter_ids: list[str] = Field(default_factory=list)
    runtime_ready_adapter_ids: list[str] = Field(default_factory=list)
    adapter_verified: bool = False
    artifact_execution_ready: bool = False

    @property
    def executable(self) -> bool:
        return self.adapter_verified and self.artifact_execution_ready


class ComponentAliasResolution(BaseModel):
    """Resolution result for one paper-supplied component identifier."""

    model_config = ConfigDict(extra="forbid")

    paper_component_id: str
    normalized_component_id: str
    match_type: AliasMatchType
    source_paper_ids: list[str] = Field(default_factory=list)
    mappings: list[ResolvedComponentAlias] = Field(default_factory=list)
    split_reason: str | None = None
    unresolved_reason: str | None = None

    @property
    def resolved(self) -> bool:
        return self.match_type != "unresolved" and bool(self.mappings)


class ComponentAliasResolver:
    """Resolve aliases without inferring adapter availability from names."""

    def __init__(
        self,
        config: ComponentAliasConfig,
        *,
        contracts: list[ComponentContract] | None = None,
        validate_reusable_adapters: bool = False,
    ) -> None:
        self.config = config
        self.contracts = {item.component_id: item for item in (contracts or [])}
        self._definitions = {item.canonical_component_id: item for item in config.canonical_components}
        if validate_reusable_adapters:
            missing_adapters = sorted(
                {
                    adapter_id
                    for definition in config.canonical_components
                    for adapter_id in definition.reusable_adapter_ids
                    if adapter_id not in self.contracts
                }
            )
            if missing_adapters:
                raise ValueError(
                    "canonical mechanism references unknown reusable adapters: "
                    f"{missing_adapters}"
                )

    @classmethod
    def from_yaml(
        cls,
        config_path: Path | str = ResourcePaths.COMPONENT_ALIASES,
        *,
        contract_paths: list[Path | str] | None = None,
    ) -> "ComponentAliasResolver":
        paths = contract_paths or [
            ResourcePaths.COMPONENT_COMPATIBILITY,
            ResourcePaths.CONFIG_DIR / "components",
        ]
        return cls(
            ComponentAliasConfig.from_yaml(config_path),
            contracts=_load_available_contracts(paths),
            validate_reusable_adapters=contract_paths is None,
        )

    def resolve(
        self,
        component_id: str,
        *,
        source_paper_ids: list[str] | None = None,
    ) -> ComponentAliasResolution:
        raw = component_id.strip()
        normalized = normalize_component_id(raw)
        source_ids = sorted(set(source_paper_ids or []))

        compound = self._match_compound(raw, normalized)
        if compound is not None:
            definition, match_type = compound
            return ComponentAliasResolution(
                paper_component_id=raw,
                normalized_component_id=normalized,
                match_type=match_type,
                source_paper_ids=source_ids,
                mappings=[
                    self._resolved_mapping(self._definitions[item], match_type, definition.split_reason)
                    for item in definition.canonical_component_ids
                ],
                split_reason=definition.split_reason,
            )

        matched = self._match_canonical(raw, normalized)
        if matched is None:
            return ComponentAliasResolution(
                paper_component_id=raw,
                normalized_component_id=normalized,
                match_type="unresolved",
                source_paper_ids=source_ids,
                unresolved_reason="No explicit canonical, normalized, or curated semantic alias matched.",
            )
        definition, match_type = matched
        return ComponentAliasResolution(
            paper_component_id=raw,
            normalized_component_id=normalized,
            match_type=match_type,
            source_paper_ids=source_ids,
            mappings=[self._resolved_mapping(definition, match_type)],
        )

    def adapter_verified(self, component_id: str) -> bool:
        """Return whether the configured adapter class imports and implements the SDK."""
        return _contract_adapter_verified(self.contracts.get(component_id))

    def _match_canonical(
        self,
        raw: str,
        normalized: str,
    ) -> tuple[CanonicalComponentDefinition, AliasMatchType] | None:
        for item in self.config.canonical_components:
            if raw == item.canonical_component_id or raw in item.aliases:
                return item, "exact_match"
        for item in self.config.canonical_components:
            if normalized in {normalize_component_id(value) for value in [item.canonical_component_id, *item.aliases]}:
                return item, "normalized_match"
        for item in self.config.canonical_components:
            if normalized in {normalize_component_id(value) for value in item.semantic_aliases}:
                return item, "semantic_match"
        return None

    def _match_compound(
        self,
        raw: str,
        normalized: str,
    ) -> tuple[CompoundAliasDefinition, AliasMatchType] | None:
        for item in self.config.compound_aliases:
            if raw in item.aliases:
                return item, "exact_match"
        for item in self.config.compound_aliases:
            if normalized in {normalize_component_id(value) for value in item.aliases}:
                return item, "normalized_match"
        for item in self.config.compound_aliases:
            if normalized in {normalize_component_id(value) for value in item.semantic_aliases}:
                return item, "semantic_match"
        return None

    def _resolved_mapping(
        self,
        definition: CanonicalComponentDefinition,
        match_type: AliasMatchType,
        split_reason: str | None = None,
    ) -> ResolvedComponentAlias:
        adapter_ids = definition.reusable_adapter_ids or [
            definition.canonical_component_id
        ]
        adapter_contracts = {
            adapter_id: self.contracts.get(adapter_id) for adapter_id in adapter_ids
        }
        verified_adapter_ids = sorted(
            adapter_id
            for adapter_id, contract in adapter_contracts.items()
            if _contract_adapter_verified(contract)
        )
        runtime_ready_adapter_ids = sorted(
            adapter_id
            for adapter_id in verified_adapter_ids
            if adapter_contracts[adapter_id] is not None
            and adapter_contracts[adapter_id].can_execute
        )
        contract = _most_mature_contract(
            [
                adapter_contracts[adapter_id]
                for adapter_id in verified_adapter_ids
                if adapter_contracts[adapter_id] is not None
            ]
        )
        adapter_verified = bool(verified_adapter_ids)
        maturity = contract.maturity if contract is not None else "metadata_only"
        status = _implementation_status(contract, adapter_verified)
        compatibility = _yolo26_compatibility(definition, contract, adapter_verified)
        reason = definition.mapping_reason
        if split_reason:
            reason = f"{reason} Split from a broad paper concept: {split_reason}"
        if definition.reusable_adapter_ids:
            reason = (
                f"{reason} Reusable component-adaptation adapters: "
                f"{', '.join(definition.reusable_adapter_ids)}; this does not claim "
                "paper-exact reproduction."
            )
        if contract is not None:
            reason = f"{reason} Local contract maturity audited as {contract.maturity}; alias resolution did not grant it."
        return ResolvedComponentAlias(
            canonical_component_id=definition.canonical_component_id,
            category=definition.category,
            detector_families=(contract.supported_detector_families if contract else definition.detector_families),
            target_error_types=definition.target_error_types,
            target_metrics=definition.target_metrics,
            insertion_point=(contract.insertion_point if contract else definition.insertion_point),
            yolo26_compatibility=compatibility,
            maturity=maturity,
            implementation_status=status,
            alias_confidence={"exact_match": 1.0, "normalized_match": 0.95, "semantic_match": 0.8}[match_type],
            mapping_reason=reason,
            reusable_adapter_ids=definition.reusable_adapter_ids,
            verified_adapter_ids=verified_adapter_ids,
            runtime_ready_adapter_ids=runtime_ready_adapter_ids,
            adapter_verified=adapter_verified,
            artifact_execution_ready=bool(runtime_ready_adapter_ids),
        )


def _most_mature_contract(
    contracts: list[ComponentContract],
) -> ComponentContract | None:
    if not contracts:
        return None
    return max(contracts, key=lambda item: maturity_rank(item.maturity))


def normalize_component_id(value: str) -> str:
    """Normalize spelling only; this intentionally performs no prefix inference."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value.strip())
    return re.sub(r"[^a-z0-9]+", "_", spaced.lower()).strip("_")


def _claim_terms(owners: dict[str, str], terms: list[str], owner: str) -> None:
    for term in terms:
        normalized = normalize_component_id(term)
        if not normalized:
            raise ValueError(f"empty alias configured for {owner}")
        previous = owners.get(normalized)
        if previous is not None and previous != owner:
            raise ValueError(f"conflicting component alias {term!r}: {previous} vs {owner}")
        owners[normalized] = owner


def _load_available_contracts(paths: list[Path | str]) -> list[ComponentContract]:
    contracts: dict[str, ComponentContract] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else sorted(path.rglob("*.yaml"))
        for candidate in candidates:
            payload = yaml.safe_load(candidate.read_text(encoding="utf-8-sig")) or {}
            if not isinstance(payload, dict) or not isinstance(payload.get("components"), dict):
                continue
            for contract in load_contracts(candidate):
                existing = contracts.get(contract.component_id)
                if existing is None or maturity_rank(contract.maturity) > maturity_rank(existing.maturity):
                    contracts[contract.component_id] = contract
    return [contracts[key] for key in sorted(contracts)]


def _contract_adapter_verified(contract: ComponentContract | None) -> bool:
    if contract is None or not contract.implementation_path or not contract.adapter_class:
        return False
    try:
        module = importlib.import_module(contract.implementation_path)
        adapter_type = getattr(module, contract.adapter_class, None)
    except Exception:
        return False
    return isinstance(adapter_type, type) and issubclass(adapter_type, ComponentAdapter)


def _implementation_status(
    contract: ComponentContract | None,
    adapter_verified: bool,
) -> ImplementationStatus:
    if contract is None:
        return "metadata_only"
    if not adapter_verified:
        return "metadata_only" if contract.maturity == "metadata_only" else "adapter_required"
    if maturity_rank(contract.maturity) >= maturity_rank("adapter_implemented"):
        return contract.maturity
    return "adapter_required"


def _yolo26_compatibility(
    definition: CanonicalComponentDefinition,
    contract: ComponentContract | None,
    adapter_verified: bool,
) -> YOLO26Compatibility:
    if contract is None:
        return definition.yolo26_compatibility
    families = set(contract.supported_detector_families)
    if "generic" not in families and "yolo26" not in families:
        return "incompatible"
    if contract.fixed_imgsz_compatible is False or "yolo26_one_to_one" in contract.incompatible_heads:
        return "incompatible"
    if adapter_verified and maturity_rank(contract.maturity) >= maturity_rank("smoke_passed"):
        return "compatible"
    return "adapter_required"


__all__ = [
    "AliasMatchType",
    "CanonicalComponentDefinition",
    "ComponentAliasConfig",
    "ComponentAliasResolution",
    "ComponentAliasResolver",
    "CompoundAliasDefinition",
    "PaperMechanismDefinition",
    "ImplementationStatus",
    "ResolvedComponentAlias",
    "YOLO26Compatibility",
    "normalize_component_id",
]
