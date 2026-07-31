"""Resolve artifact-backed component maturity before paper recipe execution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
    installed_ultralytics_version,
)
from yolo_agent.research.maturity_snapshot import FrozenComponentMaturity


class EffectiveComponentMaturity(BaseModel):
    """One contract plus the identity of the evidence that made it effective."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    component_id: str
    source_maturity: str
    effective_maturity: str
    contract: ComponentContract
    adapter_hash: str | None = None
    evidence_source: Literal[
        "machine_overlay",
        "frozen_snapshot_artifact",
        "none",
    ] = "none"
    overlay_status: str = "not_checked"
    maturity_artifact_hashes: list[str] = Field(default_factory=list)
    valid_for_training: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)


class EffectiveMaturityResolver:
    """Apply only hash-valid overlays and fail closed below ``smoke_passed``."""

    def __init__(
        self,
        registry: ComponentMaturityRegistry | Path | str | None = None,
        *,
        ultralytics_version: str | None = None,
        certification_protocol_hash: str | None = None,
        frozen_identities: Mapping[str, FrozenComponentMaturity] | None = None,
    ) -> None:
        self.registry = (
            registry
            if isinstance(registry, ComponentMaturityRegistry)
            else ComponentMaturityRegistry(registry)
            if registry is not None
            else None
        )
        self.ultralytics_version = (
            ultralytics_version or installed_ultralytics_version()
        )
        self.certification_protocol_hash = certification_protocol_hash
        self.frozen_identities = dict(frozen_identities or {})

    def resolve(
        self,
        contracts: Mapping[str, ComponentContract],
    ) -> dict[str, EffectiveComponentMaturity]:
        return {
            component_id: self._resolve_one(contract)
            for component_id, contract in sorted(contracts.items())
        }

    def _resolve_one(
        self,
        contract: ComponentContract,
    ) -> EffectiveComponentMaturity:
        reasons: list[str] = []
        try:
            adapter_hash = adapter_source_hash(contract)
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            return EffectiveComponentMaturity(
                component_id=contract.component_id,
                source_maturity=contract.maturity,
                effective_maturity=contract.maturity,
                contract=contract,
                rejection_reasons=[f"adapter_identity_unavailable:{exc}"],
            )

        effective = contract
        overlay_status = "not_configured"
        evidence_source: Literal[
            "machine_overlay", "frozen_snapshot_artifact", "none"
        ] = "none"
        if self.frozen_identities:
            identity = self.frozen_identities.get(contract.component_id)
            overlay_status = "frozen_snapshot"
            if identity is None:
                reasons.append("frozen_runtime_identity_missing")
            else:
                reasons.extend(
                    _frozen_identity_rejections(
                        contract,
                        identity,
                        adapter_hash=adapter_hash,
                        ultralytics_version=self.ultralytics_version,
                    )
                )
                if not reasons and identity.runtime_execution_ready:
                    evidence_source = "frozen_snapshot_artifact"
                elif not identity.runtime_execution_ready:
                    reasons.append("frozen_runtime_not_execution_ready")
        elif self.registry is not None:
            effective, resolution = self.registry.apply(
                contract,
                adapter_hash=adapter_hash,
                ultralytics_version=self.ultralytics_version,
                protocol_hash=self.certification_protocol_hash,
            )
            overlay_status = resolution.status
            if resolution.status == "applied" and effective.can_execute:
                evidence_source = "machine_overlay"
            elif resolution.invalid_artifacts:
                reasons.extend(resolution.invalid_artifacts)

        if (
            evidence_source == "none"
            and contract.can_execute
            and not self.frozen_identities
        ):
            effective = contract
            evidence_source = "frozen_snapshot_artifact"

        valid = effective.can_execute and evidence_source != "none"
        if not valid:
            reasons.append(
                f"effective_maturity_below_smoke_passed:{effective.maturity}"
            )
            if self.registry is not None and overlay_status in {
                "no_match",
                "invalid",
                "not_configured",
            }:
                reasons.append(f"valid_maturity_overlay_required:{overlay_status}")
        return EffectiveComponentMaturity(
            component_id=contract.component_id,
            source_maturity=contract.maturity,
            effective_maturity=effective.maturity,
            contract=effective,
            adapter_hash=adapter_hash,
            evidence_source=evidence_source,
            overlay_status=overlay_status,
            maturity_artifact_hashes=sorted(
                {
                    item.artifact_sha256
                    for item in effective.maturity_artifacts
                    if item.status == "passed" and not item.mock
                }
            ),
            valid_for_training=valid,
            rejection_reasons=list(dict.fromkeys(reasons)),
        )


def _frozen_identity_rejections(
    contract: ComponentContract,
    identity: FrozenComponentMaturity,
    *,
    adapter_hash: str,
    ultralytics_version: str,
) -> list[str]:
    reasons: list[str] = []
    if identity.adapter_hash != adapter_hash:
        reasons.append(
            f"frozen_adapter_hash_mismatch:{identity.adapter_hash}:{adapter_hash}"
        )
    if identity.ultralytics_version != ultralytics_version:
        reasons.append(
            "frozen_ultralytics_version_mismatch:"
            f"{identity.ultralytics_version}:{ultralytics_version}"
        )
    if identity.effective_maturity != contract.maturity:
        reasons.append(
            "frozen_maturity_mismatch:"
            f"{identity.effective_maturity}:{contract.maturity}"
        )
    contract_artifacts = {
        item.artifact_sha256: item for item in contract.maturity_artifacts
    }
    for artifact in identity.artifacts:
        contract_artifact = contract_artifacts.get(artifact.artifact_sha256)
        if contract_artifact is None:
            reasons.append(
                f"frozen_maturity_artifact_missing:{artifact.artifact_sha256}"
            )
            continue
        if (
            identity.protocol_hash != "unavailable"
            and contract_artifact.protocol_hash != identity.protocol_hash
        ):
            reasons.append(
                "frozen_maturity_protocol_mismatch:"
                f"{artifact.artifact_sha256}"
            )
    return reasons


__all__ = ["EffectiveComponentMaturity", "EffectiveMaturityResolver"]
