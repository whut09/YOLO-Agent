"""Serializable contracts for machine-local component maturity evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.maturity import ComponentMaturityArtifact, MaturityName
from yolo_agent.core.yaml_io import YAMLModelMixin


OverlayResolutionStatus = Literal["applied", "no_match", "invalid"]


class ComponentEvidenceOverlay(BaseModel):
    """Artifact evidence produced for one adapter/runtime/protocol identity."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_evidence_overlay.v1"
    component_id: str
    adapter_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_commit: str = Field(min_length=1)
    ultralytics_version: str = Field(min_length=1)
    protocol_hash: str = Field(min_length=1)
    artifacts: list[ComponentMaturityArtifact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_artifact_components(self) -> "ComponentEvidenceOverlay":
        mismatched = [
            item.component_id
            for item in self.artifacts
            if item.component_id != self.component_id
        ]
        if mismatched:
            raise ValueError("overlay artifacts must match component_id")
        return self

    @property
    def identity_key(self) -> str:
        """Return the stable key for one complete execution environment."""
        return _hash(
            {
                "component_id": self.component_id,
                "adapter_hash": self.adapter_hash,
                "code_commit": self.code_commit,
                "ultralytics_version": self.ultralytics_version,
                "protocol_hash": self.protocol_hash,
            }
        )

    @property
    def evidence_hash(self) -> str:
        """Return a timestamp-independent hash for idempotent updates."""
        return _hash(
            {
                "identity_key": self.identity_key,
                "artifacts": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in self.artifacts
                ],
            }
        )


class ComponentMaturityRegistryDocument(BaseModel, YAMLModelMixin):
    """On-disk registry containing machine-local evidence overlays."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_maturity_registry.v1"
    overlays: list[ComponentEvidenceOverlay] = Field(default_factory=list)


class ComponentOverlayResolution(BaseModel):
    """Auditable result of resolving one source contract against overlays."""

    model_config = ConfigDict(extra="forbid")

    status: OverlayResolutionStatus
    component_id: str
    source_maturity: MaturityName
    effective_maturity: MaturityName
    overlay_identity_key: str | None = None
    applied_artifact_hashes: list[str] = Field(default_factory=list)
    retained_artifact_hashes: list[str] = Field(default_factory=list)
    invalid_artifacts: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def _hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ComponentEvidenceOverlay",
    "ComponentMaturityRegistryDocument",
    "ComponentOverlayResolution",
    "OverlayResolutionStatus",
]
