"""Frozen identities for component runtime maturity used by one snapshot."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.maturity import MaturityArtifactType, MaturityName
from yolo_agent.core.yaml_io import YAMLModelMixin


class FrozenMaturityArtifact(BaseModel):
    """Identity of one maturity artifact copied into a research snapshot."""

    model_config = ConfigDict(extra="forbid")

    snapshot_artifact_name: str
    target_maturity: MaturityName
    artifact_type: MaturityArtifactType
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_hash: str | None = None
    status: str
    mock: bool = False


class FrozenComponentMaturity(BaseModel):
    """Effective adapter/runtime identity frozen for one component."""

    model_config = ConfigDict(extra="forbid")

    component_id: str
    adapter_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_commit: str
    ultralytics_version: str
    protocol_hash: str
    overlay_identity_key: str | None = None
    overlay_evidence_hash: str | None = None
    effective_maturity: MaturityName
    runtime_execution_ready: bool = False
    artifacts: list[FrozenMaturityArtifact] = Field(default_factory=list)


class EffectiveComponentMaturityManifest(BaseModel, YAMLModelMixin):
    """Content-addressed effective maturity state consumed during training."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "effective_component_maturity.v1"
    entries: list[FrozenComponentMaturity] = Field(default_factory=list)

    @property
    def manifest_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def by_component(self) -> dict[str, FrozenComponentMaturity]:
        return {item.component_id: item for item in self.entries}


__all__ = [
    "EffectiveComponentMaturityManifest",
    "FrozenComponentMaturity",
    "FrozenMaturityArtifact",
]
