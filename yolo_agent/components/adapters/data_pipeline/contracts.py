"""Typed contracts shared by train-only data pipeline adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DataMechanismKind = Literal["weighted_sampler", "replay", "transform", "schedule"]


class DataPipelineIdentity(BaseModel):
    """Runtime identity for one mechanism, never a paper reproduction claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_id: str
    component_id: str
    adapter_family: str
    mechanism_kind: DataMechanismKind
    changed_variable: str
    exact_reproduction: bool = False

    @field_validator(
        "mechanism_id",
        "component_id",
        "adapter_family",
        "changed_variable",
    )
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("data pipeline identity fields must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_changed_variable(self) -> "DataPipelineIdentity":
        expected = f"data.{self.mechanism_id}"
        if self.changed_variable != expected:
            raise ValueError(
                f"changed_variable must identify one mechanism exactly: {expected}"
            )
        return self


class DataSampleRecord(BaseModel):
    """Minimal annotation surface used by samplers and transform selectors."""

    model_config = ConfigDict(extra="forbid")

    image_path: str
    split: str = "train"
    normalized_areas: list[float] = Field(default_factory=list)
    class_ids: list[int] = Field(default_factory=list)
    is_hard_negative: bool = False
    false_negative_score: float = Field(default=0.0, ge=0.0)


class DataPipelineManifest(BaseModel):
    """Machine-readable exposure/transform contract emitted by one mechanism."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "data_pipeline_manifest.v1"
    identity: DataPipelineIdentity
    dataset_manifest: str
    protocol_hash: str = "unbound"
    runtime_payload_hash: str = "unbound"
    adapter_hash: str
    plugin_version: str
    split: str = "train"
    seed: int = Field(default=0, ge=0)
    epoch: int = Field(default=0, ge=0)
    rank: int = Field(default=0, ge=0)
    world_size: int = Field(default=1, ge=1)
    image_paths: list[str] = Field(default_factory=list)
    class_counts: dict[str, int] = Field(default_factory=dict)
    raw_exposure: list[float] = Field(default_factory=list)
    final_exposure: list[float] = Field(default_factory=list)
    selected_indices: list[int] = Field(default_factory=list)
    transform_parameters: dict[str, Any] = Field(default_factory=dict)
    clipping_statistics: dict[str, int | float] = Field(default_factory=dict)
    sample_count: int = Field(default=0, ge=0)
    val_unchanged: bool = True
    test_unchanged: bool = True
    exact_reproduction: bool = False
    paper_method_profiles: list[str] = Field(default_factory=list)
    manifest_hash: str = ""

    @model_validator(mode="after")
    def validate_manifest(self) -> "DataPipelineManifest":
        if self.split != "train":
            raise ValueError("data pipeline adapters may emit train manifests only")
        if not self.val_unchanged or not self.test_unchanged:
            raise ValueError("data pipeline adapters cannot modify val/test dataloaders")
        if self.exact_reproduction and not self.paper_method_profiles:
            raise ValueError("exact reproduction requires explicit MethodProfile identity")
        if self.raw_exposure and len(self.raw_exposure) != len(self.image_paths):
            raise ValueError("raw exposure must align with image paths")
        if self.final_exposure and len(self.final_exposure) != len(self.image_paths):
            raise ValueError("final exposure must align with image paths")
        return self

    def with_hash(self) -> "DataPipelineManifest":
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return self.model_copy(update={"manifest_hash": digest})

    def write(self, path: str | Path) -> Path:
        from yolo_agent.components.adapters.data_pipeline.runtime import (
            write_json_atomic,
        )

        target = Path(path)
        return write_json_atomic(
            target,
            self.with_hash().model_dump(mode="json"),
        )


__all__ = [
    "DataMechanismKind",
    "DataPipelineIdentity",
    "DataPipelineManifest",
    "DataSampleRecord",
]
