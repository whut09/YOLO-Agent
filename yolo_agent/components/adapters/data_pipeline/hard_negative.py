"""Split-safe local hard-negative manifest used by replay sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HardNegativeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    sample_index: int = Field(ge=0)
    predicted_class: int | None = None
    score: float | None = None
    bbox: list[float] = Field(default_factory=list)
    error_type: str


class HardNegativeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "hard_negative_manifest.v1"
    dataset_manifest_hash: str
    source_split: str
    source_run_id: str
    baseline_protocol_hash: str
    records: list[HardNegativeRecord] = Field(default_factory=list)
    manifest_hash: str = ""

    @model_validator(mode="after")
    def validate_manifest(self) -> "HardNegativeManifest":
        if self.source_split != "train":
            raise ValueError("hard-negative replay requires a train split manifest")
        indices = [item.sample_index for item in self.records]
        if len(indices) != len(set(indices)):
            raise ValueError("hard-negative manifest contains duplicate sample indices")
        expected = self.compute_hash()
        if self.manifest_hash and self.manifest_hash != expected:
            raise ValueError("hard-negative manifest hash mismatch")
        self.manifest_hash = expected
        return self

    def compute_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def sample_indices(self) -> list[int]:
        return sorted(item.sample_index for item in self.records)

    @classmethod
    def from_path(cls, path: Path | str) -> "HardNegativeManifest":
        payload: Any = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("hard-negative manifest must contain a mapping")
        return cls.model_validate(payload)

    def write(self, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output


__all__ = ["HardNegativeManifest", "HardNegativeRecord"]
