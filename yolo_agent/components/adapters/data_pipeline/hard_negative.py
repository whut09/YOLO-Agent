"""Split-safe local hard-negative manifest used by replay sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HardNegativeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    sample_index: int = Field(ge=0)
    predicted_class: int | None = None
    score: float | None = None
    bbox: list[float] = Field(default_factory=list)
    error_type: str

    @model_validator(mode="after")
    def validate_record(self) -> "HardNegativeRecord":
        if not self.image_id.strip():
            raise ValueError("hard-negative image_id must not be empty")
        if len(self.bbox) not in {0, 4}:
            raise ValueError("hard-negative bbox must be empty or [x, y, w, h]")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("hard-negative score must be between 0 and 1")
        if not self.error_type.strip():
            raise ValueError("hard-negative error_type must not be empty")
        return self


class HardNegativeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "hard_negative_manifest.v1"
    dataset_manifest_hash: str
    source_split: str
    source_run_id: str
    baseline_protocol_hash: str
    train_index_hash: str | None = None
    prediction_artifact_sha256: str | None = None
    dataset_sample_count: int | None = Field(default=None, ge=1)
    records: list[HardNegativeRecord] = Field(default_factory=list)
    manifest_hash: str = ""

    @model_validator(mode="after")
    def validate_manifest(self) -> "HardNegativeManifest":
        if not self.dataset_manifest_hash.strip():
            raise ValueError("hard-negative manifest requires dataset_manifest_hash")
        if not self.source_run_id.strip():
            raise ValueError("hard-negative manifest requires source_run_id")
        if not self.baseline_protocol_hash.strip():
            raise ValueError("hard-negative manifest requires baseline_protocol_hash")
        if self.source_split != "train":
            raise ValueError("hard-negative replay requires a train split manifest")
        indices = [item.sample_index for item in self.records]
        if len(indices) != len(set(indices)):
            raise ValueError("hard-negative manifest contains duplicate sample indices")
        if self.dataset_sample_count is not None and any(
            index >= self.dataset_sample_count for index in indices
        ):
            raise ValueError(
                "hard-negative manifest sample index is outside the indexed train dataset"
            )
        expected = self.compute_hash()
        if self.manifest_hash and self.manifest_hash != expected:
            raise ValueError("hard-negative manifest hash mismatch")
        self.manifest_hash = expected
        return self

    def validate_runtime(
        self,
        *,
        dataset_manifest_hash: str,
        protocol_hash: str,
        dataset_length: int,
        split: str = "train",
        valid_sample_indices: set[int] | None = None,
        train_index_hash: str | None = None,
    ) -> None:
        """Validate the manifest against the exact train runtime contract."""
        if split != "train" or self.source_split != "train":
            raise ValueError("hard-negative replay requires a train split runtime")
        if self.dataset_manifest_hash != dataset_manifest_hash:
            raise ValueError("hard-negative manifest dataset hash does not match the train dataset")
        if self.baseline_protocol_hash != protocol_hash:
            raise ValueError("hard-negative manifest baseline protocol hash does not match runtime")
        if not self.records:
            raise ValueError("hard-negative replay requires a non-empty evidence manifest")
        if any(item.sample_index >= dataset_length for item in self.records):
            raise ValueError("hard-negative manifest sample index is outside the train dataset")
        if valid_sample_indices is not None and any(
            item.sample_index not in valid_sample_indices for item in self.records
        ):
            raise ValueError(
                "hard-negative manifest sample index is not present in the train dataset manifest"
            )
        if train_index_hash is not None and self.train_index_hash != train_index_hash:
            raise ValueError("hard-negative manifest train index hash does not match runtime")

    @property
    def evidence_id(self) -> str:
        """Stable evidence identity usable by atomic and coupled candidates."""
        return f"hard_negative_replay:{self.manifest_hash}"

    @classmethod
    def from_records(
        cls,
        *,
        dataset_manifest_hash: str,
        source_run_id: str,
        baseline_protocol_hash: str,
        records: Iterable[HardNegativeRecord],
        train_index_hash: str | None = None,
        prediction_artifact_sha256: str | None = None,
        dataset_sample_count: int | None = None,
    ) -> "HardNegativeManifest":
        return cls(
            dataset_manifest_hash=dataset_manifest_hash,
            source_split="train",
            source_run_id=source_run_id,
            baseline_protocol_hash=baseline_protocol_hash,
            train_index_hash=train_index_hash,
            prediction_artifact_sha256=prediction_artifact_sha256,
            dataset_sample_count=dataset_sample_count,
            records=list(records),
        )

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
        source = Path(path)
        if source.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
        else:
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
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
