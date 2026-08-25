"""Schemas for real, file-backed assets required by paper execution.

Asset availability is deliberately separate from CPU readiness and ASHA
eligibility.  A row may contain a complete paper identity while still being
unavailable because the required files do not exist or do not match the
experiment protocol.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


AssetAvailability = Literal["available", "unavailable"]


_ASSET_FIELDS = (
    "source_dataset_manifest",
    "target_dataset_manifest",
    "teacher_checkpoint",
    "hard_negative_manifest",
    "graph_config",
    "matched_baseline_artifact",
)


class PaperAssetRecord(BaseModel, YAMLModelMixin):
    """One paper's actual filesystem assets and their verification outcome."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_asset_record.v1"
    paper_id: str
    mechanism_id: str
    source_dataset_manifest: str | None = None
    target_dataset_manifest: str | None = None
    teacher_checkpoint: str | None = None
    teacher_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    hard_negative_manifest: str | None = None
    graph_config: str | None = None
    matched_baseline_artifact: str | None = None
    asset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    availability: AssetAvailability
    exact_blocker: str
    recovery_action: str
    current_disposition: str
    asset_hashes: dict[str, str] = Field(default_factory=dict)
    validated_assets: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_asset_record(self) -> "PaperAssetRecord":
        if not self.paper_id.strip():
            raise ValueError("paper asset record requires paper_id")
        if not self.mechanism_id.strip():
            raise ValueError("paper asset record requires mechanism_id")
        if not self.exact_blocker.strip() and self.availability == "unavailable":
            raise ValueError("unavailable asset record requires exact_blocker")
        if not self.recovery_action.strip():
            raise ValueError("paper asset record requires recovery_action")

        paths: dict[str, Path] = {}
        for field_name in _ASSET_FIELDS:
            value = getattr(self, field_name)
            if value is None:
                continue
            path = Path(value)
            if not path.is_absolute():
                raise ValueError(f"{field_name} must be an absolute path")
            if not path.is_file():
                raise ValueError(f"{field_name} does not exist: {value}")
            paths[field_name] = path

        source = paths.get("source_dataset_manifest")
        target = paths.get("target_dataset_manifest")
        if source is not None and target is not None:
            if source.resolve() == target.resolve():
                raise ValueError("source and target dataset manifests must differ")

        expected_hashes = {
            name: _sha256(path)
            for name, path in paths.items()
        }
        if dict(sorted(self.asset_hashes.items())) != dict(sorted(expected_hashes.items())):
            raise ValueError("asset_hashes do not match the referenced files")
        if self.teacher_checkpoint:
            if self.teacher_sha256 != expected_hashes["teacher_checkpoint"]:
                raise ValueError("teacher_sha256 does not match teacher_checkpoint")
        elif self.teacher_sha256 is not None:
            raise ValueError("teacher_sha256 requires teacher_checkpoint")

        if self.asset_sha256 is not None:
            aggregate = _aggregate_hash(expected_hashes)
            if self.asset_sha256 != aggregate:
                raise ValueError("asset_sha256 does not match the referenced files")
        if self.availability == "available" and self.exact_blocker.strip():
            raise ValueError("available asset record cannot retain a blocker")
        if self.availability == "available" and self.asset_sha256 is None:
            raise ValueError("available asset record requires asset_sha256")
        return self


class PaperAssetRegistry(BaseModel, YAMLModelMixin):
    """The complete real-asset denominator for compatible papers."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_asset_registry.v1"
    source_inventory_path: str
    source_inventory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_requirements_path: str
    source_requirements_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatible_paper_count: int = Field(ge=0)
    records: list[PaperAssetRecord] = Field(default_factory=list)
    registry_hash: str = ""
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_registry(self) -> "PaperAssetRegistry":
        for field_name in ("source_inventory_path", "source_requirements_path"):
            path = Path(getattr(self, field_name))
            if not path.is_absolute() or not path.is_file():
                raise ValueError(
                    f"{field_name} must be an absolute existing file"
                )
        ids = [record.paper_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("paper asset registry contains duplicate paper IDs")
        if ids != sorted(ids):
            raise ValueError("paper asset registry records must be sorted by paper_id")
        if len(ids) != self.compatible_paper_count:
            raise ValueError("asset registry must contain every compatible paper")
        if self.registry_hash and self.registry_hash != self.calculate_hash():
            raise ValueError("paper asset registry hash mismatch")
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"registry_hash", "generated_at"},
        )
        for record in payload["records"]:
            record.pop("generated_at", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def with_hash(self) -> "PaperAssetRegistry":
        return self.model_copy(update={"registry_hash": self.calculate_hash()})

    @property
    def availability_counts(self) -> dict[str, int]:
        return {
            state: sum(record.availability == state for record in self.records)
            for state in ("available", "unavailable")
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_hash(asset_hashes: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(asset_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "AssetAvailability",
    "PaperAssetRecord",
    "PaperAssetRegistry",
]
