"""Split-safe local hard-negative manifest used by replay sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal

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
    baseline_checkpoint_hash: str | None = None
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
        if self.baseline_checkpoint_hash is not None and not self.baseline_checkpoint_hash.strip():
            raise ValueError("hard-negative manifest baseline_checkpoint_hash must not be empty")
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
        baseline_checkpoint_hash: str | None = None,
        require_provenance: bool = False,
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
        if require_provenance:
            missing = [
                name
                for name, value in (
                    ("train_index_hash", self.train_index_hash),
                    ("baseline_checkpoint_hash", self.baseline_checkpoint_hash),
                )
                if not str(value or "").strip()
            ]
            if missing:
                raise ValueError(
                    "hard-negative manifest provenance is incomplete: "
                    + ", ".join(missing)
                )
        if baseline_checkpoint_hash is not None and self.baseline_checkpoint_hash != baseline_checkpoint_hash:
            raise ValueError(
                "hard-negative manifest baseline checkpoint hash does not match runtime"
            )

    @property
    def provenance_complete(self) -> bool:
        """Whether the manifest can authorize production replay."""
        return bool(
            self.dataset_manifest_hash.strip()
            and self.source_split == "train"
            and self.source_run_id.strip()
            and self.baseline_protocol_hash.strip()
            and self.baseline_checkpoint_hash
            and self.train_index_hash
            and self.records
            and self.manifest_hash
        )

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
        baseline_checkpoint_hash: str | None = None,
        train_index_hash: str | None = None,
        prediction_artifact_sha256: str | None = None,
        dataset_sample_count: int | None = None,
    ) -> "HardNegativeManifest":
        return cls(
            dataset_manifest_hash=dataset_manifest_hash,
            source_split="train",
            source_run_id=source_run_id,
            baseline_protocol_hash=baseline_protocol_hash,
            baseline_checkpoint_hash=baseline_checkpoint_hash,
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


HardNegativeBootstrapStageName = Literal[
    "baseline_train",
    "train_split_inference",
    "hard_negative_manifest",
    "hard_negative_candidate",
]
HardNegativeBootstrapStageStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
]


class HardNegativeBootstrapStage(BaseModel):
    """One ordered, auditable stage in train-side replay evidence recovery."""

    model_config = ConfigDict(extra="forbid")

    stage: HardNegativeBootstrapStageName
    status: HardNegativeBootstrapStageStatus = "pending"
    node_id: str | None = None
    artifact_path: Path | None = None
    reason_codes: list[str] = Field(default_factory=list)


class HardNegativeEvidenceBootstrap(BaseModel):
    """Persistent state machine for automatic hard-negative evidence production.

    This object describes work that must be scheduled.  It never manufactures a
    prediction, manifest, checkpoint, or metric result.  A replay candidate can
    only become active after all four ordered stages have completed with real
    artifacts.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "hard_negative_evidence_bootstrap.v1"
    candidate_id: str
    dependent_component_id: Literal["sampling.hard_negative_replay"] = (
        "sampling.hard_negative_replay"
    )
    source_run_id: str
    source_split: Literal["train"] = "train"
    dataset_manifest_hash: str | None = None
    train_index_hash: str | None = None
    baseline_protocol_hash: str | None = None
    baseline_checkpoint_hash: str | None = None
    manifest_path: Path | None = None
    manifest_hash: str | None = None
    stages: list[HardNegativeBootstrapStage] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    execution_fingerprint: str | None = None

    @model_validator(mode="after")
    def validate_bootstrap(self) -> "HardNegativeEvidenceBootstrap":
        for field_name in (
            "candidate_id",
            "source_run_id",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"hard-negative bootstrap requires {field_name}")
        if self.source_split != "train":
            raise ValueError("hard-negative bootstrap only supports the train split")
        expected = [
            "baseline_train",
            "train_split_inference",
            "hard_negative_manifest",
            "hard_negative_candidate",
        ]
        actual = [item.stage for item in self.stages]
        if actual and actual != expected[: len(actual)]:
            raise ValueError("hard-negative bootstrap stages must follow the train evidence order")
        return self

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        source_run_id: str,
        dataset_manifest_hash: str | None,
        baseline_protocol_hash: str | None,
        dependent_component_id: str = "sampling.hard_negative_replay",
        execution_fingerprint: str | None = None,
    ) -> "HardNegativeEvidenceBootstrap":
        if dependent_component_id != "sampling.hard_negative_replay":
            raise ValueError(
                "hard-negative evidence bootstrap is only a dependency of "
                "sampling.hard_negative_replay"
            )
        return cls(
            candidate_id=candidate_id,
            source_run_id=source_run_id,
            dataset_manifest_hash=dataset_manifest_hash,
            baseline_protocol_hash=baseline_protocol_hash,
            execution_fingerprint=execution_fingerprint,
            recovery_actions=["recover_train_hard_negative_evidence"],
            reason_codes=["hard_negative_manifest_missing"],
            stages=[
                HardNegativeBootstrapStage(stage=stage)
                for stage in (
                    "baseline_train",
                    "train_split_inference",
                    "hard_negative_manifest",
                    "hard_negative_candidate",
                )
            ],
        )

    def stage(self, stage: HardNegativeBootstrapStageName) -> HardNegativeBootstrapStage:
        """Return or lazily create one ordered stage."""
        for item in self.stages:
            if item.stage == stage:
                return item
        order = [
            "baseline_train",
            "train_split_inference",
            "hard_negative_manifest",
            "hard_negative_candidate",
        ]
        index = order.index(stage)
        if len(self.stages) < index:
            raise ValueError(
                f"cannot materialize {stage} before {order[len(self.stages)]}"
            )
        item = HardNegativeBootstrapStage(stage=stage)
        self.stages.append(item)
        return item

    def advance(
        self,
        stage: HardNegativeBootstrapStageName,
        *,
        status: HardNegativeBootstrapStageStatus = "completed",
        node_id: str | None = None,
        artifact_path: Path | str | None = None,
        reason_codes: Iterable[str] = (),
    ) -> "HardNegativeEvidenceBootstrap":
        """Advance one stage without skipping an evidence dependency."""
        order = [
            "baseline_train",
            "train_split_inference",
            "hard_negative_manifest",
            "hard_negative_candidate",
        ]
        index = order.index(stage)
        for prior in self.stages[:index]:
            if prior.status != "completed":
                raise ValueError(
                    f"cannot advance {stage} before {prior.stage} is completed"
                )
        item = self.stage(stage)  # validates append order
        item.status = status
        item.node_id = node_id or item.node_id
        item.artifact_path = (
            Path(artifact_path).resolve() if artifact_path is not None else item.artifact_path
        )
        item.reason_codes = list(dict.fromkeys(str(code) for code in reason_codes if str(code).strip()))
        if status == "failed" and not item.reason_codes:
            item.reason_codes = [f"{stage}_failed"]
        return self

    @classmethod
    def from_path(cls, path: Path | str) -> "HardNegativeEvidenceBootstrap":
        """Load a persisted bootstrap state without creating evidence."""
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("hard-negative bootstrap must contain a mapping")
        return cls.model_validate(value)

    def write(self, path: Path | str) -> Path:
        """Persist bootstrap state atomically as an auditable artifact."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        temporary.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(output)
        return output

    @property
    def next_stage(self) -> HardNegativeBootstrapStageName:
        """Return the next stage that requires scheduling."""
        for stage in (
            "baseline_train",
            "train_split_inference",
            "hard_negative_manifest",
            "hard_negative_candidate",
        ):
            item = next((entry for entry in self.stages if entry.stage == stage), None)
            if item is None or item.status != "completed":
                return stage
        return "hard_negative_candidate"

    @property
    def candidate_activation_allowed(self) -> bool:
        """Whether the replay candidate may be re-materialized as executable."""
        return bool(
            self.manifest_path
            and self.manifest_hash
            and self.dataset_manifest_hash
            and self.baseline_protocol_hash
            and self.baseline_checkpoint_hash
            and self.train_index_hash
            and all(
                any(
                    item.stage == stage and item.status == "completed"
                    for item in self.stages
                )
                for stage in (
                    "baseline_train",
                    "train_split_inference",
                    "hard_negative_manifest",
                )
            )
        )

    def bind_manifest(
        self,
        manifest: HardNegativeManifest,
        *,
        path: Path | str,
        baseline_checkpoint_hash: str,
    ) -> "HardNegativeEvidenceBootstrap":
        """Bind a validated production manifest and complete the evidence stage."""
        if manifest.source_split != "train":
            raise ValueError("hard-negative bootstrap manifest must use source_split=train")
        if not self.dataset_manifest_hash or not self.baseline_protocol_hash:
            raise ValueError(
                "hard-negative bootstrap is missing dataset or baseline protocol identity"
            )
        dataset_length = manifest.dataset_sample_count
        if dataset_length is None:
            dataset_length = max(manifest.sample_indices, default=-1) + 1
        manifest.validate_runtime(
            dataset_manifest_hash=self.dataset_manifest_hash,
            protocol_hash=self.baseline_protocol_hash,
            dataset_length=dataset_length,
            train_index_hash=self.train_index_hash,
            baseline_checkpoint_hash=baseline_checkpoint_hash,
            require_provenance=True,
        )
        if manifest.baseline_checkpoint_hash != baseline_checkpoint_hash:
            raise ValueError("hard-negative bootstrap baseline checkpoint hash mismatch")
        self.baseline_checkpoint_hash = baseline_checkpoint_hash
        self.train_index_hash = manifest.train_index_hash
        self.manifest_path = Path(path).resolve()
        self.manifest_hash = manifest.manifest_hash
        self.advance(
            "hard_negative_manifest",
            status="completed",
            artifact_path=self.manifest_path,
            reason_codes=["train_hard_negative_manifest_validated"],
        )
        return self


__all__ = [
    "HardNegativeBootstrapStage",
    "HardNegativeBootstrapStageName",
    "HardNegativeBootstrapStageStatus",
    "HardNegativeEvidenceBootstrap",
    "HardNegativeManifest",
    "HardNegativeRecord",
]
