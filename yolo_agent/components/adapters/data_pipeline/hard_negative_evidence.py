"""Production contracts for split-safe train-side hard-negative evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.data_pipeline.hard_negative import (
    HardNegativeManifest,
    HardNegativeRecord,
)


class TrainSampleIndexRecord(BaseModel):
    """One immutable image-to-index binding from the train dataset manifest."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    sample_index: int = Field(ge=0)
    image_path: str | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> "TrainSampleIndexRecord":
        if not self.image_id.strip():
            raise ValueError("train sample image_id must not be empty")
        return self


class TrainSampleIndex(BaseModel):
    """Exact train sample ordering bound to one dataset manifest hash."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "train_sample_index.v1"
    dataset_manifest_hash: str
    source_split: Literal["train"] = "train"
    samples: list[TrainSampleIndexRecord] = Field(min_length=1)
    index_hash: str = ""

    @model_validator(mode="after")
    def validate_index(self) -> "TrainSampleIndex":
        if not self.dataset_manifest_hash.strip():
            raise ValueError("train sample index requires dataset_manifest_hash")
        image_ids = [item.image_id for item in self.samples]
        sample_indices = [item.sample_index for item in self.samples]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("train sample index contains duplicate image IDs")
        if len(sample_indices) != len(set(sample_indices)):
            raise ValueError("train sample index contains duplicate sample indices")
        if set(sample_indices) != set(range(len(self.samples))):
            raise ValueError("train sample indices must be contiguous from zero")
        expected = self.compute_hash()
        if self.index_hash and self.index_hash != expected:
            raise ValueError("train sample index hash mismatch")
        self.index_hash = expected
        return self

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str | int, int],
        *,
        dataset_manifest_hash: str,
    ) -> "TrainSampleIndex":
        return cls(
            dataset_manifest_hash=dataset_manifest_hash,
            samples=sorted(
                (
                    TrainSampleIndexRecord(
                        image_id=str(image_id),
                        sample_index=int(sample_index),
                    )
                    for image_id, sample_index in mapping.items()
                ),
                key=lambda item: item.sample_index,
            ),
        )

    def compute_hash(self) -> str:
        return _semantic_hash(
            self.model_dump(mode="json", exclude={"index_hash"})
        )

    @property
    def image_to_sample_index(self) -> dict[str, int]:
        return {item.image_id: item.sample_index for item in self.samples}

    @property
    def valid_sample_indices(self) -> set[int]:
        return {item.sample_index for item in self.samples}

    def write(self, path: Path | str) -> Path:
        return _write_json_atomic(path, self.model_dump(mode="json"))

    @classmethod
    def from_path(cls, path: Path | str) -> "TrainSampleIndex":
        return cls.model_validate(_read_mapping(path))


class TrainHardNegativePrediction(BaseModel):
    """One train-side prediction eligible for hard-negative replay mining."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    predicted_class: int
    score: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)
    error_type: str = "background_false_positive"

    @model_validator(mode="after")
    def validate_prediction(self) -> "TrainHardNegativePrediction":
        if not self.image_id.strip():
            raise ValueError("train prediction image_id must not be empty")
        if not self.error_type.strip():
            raise ValueError("train prediction error_type must not be empty")
        return self


class TrainHardNegativePredictionBatch(BaseModel):
    """Predictions whose split and baseline provenance are explicit."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "train_hard_negative_predictions.v1"
    dataset_manifest_hash: str
    source_split: Literal["train"] = "train"
    source_run_id: str
    baseline_protocol_hash: str
    predictions: list[TrainHardNegativePrediction] = Field(default_factory=list)
    batch_hash: str = ""

    @model_validator(mode="after")
    def validate_batch(self) -> "TrainHardNegativePredictionBatch":
        for field_name in (
            "dataset_manifest_hash",
            "source_run_id",
            "baseline_protocol_hash",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"train prediction batch requires {field_name}")
        expected = self.compute_hash()
        if self.batch_hash and self.batch_hash != expected:
            raise ValueError("train prediction batch hash mismatch")
        self.batch_hash = expected
        return self

    def compute_hash(self) -> str:
        return _semantic_hash(
            self.model_dump(mode="json", exclude={"batch_hash"})
        )

    def write(self, path: Path | str) -> Path:
        return _write_json_atomic(path, self.model_dump(mode="json"))

    @classmethod
    def from_path(cls, path: Path | str) -> "TrainHardNegativePredictionBatch":
        return cls.model_validate(_read_mapping(path))


def produce_train_hard_negative_manifest(
    predictions: TrainHardNegativePredictionBatch | Path | str,
    train_index: TrainSampleIndex | Path | str,
    *,
    output_path: Path | str | None = None,
    score_threshold: float = 0.5,
) -> HardNegativeManifest:
    """Create replay evidence only from an exact train prediction/index pair."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("hard-negative score threshold must be between zero and one")
    prediction_batch = (
        predictions
        if isinstance(predictions, TrainHardNegativePredictionBatch)
        else TrainHardNegativePredictionBatch.from_path(predictions)
    )
    sample_index = (
        train_index
        if isinstance(train_index, TrainSampleIndex)
        else TrainSampleIndex.from_path(train_index)
    )
    if prediction_batch.dataset_manifest_hash != sample_index.dataset_manifest_hash:
        raise ValueError(
            "train predictions and sample index use different dataset manifest hashes"
        )

    mapping = sample_index.image_to_sample_index
    selected: dict[int, HardNegativeRecord] = {}
    unknown_image_ids: set[str] = set()
    for prediction in prediction_batch.predictions:
        if prediction.score < score_threshold:
            continue
        index = mapping.get(prediction.image_id)
        if index is None:
            unknown_image_ids.add(prediction.image_id)
            continue
        record = HardNegativeRecord(
            image_id=prediction.image_id,
            sample_index=index,
            predicted_class=prediction.predicted_class,
            score=prediction.score,
            bbox=prediction.bbox,
            error_type=prediction.error_type,
        )
        previous = selected.get(index)
        if previous is None or float(record.score or 0.0) > float(previous.score or 0.0):
            selected[index] = record
    if unknown_image_ids:
        raise ValueError(
            "train predictions reference images absent from the train dataset manifest: "
            + ", ".join(sorted(unknown_image_ids)[:10])
        )
    if not selected:
        raise ValueError("train hard-negative evidence contains no eligible predictions")

    manifest = HardNegativeManifest.from_records(
        dataset_manifest_hash=sample_index.dataset_manifest_hash,
        source_run_id=prediction_batch.source_run_id,
        baseline_protocol_hash=prediction_batch.baseline_protocol_hash,
        train_index_hash=sample_index.index_hash,
        prediction_artifact_sha256=prediction_batch.batch_hash,
        dataset_sample_count=len(sample_index.samples),
        records=sorted(selected.values(), key=lambda item: item.sample_index),
    )
    manifest.validate_runtime(
        dataset_manifest_hash=sample_index.dataset_manifest_hash,
        protocol_hash=prediction_batch.baseline_protocol_hash,
        dataset_length=len(sample_index.samples),
        valid_sample_indices=sample_index.valid_sample_indices,
        train_index_hash=sample_index.index_hash,
    )
    if output_path is not None:
        manifest.write(output_path)
    return manifest


def train_sample_index_from_records(
    records: Iterable[Any],
    *,
    dataset_manifest_hash: str,
) -> TrainSampleIndex:
    """Build the runtime index from dataset records without changing ordering."""
    samples = []
    for index, record in enumerate(records):
        split = str(getattr(record, "split", "train"))
        if split != "train":
            raise ValueError("train sample index cannot contain validation or test records")
        image_path = str(getattr(record, "image_path", ""))
        image_id = str(getattr(record, "image_id", "") or Path(image_path).stem)
        samples.append(
            TrainSampleIndexRecord(
                image_id=image_id,
                sample_index=index,
                image_path=image_path or None,
            )
        )
    return TrainSampleIndex(
        dataset_manifest_hash=dataset_manifest_hash,
        samples=samples,
    )


def _read_mapping(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"evidence artifact must contain a mapping: {path}")
    return value


def _write_json_atomic(path: Path | str, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def _semantic_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "TrainHardNegativePrediction",
    "TrainHardNegativePredictionBatch",
    "TrainSampleIndex",
    "TrainSampleIndexRecord",
    "produce_train_hard_negative_manifest",
    "train_sample_index_from_records",
]
