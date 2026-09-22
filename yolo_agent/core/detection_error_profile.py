"""Detection error profiles for closed-loop optimization (Prompt-18I).

A :class:`DetectionErrorProfile` is the complete, auditable error
description of ONE evaluation (baseline or candidate) computed from real
ground truth and real predictions — never from narrative or a mAP drop
alone.  It carries nine sections:

``global``        map50, map50_95, precision, recall;
``scale``         AP by COCO area bucket;
``per_class``     AP, precision, recall, support;
``false_negative`` total + by class / scale / confidence / scene slice;
``false_positive`` total + by class, background/duplicate/class-confusion/
                  high-confidence splits;
``localization``  matched-IoU distribution, error count, AP50-vs-AP75 gap;
``classification`` confusion matrix + top confusion pairs;
``confidence``    TP/FP histograms + calibration summary;
``scene_slices``  optional metadata-driven slices (empty when the dataset
                  carries no such metadata — never fabricated).

The profile is deterministic in its inputs: the same GT + predictions +
metadata produce byte-identical content, which is what the per-round
``artifacts/error_profile.yaml`` artifact and the evidence-completeness
gate rely on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROFILE_SCHEMA_VERSION = "detection_error_profile.v1"

AreaBucket = Literal["small", "medium", "large"]


class ResourceSnapshot(BaseModel):
    """Runtime resource measurements taken from one evaluation node's run.

    Values come from the run's own resource manifest (latency audit,
    parameter count, FLOPs audit, peak memory) — never estimated by the
    agent.  A profile without a snapshot simply omits the resources section
    from its delta instead of fabricating numbers.
    """

    model_config = ConfigDict(extra="forbid")

    latency_ms: float | None = None
    params: int | None = None
    flops_g: float | None = None
    peak_memory_mb: float | None = None


class GlobalMetrics(BaseModel):
    """Headline quality metrics for one evaluation."""

    model_config = ConfigDict(extra="forbid")

    map50: float | None = None
    map50_95: float | None = None
    precision: float | None = None
    recall: float | None = None


class ScaleMetrics(BaseModel):
    """AP and recall by COCO area bucket (small/medium/large)."""

    model_config = ConfigDict(extra="forbid")

    ap_small: float | None = None
    ap_medium: float | None = None
    ap_large: float | None = None
    recall_small: float | None = None
    recall_medium: float | None = None
    recall_large: float | None = None


class PerClassMetrics(BaseModel):
    """Per-class quality with support."""

    model_config = ConfigDict(extra="forbid")

    category_id: int
    name: str
    ap: float | None = None
    ap50: float | None = None
    precision: float | None = None
    recall: float | None = None
    support: int = 0


class FalseNegativeFacts(BaseModel):
    """Missed ground truth, decomposed."""

    model_config = ConfigDict(extra="forbid")

    total: int = 0
    by_class: dict[str, int] = Field(default_factory=dict)
    by_scale: dict[str, int] = Field(default_factory=dict)
    by_confidence: dict[str, int] = Field(default_factory=dict)
    by_scene_slice: dict[str, int] = Field(default_factory=dict)


class FalsePositiveFacts(BaseModel):
    """False positives, decomposed into auditable kinds."""

    model_config = ConfigDict(extra="forbid")

    total: int = 0
    by_class: dict[str, int] = Field(default_factory=dict)
    background_fp: int = 0
    duplicate_fp: int = 0
    class_confusion_fp: int = 0
    high_confidence_fp: int = 0


class LocalizationFacts(BaseModel):
    """Where matched boxes sit on the IoU spectrum."""

    model_config = ConfigDict(extra="forbid")

    matched_iou_distribution: dict[str, int] = Field(default_factory=dict)
    mean_matched_iou: float | None = None
    localization_error_count: int = 0
    ap50_vs_ap75_gap: float | None = None


class ClassificationFacts(BaseModel):
    """Cross-class confusion on the validation split."""

    model_config = ConfigDict(extra="forbid")

    confusion_matrix: dict[str, int] = Field(default_factory=dict)
    top_confusion_pairs: list[tuple[str, int]] = Field(default_factory=list)


class ConfidenceFacts(BaseModel):
    """Score behaviour of TP vs FP populations and calibration."""

    model_config = ConfigDict(extra="forbid")

    tp_confidence_histogram: dict[str, int] = Field(default_factory=dict)
    fp_confidence_histogram: dict[str, int] = Field(default_factory=dict)
    calibration_bins: list[dict[str, float]] = Field(default_factory=list)
    expected_calibration_error: float | None = None


class SceneSliceFacts(BaseModel):
    """One metadata slice's quality, when the dataset provides metadata."""

    model_config = ConfigDict(extra="forbid")

    slice_name: str
    gt_count: int = 0
    true_positives: int = 0
    recall: float | None = None
    ap50: float | None = None


class DetectionErrorProfile(BaseModel):
    """The complete error profile of one evaluation node."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PROFILE_SCHEMA_VERSION
    profile_id: str
    run_id: str
    candidate_id: str
    node_id: str = ""
    split: Literal["val", "test", "train"] = "val"
    role: Literal["baseline", "candidate"] = "candidate"
    protocol_hash: str = ""

    gt_artifact: str = ""
    predictions_artifact: str = ""
    dataset_manifest_hash: str = ""

    global_: GlobalMetrics = Field(default_factory=GlobalMetrics, alias="global")
    scale: ScaleMetrics = Field(default_factory=ScaleMetrics)
    per_class: list[PerClassMetrics] = Field(default_factory=list)
    false_negative: FalseNegativeFacts = Field(default_factory=FalseNegativeFacts)
    false_positive: FalsePositiveFacts = Field(default_factory=FalsePositiveFacts)
    localization: LocalizationFacts = Field(default_factory=LocalizationFacts)
    classification: ClassificationFacts = Field(default_factory=ClassificationFacts)
    confidence: ConfidenceFacts = Field(default_factory=ConfidenceFacts)
    scene_slices: list[SceneSliceFacts] = Field(default_factory=list)
    #: Scene-slice metadata the dataset did NOT provide — recorded so the
    #: absence is auditable instead of silently fabricated.
    unavailable_scene_slices: list[str] = Field(default_factory=list)
    #: Real runtime resource measurements from this node's own run (from the
    #: runtime resource manifest, not agent estimates).  Optional: a profile
    #: without resource telemetry simply drops the resources delta section.
    resources: ResourceSnapshot | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("profile_id", "run_id", "candidate_id")
    @classmethod
    def _identity_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("profile_id, run_id, and candidate_id must not be empty")
        return value

    def to_yaml_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload["schema_version"] = self.schema_version
        return payload


class ErrorProfileSource(BaseModel):
    """Provenance for building a profile from real artifacts."""

    model_config = ConfigDict(extra="forbid")

    gt_json: Path
    predictions_json: Path
    image_metadata_json: Path | None = None
    official_metrics_json: Path | None = None
    run_id: str = ""
    candidate_id: str = ""
    node_id: str = ""
    split: Literal["val", "test", "train"] = "val"
    role: Literal["baseline", "candidate"] = "candidate"
    protocol_hash: str = ""
    dataset_manifest_hash: str = ""
