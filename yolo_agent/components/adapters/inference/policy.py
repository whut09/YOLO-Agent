"""Typed contracts for isolated inference policy execution."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


InferencePolicyKind = Literal[
    "sahi_slicing",
    "tiled_multi_scale",
    "test_time_augmentation",
    "confidence_calibration",
    "class_aware_thresholding",
    "merge_policy",
]
InferenceMetricNamespace = Literal[
    "sliced_inference",
    "tiled_multi_scale_inference",
    "tta_inference",
    "calibrated_inference",
    "class_threshold_inference",
    "merged_inference",
]
MergePolicyName = Literal["none", "nms", "nmm", "weighted_box_fusion"]


_NAMESPACE_BY_KIND: dict[str, InferenceMetricNamespace] = {
    "sahi_slicing": "sliced_inference",
    "tiled_multi_scale": "tiled_multi_scale_inference",
    "test_time_augmentation": "tta_inference",
    "confidence_calibration": "calibrated_inference",
    "class_aware_thresholding": "class_threshold_inference",
    "merge_policy": "merged_inference",
}


class InferencePolicyConfig(BaseModel):
    """One inference-only policy with a fixed standard comparison protocol."""

    policy_id: str
    kind: InferencePolicyKind
    model_path: str | None = None
    device: str = "cpu"
    standard_imgsz: Literal[640] = 640
    confidence_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0)
    class_thresholds: dict[int, float] = Field(default_factory=dict)
    scales: list[float] = Field(default_factory=lambda: [1.0])
    horizontal_flip: bool = False
    tile_sizes: list[int] = Field(default_factory=lambda: [640])
    overlap_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    merge_policy: MergePolicyName = "none"
    merge_iou_threshold: float = Field(default=0.55, gt=0.0, le=1.0)
    one_to_one_head: bool = True
    allow_cross_view_merge: bool = False
    max_detections: int = Field(default=300, ge=1)

    @model_validator(mode="after")
    def _validate_policy(self) -> "InferencePolicyConfig":
        if not self.policy_id.strip():
            raise ValueError("inference policy requires policy_id")
        if not self.scales or any(value <= 0 for value in self.scales):
            raise ValueError("inference scales must be positive")
        if not self.tile_sizes or any(value < 32 for value in self.tile_sizes):
            raise ValueError("tile sizes must be at least 32 pixels")
        if any(not 0.0 <= value <= 1.0 for value in self.class_thresholds.values()):
            raise ValueError("class thresholds must be in [0, 1]")
        if self.kind == "confidence_calibration" and self.temperature == 1.0:
            raise ValueError("confidence calibration requires a non-neutral temperature")
        if self.kind == "class_aware_thresholding" and not self.class_thresholds:
            raise ValueError("class-aware thresholding requires class_thresholds")
        if (
            self.kind == "merge_policy"
            and len(self.scales) < 2
            and not self.horizontal_flip
        ):
            raise ValueError("merge policy requires at least two fixed inference views")
        if self.one_to_one_head and self.merge_policy != "none" and not self.allow_cross_view_merge:
            raise ValueError(
                "YOLO26 one-to-one requires explicit allow_cross_view_merge before adding merge"
            )
        return self

    @property
    def metric_namespace(self) -> InferenceMetricNamespace:
        return _NAMESPACE_BY_KIND[self.kind]


class InferencePolicyProtocol(BaseModel):
    """Frozen protocol persisted beside every inference policy result."""

    schema_version: str = "inference_policy_protocol.v1"
    config: InferencePolicyConfig
    metric_namespace: InferenceMetricNamespace
    inference_policy_changed: Literal[True] = True
    training_attribution_allowed: Literal[False] = False
    standard_metrics_preserved: Literal[True] = True
    extra_nms_applied: bool = False

    @model_validator(mode="after")
    def _validate_namespace(self) -> "InferencePolicyProtocol":
        if self.metric_namespace != self.config.metric_namespace:
            raise ValueError("inference metric namespace does not match policy kind")
        if self.extra_nms_applied and self.config.merge_policy != "nms":
            raise ValueError("extra_nms_applied requires merge_policy=nms")
        return self

    @property
    def protocol_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def write(self, path: Path | str) -> Path:
        return _atomic_json(Path(path), self.model_dump(mode="json"))


class InferenceResourceMetrics(BaseModel):
    latency_ms: float = Field(ge=0.0)
    throughput: float = Field(ge=0.0)
    peak_vram_mb: float = Field(default=0.0, ge=0.0)


class InferencePolicyMetrics(BaseModel):
    """Metrics kept in a policy-specific namespace, never standard metric keys."""

    metric_namespace: InferenceMetricNamespace
    map50_95: float | None = None
    ap_small: float | None = None
    recall: float | None = None
    resources: InferenceResourceMetrics
    inference_policy_changed: Literal[True] = True

    def namespaced(self) -> dict[str, float | bool]:
        prefix = self.metric_namespace.removesuffix("_inference")
        values: dict[str, float | bool] = {
            f"{prefix}_latency_ms": self.resources.latency_ms,
            f"{prefix}_throughput": self.resources.throughput,
            f"{prefix}_peak_vram_mb": self.resources.peak_vram_mb,
            "inference_policy_changed": True,
        }
        for name in ("map50_95", "ap_small", "recall"):
            value = getattr(self, name)
            if value is not None:
                values[f"{prefix}_{name}"] = value
        return values


class InferencePolicyResult(BaseModel):
    status: Literal["completed", "failed", "skipped"]
    protocol: InferencePolicyProtocol
    metrics: InferencePolicyMetrics | None = None
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    merge_statistics: dict[str, int | float | str | bool] = Field(default_factory=dict)
    reason: str | None = None

    @model_validator(mode="after")
    def _completed_has_metrics(self) -> "InferencePolicyResult":
        if self.status == "completed" and self.metrics is None:
            raise ValueError("completed inference policy requires metrics")
        return self


def protocol_from_policy(config: InferencePolicyConfig) -> InferencePolicyProtocol:
    return InferencePolicyProtocol(
        config=config,
        metric_namespace=config.metric_namespace,
        extra_nms_applied=config.merge_policy == "nms",
    )


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, default=str)
            file.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


__all__ = [
    "InferenceMetricNamespace",
    "InferencePolicyConfig",
    "InferencePolicyKind",
    "InferencePolicyMetrics",
    "InferencePolicyProtocol",
    "InferencePolicyResult",
    "InferenceResourceMetrics",
    "MergePolicyName",
    "protocol_from_policy",
]
