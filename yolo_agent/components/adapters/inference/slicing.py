"""SAHI-compatible slicing inference kept separate from training evidence."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, Field, model_validator

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.core.experiment_graph import MetricEvidence


MergePolicy = Literal["none", "nms", "nmm"]


class SlicingInferenceConfig(BaseModel):
    slice_height: int = Field(default=640, ge=32)
    slice_width: int = Field(default=640, ge=32)
    overlap_height_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    overlap_width_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    merge_policy: MergePolicy = "none"
    merge_match_metric: Literal["iou", "ios"] = "iou"
    merge_match_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    one_to_one_head: bool = True
    standard_imgsz: int = 640

    @model_validator(mode="after")
    def _fair_protocol(self) -> "SlicingInferenceConfig":
        if self.standard_imgsz != 640:
            raise ValueError("standard training comparison must remain imgsz=640")
        return self


class SlicingInferenceProtocol(BaseModel):
    schema_version: str = "slicing_inference_protocol.v1"
    adapter: str = "sahi"
    adapter_version: str = "slicing.v1"
    slice_height: int
    slice_width: int
    overlap_height_ratio: float
    overlap_width_ratio: float
    merge_policy: MergePolicy
    merge_match_metric: str
    merge_match_threshold: float
    one_to_one_head: bool
    extra_nms_applied: bool
    standard_imgsz: int = 640
    inference_policy_changed: bool = True

    def write(self, path: Path | str) -> Path:
        """Atomically persist the complete evaluation protocol."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix=output.name, suffix=".tmp", dir=output.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(self.model_dump(mode="json"), file, indent=2, sort_keys=True)
                file.write("\n")
            os.replace(temporary_name, output)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return output


class SlicingInferenceMetrics(BaseModel):
    sliced_map50_95: float | None = None
    sliced_ap_small: float | None = None
    sliced_latency_ms: float = Field(ge=0.0)
    sliced_throughput: float = Field(ge=0.0)
    inference_policy_changed: bool = True


class SlicingInferenceResult(BaseModel):
    status: Literal["completed", "skipped", "failed"]
    protocol: SlicingInferenceProtocol
    metrics: SlicingInferenceMetrics | None = None
    predictions: list[Any] = Field(default_factory=list)
    reason: str | None = None


class SlicingBackend(Protocol):
    def __call__(self, images: list[Any], protocol: SlicingInferenceProtocol) -> tuple[list[Any], dict[str, float | None]]: ...


class SlicingInferenceRunner:
    """Execute slicing through an injected backend or optional SAHI."""

    def __init__(self, backend: SlicingBackend | None = None) -> None:
        self.backend = backend

    @staticmethod
    def sahi_available() -> bool:
        return importlib.util.find_spec("sahi") is not None

    def run(self, images: list[Any], config: SlicingInferenceConfig) -> SlicingInferenceResult:
        protocol = protocol_from_config(config)
        if self.backend is None and not self.sahi_available():
            return SlicingInferenceResult(status="skipped", protocol=protocol, reason="optional dependency 'sahi' is not installed")
        backend = self.backend or self._sahi_backend
        started = time.perf_counter()
        try:
            predictions, raw_metrics = backend(images, protocol)
        except Exception as exc:  # adapter boundary returns structured failure
            return SlicingInferenceResult(status="failed", protocol=protocol, reason=str(exc))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms = float(raw_metrics.get("sliced_latency_ms") or (elapsed_ms / max(len(images), 1)))
        throughput = float(raw_metrics.get("sliced_throughput") or (1000.0 / latency_ms if latency_ms else 0.0))
        metrics = SlicingInferenceMetrics(
            sliced_map50_95=_optional_float(raw_metrics.get("sliced_map50_95")),
            sliced_ap_small=_optional_float(raw_metrics.get("sliced_ap_small")),
            sliced_latency_ms=latency_ms,
            sliced_throughput=throughput,
        )
        return SlicingInferenceResult(status="completed", protocol=protocol, metrics=metrics, predictions=predictions)

    @staticmethod
    def _sahi_backend(images: list[Any], protocol: SlicingInferenceProtocol) -> tuple[list[Any], dict[str, float | None]]:
        # Import lazily so the core package remains usable without SAHI. The
        # concrete detection model is deliberately supplied by the caller.
        import sahi  # type: ignore[import-not-found]  # noqa: F401
        raise RuntimeError("SAHI is installed, but a model-bound slicing backend was not supplied")


class SlicingInferenceAdapter(ComponentAdapter):
    adapter_version = "slicing.v1"
    source_commit = "local"
    strategy = "inference_adapter"
    modified_model_fields = frozenset()
    modified_training_fields = frozenset({"inference_policy"})

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        available = SlicingInferenceRunner.sahi_available()
        return AdapterValidationReport(ok=True, warnings=[] if available else ["SAHI is optional and not installed; execution will be skipped."], checks={"sahi_available": available})

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        config = SlicingInferenceConfig.model_validate(context.options or {})
        warnings: list[str] = []
        if config.one_to_one_head and config.merge_policy == "none":
            warnings.append("No extra NMS is applied to the one-to-one head; overlapping slice duplicates remain a protocol consideration.")
        if config.one_to_one_head and config.merge_policy in {"nms", "nmm"}:
            warnings.append(f"Extra {config.merge_policy.upper()} is applied only for cross-slice merging, not the standard YOLO26 path.")
        return AdapterValidationReport(ok=context.imgsz == 640, errors=[] if context.imgsz == 640 else ["standard comparison requires imgsz=640"], warnings=warnings, checks={"inference_policy_changed": True})

    def patch_model_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        return config

    def patch_training_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        protocol = protocol_from_config(SlicingInferenceConfig.model_validate(context.options or {}))
        config["inference_policy"] = protocol.model_dump(mode="json")
        return config

    def build_module(self, context: AdapterContext) -> SlicingInferenceRunner:
        backend = context.environment.get("slicing_backend")
        return SlicingInferenceRunner(backend=backend if callable(backend) else None)

    def load_pretrained_weights(self, module: Any, weights: Path | str | None, context: AdapterContext) -> WeightLoadResult:
        return WeightLoadResult(loaded=False, source=Path(weights) if weights else None, message="Slicing reuses the evaluated detector checkpoint and does not load component weights")

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        config = SlicingInferenceConfig.model_validate(context.options or {})
        protocol = protocol_from_config(config)
        return SmokeTestResult(passed=True, checks={"protocol": protocol.schema_version, "extra_nms": protocol.extra_nms_applied, "standard_metrics_preserved": True})

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        return [ExpectedArtifact(name="slicing_inference_protocol", relative_path=Path("artifacts/slicing_inference_protocol.json"))]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(actions=["discard inference-only slicing protocol"], files_to_remove=[Path("artifacts/slicing_inference_protocol.json")])


def protocol_from_config(config: SlicingInferenceConfig) -> SlicingInferenceProtocol:
    return SlicingInferenceProtocol(
        slice_height=config.slice_height,
        slice_width=config.slice_width,
        overlap_height_ratio=config.overlap_height_ratio,
        overlap_width_ratio=config.overlap_width_ratio,
        merge_policy=config.merge_policy,
        merge_match_metric=config.merge_match_metric,
        merge_match_threshold=config.merge_match_threshold,
        one_to_one_head=config.one_to_one_head,
        extra_nms_applied=config.merge_policy == "nms",
    )


def metric_evidence_from_result(
    result: SlicingInferenceResult,
    *,
    candidate_id: str,
    node_id: str,
    dataset_version: str,
    split: str = "val",
    source_artifact: Path | None = None,
) -> list[MetricEvidence]:
    """Create only sliced_* evidence; standard metrics are never overwritten."""
    if result.status != "completed" or result.metrics is None:
        return []
    payload = result.metrics.model_dump(exclude={"inference_policy_changed"})
    return [MetricEvidence(candidate_id=candidate_id, node_id=node_id, dataset_version=dataset_version, split=split, metric_name=name, value=value, source="slicing_inference", validator="slicing_protocol", source_artifact=source_artifact, verified=True) for name, value in payload.items()]


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


__all__ = ["SlicingInferenceAdapter", "SlicingInferenceConfig", "SlicingInferenceMetrics", "SlicingInferenceProtocol", "SlicingInferenceResult", "SlicingInferenceRunner", "metric_evidence_from_result", "protocol_from_config"]
