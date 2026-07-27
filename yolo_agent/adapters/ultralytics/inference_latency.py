"""Fixed-protocol checkpoint inference latency measurement."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.experiment_graph import MetricValue


class InferenceLatencyConfig(BaseModel):
    """Comparable single-image inference benchmark settings."""

    enabled: bool = False
    profiles: list[str] = Field(
        default_factory=lambda: [
            "pilot",
            "pilot_3",
            "pilot_10",
            "baseline_full",
            "baseline_confirm",
            "candidate_full",
            "candidate_full_seed_1",
            "candidate_full_confirmation",
        ]
    )
    imgsz: int = Field(default=640, ge=640, le=640)
    warmup_runs: int = Field(default=3, ge=0, le=100)
    timed_runs: int = Field(default=20, ge=1, le=1000)
    source: Literal["synthetic_rgb_zeros"] = "synthetic_rgb_zeros"


class InferenceLatencyResult(BaseModel):
    """Machine-readable result for one fixed inference protocol."""

    status: Literal["completed", "failed"]
    checkpoint: Path
    device: str
    imgsz: int
    warmup_runs: int
    timed_runs: int
    latency_ms: float | None = Field(default=None, ge=0.0)
    throughput: float | None = Field(default=None, ge=0.0)
    source: str = "synthetic_rgb_zeros"
    error: str | None = None

    @field_serializer("checkpoint")
    def serialize_checkpoint(self, value: Path) -> str:
        return value.as_posix()

    def to_metrics(self) -> dict[str, MetricValue]:
        if self.status != "completed" or self.latency_ms is None or self.throughput is None:
            return {"inference_latency_complete": False, "inference_latency_error": self.error or "unknown"}
        return {
            "inference_latency_complete": True,
            "latency_ms": self.latency_ms,
            "inference_throughput": self.throughput,
        }

    def to_json(self, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        return output


def should_run_inference_latency_benchmark(profile: str | None, config: InferenceLatencyConfig) -> bool:
    """Return whether a completed profile should collect latency evidence."""
    return bool(config.enabled and profile and profile in set(config.profiles))


def requires_fixed_inference_latency(profile: str | None, round_stage: str | None) -> bool:
    """Return whether an ASHA node is invalid without fixed latency evidence."""
    return bool(
        str(round_stage or "")
        in {"pilot_3", "pilot_10", "candidate_full_seed_1", "candidate_full_confirmation"}
        or profile == "candidate_full"
    )


def benchmark_checkpoint(
    checkpoint: Path | str,
    *,
    device: str,
    config: InferenceLatencyConfig,
) -> InferenceLatencyResult:
    """Measure checkpoint inference with a deterministic 640px synthetic image."""
    model_path = Path(checkpoint)
    result_kwargs: dict[str, Any] = {
        "checkpoint": model_path,
        "device": str(device),
        "imgsz": config.imgsz,
        "warmup_runs": config.warmup_runs,
        "timed_runs": config.timed_runs,
        "source": config.source,
    }
    if not model_path.is_file():
        return InferenceLatencyResult(status="failed", error=f"Checkpoint not found: {model_path}", **result_kwargs)
    try:
        import numpy as np

        model = _load_yolo(model_path)
        image = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
        for _ in range(config.warmup_runs):
            _predict(model, image, device=device, imgsz=config.imgsz)
        _synchronize_device(device)
        elapsed: list[float] = []
        for _ in range(config.timed_runs):
            started = time.perf_counter()
            _predict(model, image, device=device, imgsz=config.imgsz)
            _synchronize_device(device)
            elapsed.append(time.perf_counter() - started)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return InferenceLatencyResult(status="failed", error=str(exc), **result_kwargs)
    mean_seconds = sum(elapsed) / len(elapsed)
    latency_ms = round(mean_seconds * 1000.0, 6)
    return InferenceLatencyResult(
        status="completed",
        latency_ms=latency_ms,
        throughput=round(1.0 / mean_seconds, 6) if mean_seconds else 0.0,
        **result_kwargs,
    )


def _load_yolo(checkpoint: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(checkpoint)


def _predict(model: Any, image: Any, *, device: str, imgsz: int) -> None:
    model.predict(source=image, imgsz=imgsz, device=device, verbose=False)


def _synchronize_device(device: str) -> None:
    try:
        import torch

        if torch.cuda.is_available() and str(device).lower() not in {"cpu", "mps"}:
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        return
