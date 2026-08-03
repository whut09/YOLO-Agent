"""Atomic artifacts for isolated inference policy results."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from yolo_agent.components.adapters.inference.policy import InferencePolicyResult


class InferencePolicyArtifactPaths(BaseModel):
    protocol: Path
    predictions: Path
    metrics: Path
    resources: Path
    merge_statistics: Path


def write_inference_policy_artifacts(
    result: InferencePolicyResult,
    output_dir: Path,
) -> InferencePolicyArtifactPaths:
    """Persist completed inference-only evidence without standard metric keys."""
    if result.status != "completed" or result.metrics is None:
        raise ValueError("only completed inference policy results can be persisted")
    output_dir.mkdir(parents=True, exist_ok=True)
    namespace = result.protocol.metric_namespace
    prefix = namespace.removesuffix("_inference")
    protocol_path = result.protocol.write(output_dir / f"{prefix}_protocol.json")
    predictions_path = _atomic_json(
        output_dir / f"{prefix}_predictions.json", result.predictions
    )
    metrics_payload = result.metrics.namespaced()
    _reject_standard_metric_keys(metrics_payload)
    metrics_path = _atomic_json(output_dir / f"{prefix}_metrics.json", metrics_payload)
    resources_path = _atomic_json(
        output_dir / f"{prefix}_resources.json",
        result.metrics.resources.model_dump(mode="json"),
    )
    merge_path = _atomic_json(
        output_dir / f"{prefix}_merge_statistics.json", result.merge_statistics
    )
    return InferencePolicyArtifactPaths(
        protocol=protocol_path,
        predictions=predictions_path,
        metrics=metrics_path,
        resources=resources_path,
        merge_statistics=merge_path,
    )


def _reject_standard_metric_keys(payload: dict[str, Any]) -> None:
    forbidden = {
        "map50_95",
        "map50",
        "ap_small",
        "recall",
        "latency_ms",
        "throughput",
        "peak_vram_mb",
    }
    overlap = sorted(forbidden & set(payload))
    if overlap:
        raise ValueError(
            "inference policy artifacts cannot overwrite standard metrics: "
            + ", ".join(overlap)
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
    "InferencePolicyArtifactPaths",
    "write_inference_policy_artifacts",
]
