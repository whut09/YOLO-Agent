"""Atomic artifacts for isolated slicing inference."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from yolo_agent.components.adapters.inference.slicing import SlicingInferenceResult


class SlicingArtifactPaths(BaseModel):
    protocol: Path
    predictions: Path
    metrics: Path


def write_slicing_artifacts(
    result: SlicingInferenceResult,
    output_dir: Path,
) -> SlicingArtifactPaths:
    """Persist a completed result without writing standard metric names."""
    if result.status != "completed" or result.metrics is None:
        raise ValueError("only completed slicing results can be persisted")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = result.protocol.write(output_dir / "slicing_inference_protocol.json")
    predictions_path = _atomic_json(output_dir / "sliced_predictions.json", result.predictions)
    metrics_payload = result.metrics.model_dump(mode="json")
    if any(name in metrics_payload for name in ("map50_95", "ap_small", "latency_ms")):
        raise ValueError("sliced artifacts cannot overwrite standard metric names")
    metrics_path = _atomic_json(output_dir / "sliced_metrics.json", metrics_payload)
    return SlicingArtifactPaths(
        protocol=protocol_path,
        predictions=predictions_path,
        metrics=metrics_path,
    )


def _atomic_json(path: Path, payload: Any) -> Path:
    handle, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, default=str)
            file.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


__all__ = ["SlicingArtifactPaths", "write_slicing_artifacts"]
