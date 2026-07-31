"""Component-specific assertions layered on the common real GPU contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    SmallObjectSamplingManifest,
)


GPUProfileValidator = Callable[
    [AdapterRuntimePayload, dict[str, Path]],
    dict[str, bool | str | int | float],
]


def validate_component_gpu_profile(
    component_id: str,
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    """Validate runtime facts that are unique to one adapter mechanism."""
    validator = _VALIDATORS.get(component_id)
    if validator is None:
        raise ValueError(f"GPU profile validator is not implemented: {component_id}")
    checks = validator(payload, artifacts)
    if not checks or not all(value is True for value in checks.values()):
        failed = sorted(name for name, value in checks.items() if value is not True)
        raise ValueError("component GPU profile failed: " + ", ".join(failed))
    return checks


def _validate_sampling(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    manifest = _json_model(
        SmallObjectSamplingManifest,
        artifacts,
        "adapter_sampler_manifest",
    )
    maximum = float(manifest.clipping_statistics.get("max_weight", 0.0))
    return {
        "sampling_payload_bound": manifest.runtime_payload_hash == payload.payload_hash,
        "sampling_protocol_bound": manifest.protocol_hash == payload.protocol_hash,
        "sampling_train_split_only": manifest.split == "train",
        "sampling_val_unchanged": manifest.val_unchanged,
        "sampling_weights_complete": bool(
            manifest.image_count > 0
            and manifest.sample_count > 0
            and len(manifest.raw_weights) == manifest.image_count
            and len(manifest.final_weights) == manifest.image_count
        ),
        "sampling_weights_bounded": bool(
            maximum >= 1.0
            and manifest.final_weights
            and max(manifest.final_weights) <= maximum
        ),
        "sampling_adapter_hash_recorded": len(manifest.adapter_hash) == 64,
    }


def _json_model(
    model: type[Any],
    artifacts: dict[str, Path],
    key: str,
) -> Any:
    path = artifacts.get(key)
    if path is None or not path.is_file():
        raise ValueError(f"component GPU artifact missing: {key}")
    return model.model_validate_json(path.read_text(encoding="utf-8-sig"))


_VALIDATORS: dict[str, GPUProfileValidator] = {
    "sampling.small_object": _validate_sampling,
}


__all__ = ["GPUProfileValidator", "validate_component_gpu_profile"]
