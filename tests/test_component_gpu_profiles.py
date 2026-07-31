from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.component_gpu_profiles import (
    validate_component_gpu_profile,
)
from yolo_agent.components.adapters import AdapterContext, AdapterRuntimePayload
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    SmallObjectSamplingManifest,
)
from yolo_agent.components.contracts import load_contracts


def _sampling_payload(tmp_path: Path) -> AdapterRuntimePayload:
    contract = load_contracts(
        "configs/components/sampling/small_object_sampling.yaml"
    )[0]
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    payload = adapter.build_runtime_payload(
        AdapterContext(contract=contract, workspace=tmp_path),
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    assert payload is not None
    return payload


def test_sampling_gpu_profile_requires_bound_train_only_manifest(
    tmp_path: Path,
) -> None:
    payload = _sampling_payload(tmp_path)
    manifest = SmallObjectSamplingManifest(
        dataset_manifest="fixture-hash",
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        split="train",
        seed=17,
        area_thresholds={"small": 0.01},
        image_count=2,
        small_image_count=1,
        raw_weights=[2.0, 1.0],
        final_weights=[2.0, 1.0],
        image_paths=["a.png", "b.png"],
        clipping_statistics={"max_weight": 3.0},
        sample_count=2,
        adapter_hash="a" * 64,
        val_unchanged=True,
    )
    path = tmp_path / "sampler_manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    checks = validate_component_gpu_profile(
        "sampling.small_object",
        payload,
        {"adapter_sampler_manifest": path},
    )

    assert all(value is True for value in checks.values())


def test_sampling_gpu_profile_rejects_unbound_manifest(tmp_path: Path) -> None:
    payload = _sampling_payload(tmp_path)
    path = tmp_path / "sampler_manifest.json"
    path.write_text(
        SmallObjectSamplingManifest(
            dataset_manifest="fixture-hash",
            protocol_hash="wrong",
            runtime_payload_hash=payload.payload_hash,
            split="train",
            seed=17,
            area_thresholds={"small": 0.01},
            image_count=1,
            small_image_count=1,
            raw_weights=[1.0],
            final_weights=[1.0],
            image_paths=["a.png"],
            clipping_statistics={"max_weight": 3.0},
            sample_count=1,
            adapter_hash="a" * 64,
        ).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sampling_protocol_bound"):
        validate_component_gpu_profile(
            "sampling.small_object",
            payload,
            {"adapter_sampler_manifest": path},
        )
