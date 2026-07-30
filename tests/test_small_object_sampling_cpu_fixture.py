"""CPU-only end-to-end fixture for the sampling runtime hook."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.small_object_sampling import (
    run_small_object_sampling_cpu_fixture,
)
from yolo_agent.components.adapters import AdapterContext
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    SmallObjectSamplingAdapter,
    SmallObjectSamplingManifest,
)
from yolo_agent.components.contracts import load_contracts


def test_cpu_fixture_exercises_hook_manifest_ddp_resume_and_validation(
    tmp_path: Path,
) -> None:
    contract = next(
        item
        for item in load_contracts(
            "configs/components/sampling/small_object_sampling.yaml"
        )
        if item.component_id == "sampling.small_object"
    )
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=tmp_path,
        options={"imgsz": 640, "seed": 17},
    )
    adapter = SmallObjectSamplingAdapter()
    preview = adapter.prepare_patch({}, {}, context)
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="cpu-golden-protocol",
        base_command=[
            "yolo",
            "detect",
            "train",
            "model=yolo26n.pt",
            "data=fixture.yaml",
            "imgsz=640",
        ],
        generated_config={
            "model_config": preview.patched_model_config,
            "training_config": preview.patched_training_config,
        },
    )
    payload_path = payload.write(tmp_path / "adapter_runtime_payload.yaml")

    report = run_small_object_sampling_cpu_fixture(
        runtime_payload_path=payload_path,
        workspace=tmp_path / "certification",
    )

    assert report.status == "passed", report.errors
    assert report.checks["train_dataloader_hook_calls"] == 3
    assert all(
        report.checks[key] is True
        for key in (
            "train_dataloader_hook_called",
            "sampler_manifest_verified",
            "ddp_deterministic_sharding",
            "resume_state_restored",
            "validation_loader_unchanged",
            "plugin_failures_empty",
        )
    )
    manifest = SmallObjectSamplingManifest.model_validate_json(
        report.sampler_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest.protocol_hash == "cpu-golden-protocol"
    assert manifest.runtime_payload_hash == payload.payload_hash
