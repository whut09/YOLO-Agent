from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.component_gpu import prepare_component_gpu_run
from yolo_agent.certification.component_runner import ComponentCertificationRunner
from yolo_agent.certification.component_schemas import ComponentSmokeWorkerRequest
from yolo_agent.components.adapters import AdapterContext, AdapterRuntimePayload
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry


def _request_and_payload(
    tmp_path: Path,
) -> tuple[ComponentSmokeWorkerRequest, AdapterRuntimePayload]:
    contract = ComponentCertificationRunner()._find_source_contract(
        "sampling.small_object"
    )
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    context = AdapterContext(contract=contract, workspace=tmp_path / "source")
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="worker-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    assert payload is not None
    payload_path = payload.write(tmp_path / "source" / "payload.yaml")
    model = tmp_path / "yolo26n.pt"
    model.write_bytes(b"local-checkpoint")
    request = ComponentSmokeWorkerRequest(
        contract=contract,
        mode="gpu",
        protocol_hash="worker-protocol",
        runtime_payload_path=payload_path,
        workspace=tmp_path / "gpu",
        device="0",
        model=str(model),
        adapter_hash="a" * 64,
        ultralytics_version="8.4.87",
        real_gpu_training=True,
    )
    return request, payload


def test_prepare_gpu_run_binds_fixture_model_and_fresh_payload(tmp_path: Path) -> None:
    request, source = _request_and_payload(tmp_path)

    prepared = prepare_component_gpu_run(request, source)

    assert prepared.protocol.component_id == "sampling.small_object"
    assert prepared.protocol.imgsz == 640
    assert len(prepared.protocol.fixture_manifest_hash) == 64
    assert len(prepared.protocol.model_sha256) == 64
    assert prepared.runtime_payload_path.is_file()
    assert prepared.fixture_manifest_path.is_file()
    assert "imgsz=640" in prepared.train_command
    assert "amp=True" in prepared.train_command
    assert "workers=0" in prepared.train_command


def test_prepare_gpu_run_requires_explicit_opt_in_and_local_model(
    tmp_path: Path,
) -> None:
    request, source = _request_and_payload(tmp_path)

    with pytest.raises(ValueError, match="real_gpu_training_not_confirmed"):
        prepare_component_gpu_run(
            request.model_copy(update={"real_gpu_training": False}),
            source,
        )
    with pytest.raises(ValueError, match="will not download"):
        prepare_component_gpu_run(
            request.model_copy(update={"model": "missing-yolo26n.pt"}),
            source,
        )
