from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.component_schemas import ComponentSmokeWorkerRequest
from yolo_agent.certification.component_worker import run_component_smoke_worker
from yolo_agent.components.adapters import AdapterContext, DummyAdapter
from yolo_agent.components.adapters.inference.slicing import SlicingInferenceAdapter
from yolo_agent.components.contracts import ComponentContract, load_contracts


def _contract() -> ComponentContract:
    return ComponentContract(
        component_id="dummy.worker",
        display_name="Dummy worker",
        category="augmentation",
        implementation_path="yolo_agent.components.adapters.dummy",
        adapter_class="DummyAdapter",
        maturity="adapter_implemented",
        fixed_imgsz_compatible=True,
    )


def _request(tmp_path: Path, *, mode: str = "cpu") -> ComponentSmokeWorkerRequest:
    contract = _contract()
    adapter = DummyAdapter()
    context = AdapterContext(contract=contract, workspace=tmp_path, imgsz=640)
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={"training_config": {"imgsz": 640}},
    )
    assert payload is not None
    path = payload.write(tmp_path / "payload.yaml")
    return ComponentSmokeWorkerRequest(
        contract=contract,
        mode=mode,
        protocol_hash="protocol-1",
        runtime_payload_path=path,
        workspace=tmp_path,
        device="cpu" if mode == "cpu" else "0",
    )


def test_cpu_worker_rejects_dummy_mock_smoke(tmp_path: Path) -> None:
    report = run_component_smoke_worker(_request(tmp_path))

    assert report.status == "failed"
    assert report.evidence_kind == "mock"
    assert "mock_smoke_evidence_cannot_certify_component" in report.errors
    assert report.process_id > 0


def test_gpu_worker_fails_closed_without_cuda_or_gpu_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "yolo_agent.certification.component_worker._cuda_available",
        lambda: False,
    )

    report = run_component_smoke_worker(_request(tmp_path, mode="gpu"))

    assert report.status == "failed"
    assert "cuda_not_available" in report.errors


def test_worker_rejects_payload_protocol_mismatch(tmp_path: Path) -> None:
    request = _request(tmp_path).model_copy(update={"protocol_hash": "protocol-2"})
    report = run_component_smoke_worker(request)

    assert report.status == "failed"
    assert "worker runtime payload protocol mismatch" in report.errors


def test_sahi_worker_cannot_promote_without_real_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    contract = load_contracts("configs/components/inference/sahi_slicing.yaml")[0]
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        workspace=tmp_path,
    )
    payload = SlicingInferenceAdapter().build_runtime_payload(
        context,
        protocol_hash="sahi-protocol",
        base_command=[
            "yolo",
            "detect",
            "train",
            "model=yolo26n.pt",
            "imgsz=640",
        ],
        generated_config={},
    )
    request = ComponentSmokeWorkerRequest(
        contract=contract,
        mode="cpu",
        protocol_hash="sahi-protocol",
        runtime_payload_path=payload.write(tmp_path / "payload.yaml"),
        workspace=tmp_path,
        device="cpu",
    )
    monkeypatch.setattr(
        "yolo_agent.components.adapters.inference.slicing.SlicingInferenceRunner.sahi_available",
        staticmethod(lambda: False),
    )

    report = run_component_smoke_worker(request)

    assert report.status == "failed"
    assert "optional dependency 'sahi' is not installed" in report.errors
    assert not (tmp_path / "sahi_runtime_evidence.json").exists()
