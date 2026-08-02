from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.adapters.ultralytics.plugin_context import (
    PluginRuntimeEvidence,
    runtime_evidence_path,
)
from yolo_agent.certification.component_gpu import (
    GPUCheckpointState,
    GPUTrainingStageResult,
    RealComponentGPUExecutionBackend,
    component_gpu_options,
    prepare_component_gpu_run,
    run_real_component_gpu_certification,
)
from yolo_agent.certification.component_runner import ComponentCertificationRunner
from yolo_agent.certification.component_schemas import (
    ComponentGPUResources,
    ComponentSmokeWorkerRequest,
)
from yolo_agent.components.adapters import AdapterContext, AdapterRuntimePayload
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    SmallObjectSamplingManifest,
)


class MockExecutionBackend:
    def __init__(self, *, fail_training: bool = False) -> None:
        self.fail_training = fail_training
        self.epoch = -1
        self.calls = 0

    def run_training(
        self,
        *,
        payload_path: Path,
        command: list[str],
        project_dir: Path,
        run_name: str,
    ) -> GPUTrainingStageResult:
        if self.fail_training:
            raise RuntimeError("synthetic GPU training failure")
        self.calls += 1
        self.epoch += 1
        run_dir = project_dir / run_name
        checkpoint = run_dir / "weights" / "last.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{self.epoch}".encode())
        results = run_dir / "results.csv"
        results.write_text(
            "epoch,train/box_loss,train/cls_loss\n0,1.0,2.0\n",
            encoding="utf-8",
        )
        payload = AdapterRuntimePayload.read(payload_path)
        for expected in payload.expected_artifacts:
            artifact = payload_path.parent / expected.relative_path
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("{}", encoding="utf-8")
        if payload.component_ids == ["sampling.small_object"]:
            manifest = SmallObjectSamplingManifest(
                dataset_manifest="fixture",
                protocol_hash=payload.protocol_hash,
                runtime_payload_hash=payload.payload_hash,
                split="train",
                seed=17,
                area_thresholds={"small": 0.01},
                image_count=1,
                small_image_count=1,
                raw_weights=[2.0],
                final_weights=[2.0],
                image_paths=["image.png"],
                clipping_statistics={"max_weight": 3.0},
                sample_count=1,
                adapter_hash="a" * 64,
            )
            (payload_path.parent / "sampler_manifest.json").write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
        evidence = PluginRuntimeEvidence(
            payload_hash=payload.payload_hash,
            protocol_hash=payload.protocol_hash,
            component_ids=payload.component_ids,
            changed_variables=payload.changed_variables,
            ultralytics_version="8.4.87",
            signature_hash="s" * 64,
            compatible=True,
            hook_call_counts={
                reference.reference: {
                    **{hook: self.calls for hook in reference.required_hooks},
                    "on_checkpoint_load": max(self.calls - 1, 0),
                }
                for reference in payload.plugin_references
            },
        )
        evidence.to_json(runtime_evidence_path(payload_path))
        return GPUTrainingStageResult(
            checkpoint=checkpoint,
            results_csv=results,
            duration_s=0.1,
            completed_epochs=self.calls,
        )

    def prepare_resume_checkpoint(self, source: Path, target: Path) -> Path:
        target.write_bytes(source.read_bytes())
        return target

    def inspect_checkpoint(self, checkpoint: Path) -> GPUCheckpointState:
        return GPUCheckpointState(epoch=self.epoch, amp=True, model_size_mb=5.0)

    def resource_evidence(
        self,
        *,
        checkpoint: Path,
        device: str,
        train_duration_s: float,
        resume_duration_s: float,
    ) -> ComponentGPUResources:
        return ComponentGPUResources(
            device=device,
            gpu_name="Mock CUDA",
            total_vram_mb=24576,
            peak_vram_mb=1024,
            train_duration_s=train_duration_s,
            resume_duration_s=resume_duration_s,
            latency_ms=2.5,
            model_size_mb=5.0,
        )


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


def test_mock_backend_exercises_full_gpu_artifact_contract(tmp_path: Path) -> None:
    request, source = _request_and_payload(tmp_path)
    backend = MockExecutionBackend()

    evidence = run_real_component_gpu_certification(
        request,
        source,
        backend=backend,
    )

    assert evidence.status == "passed"
    assert backend.calls == 2
    assert evidence.checks["required_hooks_observed"] is True
    assert evidence.checks["backward_observed"] is True
    assert evidence.checks["resume_completed"] is True
    assert evidence.checks["component_profile_verified"] is True
    assert evidence.checks["stateful_resume_hook_observed"] is True
    assert evidence.resources is not None
    assert evidence.resources.gpu_name == "Mock CUDA"
    assert (request.workspace / "component_gpu_evidence.yaml").is_file()


def test_failed_gpu_training_retains_failed_evidence(tmp_path: Path) -> None:
    request, source = _request_and_payload(tmp_path)

    evidence = run_real_component_gpu_certification(
        request,
        source,
        backend=MockExecutionBackend(fail_training=True),
    )

    assert evidence.status == "failed"
    assert "synthetic GPU training failure" in evidence.errors
    assert "gpu_contract_failed:real_ultralytics_train" in evidence.errors
    report = request.workspace / "component_gpu_evidence.yaml"
    assert report.is_file()


def test_real_backend_builds_auditable_resume_source_with_sidecars(
    tmp_path: Path,
) -> None:
    import torch

    source = tmp_path / "last.pt"
    torch.save(
        {"epoch": -1, "train_args": {"epochs": 1}, "optimizer": None},
        source,
    )
    sidecar = source.with_name(source.name + ".small_object_sampler.json")
    sidecar.write_text('{"epoch": 0}', encoding="utf-8")
    target = tmp_path / "resume_source.pt"

    RealComponentGPUExecutionBackend().prepare_resume_checkpoint(source, target)

    restored = torch.load(target, map_location="cpu", weights_only=False)
    assert restored["epoch"] == 0
    assert restored["train_args"]["epochs"] == 2
    assert target.with_name(target.name + ".small_object_sampler.json").is_file()


def test_bpc_gpu_fixture_forces_observable_calibration_penalty(tmp_path: Path) -> None:
    contract = ComponentCertificationRunner()._find_source_contract(
        "loss.calibration.bpc"
    )
    request = ComponentSmokeWorkerRequest(
        contract=contract,
        mode="gpu",
        protocol_hash="protocol",
        runtime_payload_path=tmp_path / "payload.yaml",
        workspace=tmp_path,
        real_gpu_training=True,
    )

    options = component_gpu_options(
        request,
        model_path=tmp_path / "yolo26n.pt",
        data_yaml=tmp_path / "coco.yaml",
    )

    assert options["confidence_threshold"] == 0.0
    assert options["imgsz"] == 640


def test_teacher_ensemble_gpu_options_require_two_local_teachers(
    tmp_path: Path,
) -> None:
    contract = ComponentCertificationRunner()._find_source_contract(
        "distillation.teacher_ensemble"
    )
    student = tmp_path / "yolo26n.pt"
    teacher = tmp_path / "yolo26s.pt"
    teacher_m = tmp_path / "yolo26m.pt"
    student.write_bytes(b"student")
    teacher.write_bytes(b"teacher")
    teacher_m.write_bytes(b"teacher-m")
    request = ComponentSmokeWorkerRequest(
        contract=contract,
        mode="gpu",
        protocol_hash="protocol",
        runtime_payload_path=tmp_path / "payload.yaml",
        workspace=tmp_path,
        model=str(student),
        options={"teacher": str(teacher), "teachers": [str(teacher_m)]},
        real_gpu_training=True,
    )

    options = component_gpu_options(
        request,
        model_path=student,
        data_yaml=tmp_path / "coco.yaml",
    )

    assert options["teachers"] == [str(teacher_m.resolve())]
    assert options["imgsz"] == 640

    with pytest.raises(ValueError, match="additional local teacher"):
        component_gpu_options(
            request.model_copy(update={"options": {"teacher": str(teacher)}}),
            model_path=student,
            data_yaml=tmp_path / "coco.yaml",
        )
