from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import yaml

from yolo_agent.certification.component_runner import (
    ComponentCertificationRunner,
    SubprocessComponentSmokeBackend,
)
from yolo_agent.certification.component_schemas import (
    ComponentGPUCertificationEvidence,
    ComponentGPUProtocol,
    ComponentGPUResources,
    ComponentSmokeWorkerReport,
    ComponentSmokeWorkerRequest,
)
from yolo_agent.components.adapters import (
    AdapterContext,
    AdapterRuntimePayload,
    DummyAdapter,
)
from yolo_agent.components.contracts import load_contracts


COMPONENT_ID = "dummy.certification"


class FakeSmokeBackend:
    def __init__(
        self,
        *,
        fail: bool = False,
        omit_gpu_evidence: bool = False,
    ) -> None:
        self.fail = fail
        self.omit_gpu_evidence = omit_gpu_evidence
        self.calls: list[str] = []
        self.requests: list[ComponentSmokeWorkerRequest] = []

    def run(
        self,
        request: ComponentSmokeWorkerRequest,
        *,
        workdir: Path,
    ) -> tuple[ComponentSmokeWorkerReport, Path]:
        self.calls.append(request.mode)
        self.requests.append(request)
        payload = AdapterRuntimePayload.read(request.runtime_payload_path)
        checks: dict[str, bool | str | int | float] = {
            "isolated": True,
            "cuda_available": request.mode == "gpu",
        }
        if request.mode == "gpu" and not self.fail and not self.omit_gpu_evidence:
            gpu_checks = {
                "real_ultralytics_train": True,
                "required_hooks_observed": True,
                "backward_observed": True,
                "amp_enabled": True,
                "checkpoint_saved": True,
                "resume_completed": True,
                "resume_checkpoint_saved": True,
                "adapter_hash_matched": True,
                "fixture_manifest_matched": True,
                "adapter_artifacts_complete": True,
                "component_profile_verified": True,
            }
            protocol = ComponentGPUProtocol(
                component_id=request.contract.component_id,
                adapter_hash=request.adapter_hash or "a" * 64,
                runtime_payload_hash=payload.payload_hash,
                fixture_manifest_hash="f" * 64,
                model_sha256="m" * 64,
                ultralytics_version=request.ultralytics_version or "8.4.0",
                device=request.device,
            )
            evidence = ComponentGPUCertificationEvidence(
                component_id=request.contract.component_id,
                status="passed",
                worker_protocol_hash=request.protocol_hash,
                gpu_protocol=protocol,
                runtime_payload_path=request.runtime_payload_path,
                runtime_payload_hash=payload.payload_hash,
                checks=gpu_checks,
                resources=ComponentGPUResources(
                    device=request.device,
                    gpu_name="Mock CUDA",
                    total_vram_mb=24576,
                    peak_vram_mb=1024,
                    train_duration_s=1.0,
                    resume_duration_s=0.5,
                    latency_ms=2.0,
                    model_size_mb=5.0,
                ),
            )
            evidence_path = workdir / "fake-gpu-evidence.yaml"
            evidence.to_yaml(evidence_path)
            checks.update(gpu_checks)
            checks["gpu_evidence_path"] = str(evidence_path)
        report = ComponentSmokeWorkerReport(
            component_id=request.contract.component_id,
            mode=request.mode,
            status="failed" if self.fail else "passed",
            protocol_hash=request.protocol_hash,
            payload_hash=payload.payload_hash,
            evidence_kind="local",
            process_id=os.getpid(),
            cuda_available=request.mode == "gpu" and not self.fail,
            device=request.device,
            checks=checks,
            errors=["synthetic failure"] if self.fail else [],
        )
        path = workdir / f"fake-{request.mode}.yaml"
        report.to_yaml(path, exclude_none=True, sort_keys=False)
        return report, path


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "components.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "components": {
                    COMPONENT_ID: {
                        "display_name": "Dummy certification",
                        "category": "augmentation",
                        "implementation_path": "yolo_agent.components.adapters.dummy",
                        "adapter_class": "DummyAdapter",
                        "maturity": "adapter_implemented",
                        "fixed_imgsz_compatible": True,
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _run(
    tmp_path: Path,
    backend: FakeSmokeBackend,
    *,
    mode: Literal["cpu", "gpu"],
    execute_gpu: bool = False,
):
    return ComponentCertificationRunner(
        worker_backend=backend,
        contract_paths=[_source(tmp_path)],
    ).run(
        component_id=COMPONENT_ID,
        mode=mode,
        workdir=tmp_path / "certification",
        registry_path=tmp_path / "registry.yaml",
        protocol_hash="protocol-1",
        execute_gpu=execute_gpu,
    )


def test_cpu_runner_promotes_only_after_isolated_local_smoke(tmp_path: Path) -> None:
    backend = FakeSmokeBackend()
    report = _run(tmp_path, backend, mode="cpu")

    assert report.status == "passed"
    assert report.initial_maturity == "adapter_implemented"
    assert report.final_maturity == "smoke_passed"
    assert report.next_maturity == "gpu_certified"
    assert backend.calls == ["cpu"]
    assert {item.stage_id for item in report.stages} == {
        "adapter_import",
        "runtime_payload",
        "hook_signature",
        "unit_tests",
        "isolated_smoke",
    }
    assert (tmp_path / "certification" / "component_certification.cpu.yaml").is_file()


def test_gpu_runner_requires_cpu_smoke_before_backend_execution(tmp_path: Path) -> None:
    backend = FakeSmokeBackend()
    report = _run(tmp_path, backend, mode="gpu", execute_gpu=True)

    assert report.status == "blocked"
    assert report.errors == ["cpu_smoke_passed_required"]
    assert report.missing_artifacts == ["smoke_passed"]
    assert backend.calls == []


def test_gpu_runner_requires_explicit_execution_opt_in(tmp_path: Path) -> None:
    backend = FakeSmokeBackend()
    cpu = _run(tmp_path, backend, mode="cpu")
    gpu = _run(tmp_path, backend, mode="gpu", execute_gpu=False)

    assert cpu.status == "passed"
    assert gpu.status == "blocked"
    assert gpu.errors == ["gpu_execution_not_confirmed"]
    assert backend.calls == ["cpu"]


def test_cpu_then_gpu_promotes_sequentially_in_same_registry(tmp_path: Path) -> None:
    backend = FakeSmokeBackend()
    cpu = _run(tmp_path, backend, mode="cpu")
    gpu = _run(tmp_path, backend, mode="gpu", execute_gpu=True)

    assert cpu.final_maturity == "smoke_passed"
    assert gpu.status == "passed"
    assert gpu.initial_maturity == "smoke_passed"
    assert gpu.final_maturity == "gpu_certified"
    assert gpu.next_maturity == "pilot_reproduced"
    assert backend.calls == ["cpu", "gpu"]
    gpu_request = backend.requests[-1]
    assert gpu_request.real_gpu_training is True
    assert gpu_request.model == "yolo26n.pt"
    assert gpu_request.adapter_hash
    assert gpu_request.ultralytics_version


def test_gpu_runner_rejects_worker_pass_without_real_evidence(tmp_path: Path) -> None:
    backend = FakeSmokeBackend(omit_gpu_evidence=True)
    cpu = _run(tmp_path, backend, mode="cpu")
    gpu = _run(tmp_path, backend, mode="gpu", execute_gpu=True)

    assert cpu.final_maturity == "smoke_passed"
    assert gpu.status == "failed"
    assert gpu.final_maturity == "smoke_passed"
    assert gpu.missing_artifacts == ["gpu_certified"]


def test_subprocess_backend_uses_utf8_and_tolerates_empty_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contracts(_source(tmp_path))[0]
    context = AdapterContext(contract=contract, workspace=tmp_path)
    payload = DummyAdapter().build_runtime_payload(
        context,
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    assert payload is not None
    request = ComponentSmokeWorkerRequest(
        contract=contract,
        mode="cpu",
        protocol_hash="protocol-1",
        runtime_payload_path=payload.write(tmp_path / "payload.yaml"),
        workspace=tmp_path,
    )
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        output = Path(command[command.index("--output") + 1])
        ComponentSmokeWorkerReport(
            component_id=contract.component_id,
            mode="cpu",
            status="passed",
            protocol_hash="protocol-1",
            payload_hash=payload.payload_hash,
            evidence_kind="local",
            process_id=os.getpid(),
        ).to_yaml(output)
        return SimpleNamespace(returncode=0, stdout=None, stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    report, _ = SubprocessComponentSmokeBackend().run(request, workdir=tmp_path)

    assert report.status == "passed"
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "replace"
    assert (tmp_path / "cpu_worker.log").read_text(encoding="utf-8") == ""


def test_failed_isolated_smoke_is_retained_without_promotion(tmp_path: Path) -> None:
    backend = FakeSmokeBackend(fail=True)
    report = _run(tmp_path, backend, mode="cpu")

    assert report.status == "failed"
    assert report.final_maturity == "unit_tested"
    assert report.missing_artifacts == ["smoke_passed"]
    assert report.errors == ["synthetic failure"]


def test_default_contract_discovery_skips_legacy_component_cards() -> None:
    runner = ComponentCertificationRunner(worker_backend=FakeSmokeBackend())

    contract = runner._find_source_contract("sampling.small_object")

    assert contract.component_id == "sampling.small_object"
    assert contract.adapter_class == "SmallObjectSamplingAdapter"


def test_sampling_cpu_certification_runs_complete_golden_path(tmp_path: Path) -> None:
    report = ComponentCertificationRunner().run(
        component_id="sampling.small_object",
        mode="cpu",
        workdir=tmp_path / "sampling-certification",
        registry_path=tmp_path / "registry.yaml",
    )

    assert report.status == "passed", report.errors
    assert report.final_maturity == "smoke_passed"
    assert "cpu_golden_path" in report.generated_paths
    smoke = next(item for item in report.stages if item.stage_id == "isolated_smoke")
    assert smoke.checks["train_dataloader_hook_called"] is True
    assert smoke.checks["sampler_manifest_verified"] is True
    assert smoke.checks["ddp_deterministic_sharding"] is True
    assert smoke.checks["resume_state_restored"] is True
    assert smoke.checks["validation_loader_unchanged"] is True


@pytest.mark.parametrize(
    "component_id",
    [
        "loss.quality.correlation",
        "loss.calibration.bpc",
        "loss.quality.pseudo_iou",
    ],
)
def test_quality_loss_cpu_certification_runs_complete_golden_path(
    component_id: str,
    tmp_path: Path,
) -> None:
    report = ComponentCertificationRunner().run(
        component_id=component_id,
        mode="cpu",
        workdir=tmp_path / component_id,
        registry_path=tmp_path / "registry.yaml",
    )

    assert report.status == "passed", report.errors
    assert report.final_maturity == "smoke_passed"
    assert "cpu_golden_path" in report.generated_paths
    smoke = next(item for item in report.stages if item.stage_id == "isolated_smoke")
    assert smoke.checks["trainer_bridge_called"] is True
    assert smoke.checks["total_loss_changed"] is True
    assert smoke.checks["student_backward"] is True
    assert smoke.checks["zero_weight_native_equivalent"] is True


def test_distillation_cpu_certification_runs_complete_golden_path(
    tmp_path: Path,
) -> None:
    report = ComponentCertificationRunner().run(
        component_id="distillation.yolo26_teacher_student",
        mode="cpu",
        workdir=tmp_path / "distillation-certification",
        registry_path=tmp_path / "registry.yaml",
    )

    assert report.status == "passed", report.errors
    assert report.final_maturity == "smoke_passed"
    assert "cpu_golden_path" in report.generated_paths
    smoke = next(item for item in report.stages if item.stage_id == "isolated_smoke")
    assert smoke.checks["trainer_bridge_called"] is True
    assert smoke.checks["student_backward"] is True
    assert smoke.checks["teacher_no_grad"] is True
    assert smoke.checks["method_profiles_only"] is True


@pytest.mark.parametrize(
    "component_id",
    [
        "head.p2_small_object",
        "neck.multi_scale_fusion",
        "neck.gold_gather_distribute",
        "neck.rtmdet_large_kernel",
    ],
)
def test_graph_cpu_certification_runs_complete_golden_path(
    component_id: str,
    tmp_path: Path,
) -> None:
    report = ComponentCertificationRunner().run(
        component_id=component_id,
        mode="cpu",
        workdir=tmp_path / component_id,
        registry_path=tmp_path / "registry.yaml",
        options={
            "audit_imgsz": 64,
            "latency_warmup": 0,
            "latency_iterations": 1,
            "resource_limits": {
                "max_latency_regression": 100.0,
                "max_vram_regression": 10.0,
                "max_parameter_regression": 10.0,
                "max_model_size_regression": 10.0,
            },
        },
    )

    assert report.status == "passed", report.errors
    assert report.final_maturity == "smoke_passed"
    assert "cpu_golden_path" in report.generated_paths
    smoke = next(item for item in report.stages if item.stage_id == "isolated_smoke")
    assert smoke.checks["real_forward"] is True
    assert smoke.checks["native_loss_preserved"] is True
    assert smoke.checks["backward"] is True
    assert smoke.checks["amp"] is True
    assert smoke.checks["partial_checkpoint_audit"] is True
    assert smoke.checks["export"] is True
    assert smoke.checks["resource_guard"] is True


@pytest.mark.parametrize(
    "component_id",
    [
        "assigner.task_aligned",
        "assigner.optimal_transport",
        "assigner.dynamic_smooth_label",
    ],
)
def test_assignment_cpu_certification_runs_shadow_golden_path(
    component_id: str,
    tmp_path: Path,
) -> None:
    report = ComponentCertificationRunner().run(
        component_id=component_id,
        mode="cpu",
        workdir=tmp_path / component_id,
        registry_path=tmp_path / "registry.yaml",
    )

    assert report.status == "passed", report.errors
    assert report.final_maturity == "smoke_passed"
    assert "cpu_golden_path" in report.generated_paths
    smoke = next(item for item in report.stages if item.stage_id == "isolated_smoke")
    assert smoke.checks["shadow_mode_only"] is True
    assert smoke.checks["native_audit_verified"] is True
    assert smoke.checks["positive_ratio_recorded"] is True
    assert smoke.checks["conflict_rate_recorded"] is True
    assert smoke.checks["native_loss_equivalent"] is True
    assert smoke.checks["active_pilot_blocked_until_explicit_gate"] is True
