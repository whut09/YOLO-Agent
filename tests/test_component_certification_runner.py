from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pytest
import yaml

from yolo_agent.certification.component_runner import ComponentCertificationRunner
from yolo_agent.certification.component_schemas import (
    ComponentSmokeWorkerReport,
    ComponentSmokeWorkerRequest,
)
from yolo_agent.components.adapters import AdapterRuntimePayload


COMPONENT_ID = "dummy.certification"


class FakeSmokeBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def run(
        self,
        request: ComponentSmokeWorkerRequest,
        *,
        workdir: Path,
    ) -> tuple[ComponentSmokeWorkerReport, Path]:
        self.calls.append(request.mode)
        payload = AdapterRuntimePayload.read(request.runtime_payload_path)
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
            checks={"isolated": True, "cuda_available": request.mode == "gpu"},
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
