from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.component_schemas import (
    ComponentCertificationReport,
    ComponentCertificationStage,
    ComponentGPUCertificationEvidence,
    ComponentGPUProtocol,
    ComponentGPUResources,
    ComponentSmokeWorkerReport,
)


def _report(tmp_path: Path, mode: str = "cpu") -> ComponentCertificationReport:
    stage_ids = (
        [
            "adapter_import",
            "runtime_payload",
            "hook_signature",
            "unit_tests",
            "isolated_smoke",
        ]
        if mode == "cpu"
        else ["cpu_smoke_precondition", "isolated_gpu_smoke"]
    )
    return ComponentCertificationReport(
        component_id="sampling.small_object",
        mode=mode,
        status="passed",
        initial_maturity="adapter_implemented",
        final_maturity="smoke_passed" if mode == "cpu" else "gpu_certified",
        next_maturity="gpu_certified" if mode == "cpu" else "pilot_reproduced",
        protocol_hash="protocol-1",
        registry_path=tmp_path / "registry.yaml",
        workdir=tmp_path,
        stages=[
            ComponentCertificationStage(stage_id=item, status="passed")
            for item in stage_ids
        ],
    )


def test_component_certification_report_hash_round_trips(tmp_path: Path) -> None:
    report = _report(tmp_path)
    path = report.to_yaml(tmp_path / "report.yaml")

    loaded = ComponentCertificationReport.from_yaml(path)
    assert loaded.report_hash == report.report_hash
    assert len(loaded.report_hash) == 64


def test_passed_cpu_and_gpu_reports_require_all_stages(tmp_path: Path) -> None:
    for mode in ("cpu", "gpu"):
        report = _report(tmp_path, mode)
        with pytest.raises(ValueError, match="missing stages"):
            ComponentCertificationReport.model_validate(
                {
                    **report.model_dump(mode="json"),
                    "stages": report.stages[:-1],
                    "report_hash": "",
                }
            )


def test_worker_report_records_isolation_and_runtime_identity(tmp_path: Path) -> None:
    report = ComponentSmokeWorkerReport(
        component_id="sampling.small_object",
        mode="cpu",
        status="passed",
        protocol_hash="protocol-1",
        payload_hash="a" * 64,
        evidence_kind="local",
        process_id=123,
        checks={"shape": True, "backward": True},
    )
    path = report.to_yaml(tmp_path / "worker.yaml")

    loaded = ComponentSmokeWorkerReport.from_yaml(path)
    assert loaded.process_id == 123
    assert loaded.evidence_kind == "local"


def _gpu_protocol() -> ComponentGPUProtocol:
    return ComponentGPUProtocol(
        component_id="sampling.small_object",
        adapter_hash="a" * 64,
        runtime_payload_hash="b" * 64,
        fixture_manifest_hash="c" * 64,
        model_sha256="d" * 64,
        ultralytics_version="8.4.87",
        device="0",
    )


def test_gpu_protocol_hash_is_stable_and_identity_bound() -> None:
    first = _gpu_protocol()
    second = _gpu_protocol()

    assert first.protocol_hash == second.protocol_hash
    assert len(first.protocol_hash) == 64
    changed = first.model_copy(update={"device": "1", "protocol_hash": ""})
    assert changed.protocol_hash != first.protocol_hash


def test_passed_gpu_evidence_requires_complete_real_training_contract(
    tmp_path: Path,
) -> None:
    protocol = _gpu_protocol()
    checks = {
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
    evidence = ComponentGPUCertificationEvidence(
        component_id=protocol.component_id,
        status="passed",
        worker_protocol_hash="worker-protocol",
        gpu_protocol=protocol,
        runtime_payload_path=tmp_path / "payload.yaml",
        runtime_payload_hash=protocol.runtime_payload_hash,
        checks=checks,
        resources=ComponentGPUResources(
            device="0",
            gpu_name="Mock CUDA",
            total_vram_mb=24576,
            peak_vram_mb=1024,
            train_duration_s=1.0,
            resume_duration_s=0.5,
            latency_ms=2.0,
            model_size_mb=5.2,
        ),
    )

    assert evidence.status == "passed"
    with pytest.raises(ValueError, match="backward_observed"):
        ComponentGPUCertificationEvidence.model_validate(
            {
                **evidence.model_dump(mode="json"),
                "checks": {**checks, "backward_observed": False},
            }
        )
