from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.component_schemas import (
    ComponentCertificationReport,
    ComponentCertificationStage,
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
