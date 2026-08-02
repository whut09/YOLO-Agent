from __future__ import annotations

from pathlib import Path

import yolo_agent.cli as cli
from yolo_agent.certification.component_gpu import GPU_CERTIFICATION_COMPONENTS
from yolo_agent.certification.component_gpu_suite import (
    PaperComponentGPUSuiteReport,
    PaperComponentGPUSuiteRunner,
)
from yolo_agent.certification.component_schemas import (
    ComponentCertificationReport,
    ComponentCertificationStage,
)


def _certification_report(
    tmp_path: Path,
    *,
    component_id: str,
    mode: str,
    passed: bool = True,
) -> ComponentCertificationReport:
    stages = (
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
        component_id=component_id,
        mode=mode,
        status="passed" if passed else "failed",
        initial_maturity="smoke_passed" if mode == "gpu" else "adapter_implemented",
        final_maturity=(
            "gpu_certified"
            if passed and mode == "gpu"
            else "smoke_passed"
            if passed
            else "smoke_passed"
        ),
        next_maturity="pilot_reproduced" if mode == "gpu" else "gpu_certified",
        protocol_hash="p" * 64,
        registry_path=tmp_path / "registry.yaml",
        workdir=tmp_path / component_id,
        stages=[
            ComponentCertificationStage(
                stage_id=stage,
                status="passed" if passed else "failed",
            )
            for stage in stages
        ],
        missing_artifacts=[] if passed else ["gpu_certified"],
        errors=[] if passed else ["synthetic GPU failure"],
    )


class FakeComponentRunner:
    def __init__(self, tmp_path: Path, *, fail_component: str | None = None) -> None:
        self.tmp_path = tmp_path
        self.fail_component = fail_component
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> ComponentCertificationReport:
        self.calls.append(kwargs)
        component_id = str(kwargs["component_id"])
        mode = str(kwargs["mode"])
        failed = component_id == self.fail_component and mode == "gpu"
        return _certification_report(
            self.tmp_path,
            component_id=component_id,
            mode=mode,
            passed=not failed,
        )


def test_priority_suite_runs_all_components_and_caps_maturity(tmp_path: Path) -> None:
    backend = FakeComponentRunner(tmp_path)
    report = PaperComponentGPUSuiteRunner(backend).run(
        workdir=tmp_path / "suite",
        registry_path=tmp_path / "registry.yaml",
        model="yolo26n.pt",
        teacher="yolo26s.pt",
        ensemble_teacher="yolo26m.pt",
        device="0",
        execute_real_gpu=True,
    )

    assert report.status == "passed"
    assert [item.component_id for item in report.results] == list(
        GPU_CERTIFICATION_COMPONENTS
    )
    assert all(item.final_maturity == "gpu_certified" for item in report.results)
    assert len(backend.calls) == 2 * len(GPU_CERTIFICATION_COMPONENTS)
    distillation_calls = [
        item
        for item in backend.calls
        if str(item["component_id"]).startswith("distillation.")
    ]
    assert all(item["options"] is not None for item in distillation_calls)
    ensemble_calls = [
        item
        for item in distillation_calls
        if item["component_id"] == "distillation.teacher_ensemble"
    ]
    assert all(
        item["options"]
        == {"teacher": "yolo26s.pt", "teachers": ["yolo26m.pt"]}
        for item in ensemble_calls
    )
    assert (tmp_path / "suite" / "paper_component_gpu_suite.yaml").is_file()


def test_priority_suite_stops_after_first_gpu_failure(tmp_path: Path) -> None:
    backend = FakeComponentRunner(
        tmp_path,
        fail_component="loss.quality.correlation",
    )
    report = PaperComponentGPUSuiteRunner(backend).run(
        workdir=tmp_path / "suite",
        registry_path=tmp_path / "registry.yaml",
        model="yolo26n.pt",
        device="0",
        execute_real_gpu=True,
    )

    assert report.status == "failed"
    assert report.stopped_at == "loss.quality.correlation"
    failed_index = GPU_CERTIFICATION_COMPONENTS.index("loss.quality.correlation")
    assert len(backend.calls) == 2 * (failed_index + 1)
    assert report.results[failed_index].status == "failed"
    assert all(
        item.status == "not_run" for item in report.results[failed_index + 1 :]
    )


def test_priority_suite_cli_is_blocked_without_explicit_gpu_opt_in(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    class FakeSuite:
        def run(self, **kwargs: object) -> PaperComponentGPUSuiteReport:
            values = dict(kwargs)
            values["execute_real_gpu"] = False
            return PaperComponentGPUSuiteRunner().run(**values)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "PaperComponentGPUSuiteRunner", FakeSuite)
    result = cli.main(
        [
            "advanced",
            "certify-paper-components",
            "--workdir",
            str(tmp_path / "suite"),
        ]
    )

    assert result == 1
    output = capsys.readouterr().out
    assert "Status:    blocked" in output
    assert "gpu_execution_not_confirmed" in output
    assert "pilot_reproduced is not granted" in output
