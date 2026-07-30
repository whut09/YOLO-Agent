"""Offline CLI tests for component runtime certification."""

from __future__ import annotations

from pathlib import Path

import pytest

import yolo_agent.cli as cli
from yolo_agent.certification.component_schemas import ComponentCertificationReport


COMPONENT_ID = "small_object_sampling"


def _report(
    tmp_path: Path,
    *,
    mode: str,
    status: str,
    missing: list[str] | None = None,
    errors: list[str] | None = None,
) -> ComponentCertificationReport:
    stages = (
        [
            {
                "stage_id": stage,
                "status": "passed",
            }
            for stage in (
                "adapter_import",
                "runtime_payload",
                "hook_signature",
                "unit_tests",
                "isolated_smoke",
            )
        ]
        if mode == "cpu" and status == "passed"
        else []
    )
    return ComponentCertificationReport.model_validate(
        {
            "component_id": COMPONENT_ID,
            "mode": mode,
            "status": status,
            "initial_maturity": "adapter_implemented",
            "final_maturity": (
                "smoke_passed" if status == "passed" else "adapter_implemented"
            ),
            "next_maturity": (
                "gpu_certified" if status == "passed" else "runtime_integrated"
            ),
            "protocol_hash": "a" * 64,
            "adapter_hash": "b" * 64,
            "code_commit": "test-commit",
            "ultralytics_version": "test-version",
            "registry_path": tmp_path / "registry.yaml",
            "workdir": tmp_path / "certification",
            "stages": stages,
            "missing_artifacts": missing or [],
            "generated_paths": {
                "runtime_payload": tmp_path / "runtime_payload.yaml"
            },
            "errors": errors or [],
        }
    )


class FakeRunner:
    report: ComponentCertificationReport
    calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> ComponentCertificationReport:
        self.calls.append(kwargs)
        return self.report


def test_cpu_component_certification_prints_artifacts_and_next_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    FakeRunner.calls = []
    FakeRunner.report = _report(tmp_path, mode="cpu", status="passed")
    monkeypatch.setattr(cli, "ComponentCertificationRunner", FakeRunner)

    result = cli.main(
        [
            "advanced",
            "certify-component",
            "--component",
            COMPONENT_ID,
            "--cpu",
            "--workdir",
            str(tmp_path / "certification"),
            "--registry",
            str(tmp_path / "registry.yaml"),
        ]
    )

    assert result == 0
    assert FakeRunner.calls[0]["mode"] == "cpu"
    assert FakeRunner.calls[0]["execute_gpu"] is False
    output = capsys.readouterr().out
    assert "Status:    passed" in output
    assert "Maturity:  adapter_implemented -> smoke_passed" in output
    assert "Missing:   none" in output
    assert "runtime_payload=" in output
    assert "Next:      gpu_certified" in output


def test_gpu_component_certification_is_explicit_and_reports_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    FakeRunner.calls = []
    FakeRunner.report = _report(
        tmp_path,
        mode="gpu",
        status="blocked",
        missing=["smoke_passed"],
        errors=["cpu_smoke_passed_required"],
    )
    monkeypatch.setattr(cli, "ComponentCertificationRunner", FakeRunner)

    result = cli.main(
        [
            "advanced",
            "certify-component",
            "--component",
            COMPONENT_ID,
            "--gpu",
        ]
    )

    assert result == 1
    assert FakeRunner.calls[0]["mode"] == "gpu"
    assert FakeRunner.calls[0]["execute_gpu"] is True
    output = capsys.readouterr().out
    assert "Status:    blocked" in output
    assert "Missing:   smoke_passed" in output
    assert "Reason:    cpu_smoke_passed_required" in output


@pytest.mark.parametrize(
    "mode_args",
    [[], ["--cpu", "--gpu"]],
)
def test_component_certification_requires_exactly_one_mode(
    mode_args: list[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "advanced",
                "certify-component",
                "--component",
                COMPONENT_ID,
                *mode_args,
            ]
        )


def test_unknown_component_failure_is_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class MissingRunner:
        def run(self, **kwargs: object) -> ComponentCertificationReport:
            raise ValueError("component contract not found: missing")

    monkeypatch.setattr(cli, "ComponentCertificationRunner", MissingRunner)

    result = cli.main(
        [
            "advanced",
            "certify-component",
            "--component",
            "missing",
            "--cpu",
            "--workdir",
            str(tmp_path / "missing"),
        ]
    )

    assert result == 1
    output = capsys.readouterr().out
    assert "Status:    failed" in output
    assert "component contract not found: missing" in output
