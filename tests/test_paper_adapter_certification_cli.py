from pathlib import Path

import pytest

import yolo_agent.cli as cli
from yolo_agent.certification.paper_adapter_factory_schemas import (
    AdapterCertificationIdentity,
    PaperAdapterCertificationReport,
    PaperAdapterCertificationResult,
)


class _Factory:
    calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> PaperAdapterCertificationReport:
        self.calls.append(kwargs)
        workdir = Path(str(kwargs["workdir"]))
        registry = Path(str(kwargs["registry_path"]))
        component_id = "sampling.small_object"
        return PaperAdapterCertificationReport(
            status="passed",
            mode=str(kwargs["mode"]),
            execute_real_gpu=bool(kwargs["execute_real_gpu"]),
            resume=bool(kwargs["resume"]),
            changed_only=bool(kwargs["changed_only"]),
            registry_path=registry,
            coverage_report_path=workdir / "paper_adapter_coverage.yaml",
            selected_component_ids=[component_id],
            results=[
                PaperAdapterCertificationResult(
                    component_id=component_id,
                    identity=AdapterCertificationIdentity(
                        component_id=component_id,
                        adapter_hash="a" * 64,
                        code_commit="commit-one",
                        ultralytics_version="8.4.0",
                        protocol_hash="protocol-one",
                    ),
                    status="passed",
                    initial_maturity="adapter_implemented",
                    final_maturity=(
                        "gpu_certified"
                        if kwargs["mode"] == "gpu"
                        else "smoke_passed"
                    ),
                    selection_reason="selected_all_reusable_adapters",
                    cpu_report=workdir / component_id / "component.cpu.yaml",
                    gpu_report=(
                        workdir / component_id / "component.gpu.yaml"
                        if kwargs["mode"] == "gpu"
                        else None
                    ),
                    matched_pilot_fixture=(
                        workdir / component_id / "matched_pilot_fixture.yaml"
                        if kwargs["mode"] == "gpu"
                        else None
                    ),
                )
            ],
        )


class _ExplodingFactory:
    def run(self, **kwargs: object) -> PaperAdapterCertificationReport:
        raise RuntimeError("one adapter failed contract smoke")


def test_automatic_readiness_failure_isolated_without_aborting_train_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "PaperAdapterCertificationFactory", _ExplodingFactory)

    refreshed = cli._prepare_automatic_paper_readiness(
        run_root=tmp_path / "runs",
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        component_ids=["loss.quality.correlation", "neck.rtmdet_large_kernel"],
    )

    assert refreshed is False
    output = capsys.readouterr().out
    assert "existing ready candidates will continue" in output
    assert "0 ready and 2 isolated" in output


def test_batch_certification_cli_defaults_to_cpu_and_supports_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _Factory.calls = []
    monkeypatch.setattr(cli, "PaperAdapterCertificationFactory", _Factory)

    result = cli.main(
        [
            "advanced",
            "certify-paper-adapters",
            "--workdir",
            str(tmp_path / "batch"),
            "--registry",
            str(tmp_path / "registry.yaml"),
            "--component",
            "sampling.small_object",
            "--resume",
        ]
    )

    assert result == 0
    assert _Factory.calls[0]["mode"] == "cpu"
    assert _Factory.calls[0]["resume"] is True
    assert _Factory.calls[0]["component_ids"] == ["sampling.small_object"]
    output = capsys.readouterr().out
    assert "Status:    passed" in output
    assert "maturity=adapter_implemented->smoke_passed" in output
    assert "matched fixture is not pilot evidence" in output


def test_batch_gpu_cli_requires_explicit_execution_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Factory.calls = []
    monkeypatch.setattr(cli, "PaperAdapterCertificationFactory", _Factory)

    cli.main(
        [
            "advanced",
            "certify-paper-adapters",
            "--workdir",
            str(tmp_path / "batch"),
            "--gpu",
        ]
    )

    assert _Factory.calls[0]["mode"] == "gpu"
    assert _Factory.calls[0]["execute_real_gpu"] is False


def test_execute_real_gpu_is_invalid_in_cpu_mode() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "advanced",
                "certify-paper-adapters",
                "--cpu",
                "--execute-real-gpu",
            ]
        )
