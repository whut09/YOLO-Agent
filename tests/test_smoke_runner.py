"""Smoke runner tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from yolo_agent.agents.candidate_generator import CandidateConfig, CandidatePlan
from yolo_agent.cli import main
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.tools import smoke_runner
from yolo_agent.tools.smoke_runner import SmokeRunner


def _write_plan(path: Path) -> None:
    plan = CandidatePlan(
        task_scene="generic",
        candidates=[
            CandidateConfig(
                candidate_id="yolo11n_baseline_n",
                base_model="yolo11n",
                scale="n",
                framework="ultralytics",
            )
        ],
    )
    plan.to_yaml(path)


def _write_data(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "path: dataset",
                "train: images/train",
                "val: images/val",
                "names:",
                "  - defect",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_template(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "nc: 80",
                "scales:",
                "  n: [0.50, 0.25, 1024]",
                "backbone:",
                "  - [-1, 1, Conv, [64, 3, 2]]",
                "head:",
                "  - [-1, 1, Detect, [nc]]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_smoke_runner_skips_ultralytics_import_when_missing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Missing ultralytics should skip framework checks without failing."""
    plan_path = tmp_path / "plan.yaml"
    data_path = tmp_path / "data.yaml"
    template_path = tmp_path / "template.yaml"
    _write_plan(plan_path)
    _write_data(data_path)
    _write_template(template_path)
    monkeypatch.setattr(smoke_runner, "_import_ultralytics", lambda: None)

    result = SmokeRunner(EvidenceStore(tmp_path / "runs")).run(
        plan_path=plan_path,
        data_path=data_path,
        run_id="smoke-test",
        base_template=template_path,
    )
    evidence = EvidenceStore(tmp_path / "runs").load_run("smoke-test")

    assert result.status == "skipped"
    assert result.candidates[0].status == "skipped"
    assert result.candidates[0].yaml_generated is True
    assert result.candidates[0].ultralytics_imported is False
    assert result.candidates[0].forward_checked is False
    assert evidence.metrics["skipped"] == 1
    assert "yolo11n_baseline_n.yaml" in evidence.artifacts
    guard_metrics = {record.metric_name: record for record in evidence.metric_records}
    assert guard_metrics["smoke_passed"].value is False
    assert guard_metrics["yaml_generated"].value is True
    assert guard_metrics["ultralytics_imported"].value is False
    assert guard_metrics["forward_checked"].value is False
    assert guard_metrics["smoke_passed"].candidate_id == "yolo11n_baseline_n"
    assert guard_metrics["smoke_passed"].node_id == "node_yolo11n_baseline_n"
    assert guard_metrics["smoke_passed"].split == "guard"
    assert guard_metrics["smoke_passed"].validator == "SmokeRunner"
    assert guard_metrics["smoke_passed"].metric_schema_version == "smoke_guard.v1"


def test_smoke_runner_try_forward_uses_mocked_ultralytics(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """try-forward should call YOLO(...).info() when ultralytics is available."""
    plan_path = tmp_path / "plan.yaml"
    data_path = tmp_path / "data.yaml"
    template_path = tmp_path / "template.yaml"
    calls: list[str] = []
    _write_plan(plan_path)
    _write_data(data_path)
    _write_template(template_path)

    class FakeModel:
        def __init__(self, model_path: str) -> None:
            calls.append(model_path)

        def info(self) -> None:
            calls.append("info")

    monkeypatch.setattr(smoke_runner, "_import_ultralytics", lambda: SimpleNamespace(YOLO=FakeModel))

    result = SmokeRunner(EvidenceStore(tmp_path / "runs")).run(
        plan_path=plan_path,
        data_path=data_path,
        run_id="smoke-forward",
        base_template=template_path,
        try_forward=True,
    )

    assert result.status == "passed"
    assert result.ultralytics_available is True
    assert result.candidates[0].yaml_generated is True
    assert result.candidates[0].ultralytics_imported is True
    assert result.candidates[0].forward_checked is True
    assert calls[-1] == "info"
    evidence = EvidenceStore(tmp_path / "runs").load_run("smoke-forward")
    guard_values = {record.metric_name: record.value for record in evidence.metric_records}
    assert guard_values == {
        "smoke_passed": True,
        "yaml_generated": True,
        "ultralytics_imported": True,
        "forward_checked": True,
    }


def test_smoke_cli_writes_evidence(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The smoke CLI should run and write EvidenceStore output."""
    plan_path = tmp_path / "plan.yaml"
    data_path = tmp_path / "data.yaml"
    template_path = tmp_path / "template.yaml"
    _write_plan(plan_path)
    _write_data(data_path)
    _write_template(template_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(smoke_runner, "_import_ultralytics", lambda: None)

    exit_code = main(
        [
            "smoke",
            "--plan",
            str(plan_path),
            "--data",
            str(data_path),
            "--base-template",
            str(template_path),
            "--run-id",
            "smoke-cli",
        ]
    )

    assert exit_code == 0
    metrics = yaml.safe_load((tmp_path / "runs" / "smoke-cli" / "config.yaml").read_text(encoding="utf-8"))
    assert metrics["run_id"] == "smoke-cli"
