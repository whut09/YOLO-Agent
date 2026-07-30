"""CLI smoke tests for the yolo-agent scaffold."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from yolo_agent.cli import COMMANDS, USER_COMMANDS, build_parser, main


def test_cli_import() -> None:
    """The CLI module should import and expose scaffold commands."""
    assert "init" in COMMANDS
    assert "report" in COMMANDS
    assert "optimize" in COMMANDS
    assert "doctor" in COMMANDS
    assert USER_COMMANDS == ("setup", "train", "status", "stop")


def test_cli_help_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    """Running without a command should print help and succeed."""
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "Componentized YOLO optimization harness" in output
    assert "{setup,train,status,stop}" in output
    assert "doctor" not in output
    assert "loop" not in output
    assert "optimize" not in output


def test_scaffold_commands_run(capsys) -> None:  # type: ignore[no-untyped-def]
    """Every declared command should execute the current scaffold."""
    for command in COMMANDS:
        if command in {
            "plan",
            "smoke",
            "profile-data",
            "advise-labels",
            "ablate-plan",
            "report",
            "loop",
            "optimize",
            "doctor",
        }:
            continue
        assert main([command]) == 0
        output = capsys.readouterr().out
        assert f"yolo-agent {command}: scaffold ready" in output


def test_train_command_runs_dry_run(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The beginner-facing train command should prepare a run without exposing optimize."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data_yaml = dataset / "coco.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train2017\nval: images/val2017\nnames: {0: person}\n",
        encoding="utf-8",
    )

    assert main(
        [
            "train",
            "--data",
            str(data_yaml),
            "--run-id",
            "cli-train",
            "--run-root",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "Starting YOLO Agent train" in output
    assert "Run: cli-train  Profile: debug  Mode: dry-run" in output
    assert "Budget: auto; stops when the first cost, evidence, or patience limit is reached" in output
    assert "Expected: 1-12 pilot experiments" in output
    assert "Limits: <= 24 GPU hours; concurrency=1" in output
    assert "Full: excluded from the automatic budget unless --confirm-full-run is explicit" in output
    assert f"Status:   yolo-agent status --run {tmp_path / 'runs' / 'cli-train'}" in output


def test_train_defaults_to_bounded_auto_optimization() -> None:
    """One-command train should select auto budget instead of promising a fixed round count."""
    args = build_parser().parse_args(["train", "--data", "data.yaml"])
    assert args.auto_rounds is None
    assert args.profile is None


def test_advanced_namespace_dispatches_hidden_compatibility_commands(capsys) -> None:  # type: ignore[no-untyped-def]
    """Advanced commands should remain available without appearing in beginner help."""
    assert main(["advanced"]) == 0
    output = capsys.readouterr().out
    assert "choose doctor, loop, optimize" in output
    args = build_parser().parse_args(["advanced", "doctor", "--data", "data.yaml"])
    assert args.advanced_args == ["doctor", "--data", "data.yaml"]


def test_advanced_gpu_certification_is_safe_by_default(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    workdir = tmp_path / "certification"
    assert main(["advanced", "certify-gpu", "--workdir", str(workdir)]) == 0
    output = capsys.readouterr().out
    assert "Status:   skipped" in output
    assert "real_gpu_execution_not_confirmed" in output
    assert (workdir / "certification_report.yaml").is_file()


def test_sampling_gpu_certification_prints_golden_path_outcome(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    report = SimpleNamespace(
        status="passed",
        stages=[
            SimpleNamespace(
                stage_id="component_runtime_certification",
                status="passed",
                metrics={},
            ),
            SimpleNamespace(
                stage_id="runtime_adapter",
                status="passed",
                metrics={
                    "train_dataloader_hook_called": True,
                    "manifest_payload_matched": True,
                },
            ),
        ],
        objective=SimpleNamespace(
            primary_metric="ap_small",
            observed_delta=0.012,
            passed=True,
            target_error_fact_deltas={"false_negative/object": 2.0},
        ),
        asha_survivor="small_object_sampling",
        failures=[],
    )

    class FakeSuite:
        def run(self, **kwargs):  # type: ignore[no-untyped-def]
            return report

    monkeypatch.setattr("yolo_agent.cli.RealGpuAcceptanceSuite", FakeSuite)

    result = main(
        [
            "advanced",
            "certify-gpu",
            "--workdir",
            str(tmp_path),
            "--recipe",
            "small_object_sampling",
            "--execute-real-gpu",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Component: passed" in output
    assert "Runtime:  hook_called=True manifest_matched=True" in output
    assert "Objective: ap_small delta=0.012 passed=True" in output
    assert "Error:    false_negative/object=+2.000000" in output
    assert "ASHA:     survivor=small_object_sampling" in output


def test_advanced_sahi_certification_is_safe_by_default(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    workdir = tmp_path / "sahi-certification"
    images = tmp_path / "images"
    annotations = tmp_path / "annotations.json"
    assert main([
        "advanced", "certify-sahi",
        "--workdir", str(workdir),
        "--model", "yolo26n.pt",
        "--images", str(images),
        "--annotations", str(annotations),
    ]) == 0
    output = capsys.readouterr().out
    assert "Status:    skipped" in output
    assert "Training:  unchanged; attribution disabled" in output
    assert (workdir / "sahi_certification_report.yaml").is_file()


def test_setup_supports_coco_and_custom_without_new_top_level_commands() -> None:
    parser = build_parser()
    coco = parser.parse_args(["setup", "coco", "--data", "coco.yaml"])
    custom = parser.parse_args(["setup", "custom", "--data", "custom.yaml"])

    assert coco.setup_kind == "coco"
    assert custom.setup_kind == "custom"
    assert USER_COMMANDS == ("setup", "train", "status", "stop")
