"""CLI smoke tests for the yolo-agent scaffold."""

from __future__ import annotations

from pathlib import Path

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


def test_setup_supports_coco_and_custom_without_new_top_level_commands() -> None:
    parser = build_parser()
    coco = parser.parse_args(["setup", "coco", "--data", "coco.yaml"])
    custom = parser.parse_args(["setup", "custom", "--data", "custom.yaml"])

    assert coco.setup_kind == "coco"
    assert custom.setup_kind == "custom"
    assert USER_COMMANDS == ("setup", "train", "status", "stop")
