"""CLI smoke tests for the yolo-agent scaffold."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.cli import COMMANDS, USER_COMMANDS, main


def test_cli_import() -> None:
    """The CLI module should import and expose scaffold commands."""
    assert "init" in COMMANDS
    assert "report" in COMMANDS
    assert "optimize" in COMMANDS
    assert "doctor" in COMMANDS
    assert USER_COMMANDS == ("train", "status", "stop", "doctor", "setup")


def test_cli_help_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    """Running without a command should print help and succeed."""
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "Componentized YOLO optimization harness" in output
    assert "{train,status,stop,doctor,setup}" in output
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
    assert f"Status:   yolo-agent status --run {tmp_path / 'runs' / 'cli-train'}" in output
