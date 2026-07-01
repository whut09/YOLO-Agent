"""CLI smoke tests for the yolo-agent scaffold."""

from __future__ import annotations

from yolo_agent.cli import COMMANDS, main


def test_cli_import() -> None:
    """The CLI module should import and expose scaffold commands."""
    assert "init" in COMMANDS
    assert "report" in COMMANDS


def test_cli_help_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    """Running without a command should print help and succeed."""
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "Componentized YOLO optimization harness" in output


def test_scaffold_commands_run(capsys) -> None:  # type: ignore[no-untyped-def]
    """Every declared command should execute the current scaffold."""
    for command in COMMANDS:
        if command == "plan":
            continue
        assert main([command]) == 0
        output = capsys.readouterr().out
        assert f"yolo-agent {command}: scaffold ready" in output
