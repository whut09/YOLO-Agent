"""Every `yolo-agent` command shown in the docs must pass the CLI parser."""

from __future__ import annotations

import re
import shlex

import pytest

from tests.docs_inventory import DOC_PATHS, REPO_ROOT, iter_fenced_blocks, read_doc
from yolo_agent.cli import build_parser

COMMAND_RE = re.compile(r"^yolo-agent(?:\s|$)")


def _merge_bash_continuations(block: list[str]) -> list[str]:
    """Join trailing-backslash lines the way bash does before parsing."""
    merged: list[str] = []
    pending = ""
    for line in block:
        stripped = line.strip()
        if pending:
            stripped = (pending + " " + stripped).strip()
            pending = ""
        if stripped.endswith("\\"):
            pending = stripped[:-1].rstrip()
            continue
        merged.append(stripped)
    if pending:
        merged.append(pending)
    return merged


def _collect_command_lines() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in DOC_PATHS:
        base = path.relative_to(REPO_ROOT)
        for language, block in iter_fenced_blocks(read_doc(path)):
            if language not in {"powershell", "pwsh", "bash", "sh"}:
                continue
            if language in {"bash", "sh"}:
                block = _merge_bash_continuations(block)
            for offset, line in enumerate(block, 1):
                stripped = line.strip()
                if not COMMAND_RE.match(stripped):
                    continue
                if stripped.startswith("#") or "--help" in stripped:
                    continue
                found.append((str(base), offset, stripped))
    return found


def test_documented_cli_commands_exist() -> None:
    commands = _collect_command_lines()
    subcommands = {command.split()[1] for _, _, command in commands if len(command.split()) > 1}
    assert {"setup", "train", "status", "stop"} <= subcommands


_CLI_CASES = _collect_command_lines()


def _case_id(loc: str, line: int, cmd: str) -> str:
    parts = cmd.split()
    return f"{loc}:{line}:{parts[1] if len(parts) > 1 else 'bare'}"


@pytest.mark.parametrize(
    ("location", "command"),
    [(loc, cmd) for loc, _, cmd in _CLI_CASES],
    ids=[_case_id(loc, line, cmd) for loc, line, cmd in _CLI_CASES],
)
def test_documented_cli_command_parses(location: str, command: str) -> None:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:  # unbalanced quotes in the example
        raise AssertionError(f"{location}: unparseable shell line: {command!r} ({exc})") from exc
    try:
        build_parser().parse_args(tokens[1:])
    except SystemExit as exc:
        # SystemExit(0) means --help was requested; anything else is a parse failure.
        assert exc.code in (0, None), f"{location}: command rejected by parser: {command}"
