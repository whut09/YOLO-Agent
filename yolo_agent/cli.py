"""Command line interface for yolo-agent."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from yolo_agent.core.schemas import AgentConfig


COMMANDS: tuple[str, ...] = (
    "init",
    "profile-data",
    "plan",
    "check",
    "smoke",
    "search",
    "ablate",
    "benchmark",
    "report",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="yolo-agent",
        description="Componentized YOLO optimization harness.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="yolo-agent 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command")
    for command in COMMANDS:
        command_parser = subparsers.add_parser(
            command,
            help=f"Run the {command} workflow scaffold.",
        )
        command_parser.set_defaults(handler=run_scaffold_command)

    return parser


def run_scaffold_command(args: argparse.Namespace) -> int:
    """Run a placeholder command while the harness is being built."""
    config = AgentConfig()
    print(f"yolo-agent {args.command}: scaffold ready")
    print(f"experiment_root={config.experiment_root}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the yolo-agent CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))

