"""Isolated entrypoint for component-enabled Ultralytics commands."""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path
import subprocess

from yolo_agent.components.adapters.runtime import AdapterRuntimePayload


def run_payload(payload_path: Path | str, command: list[str]) -> int:
    """Validate plugins and execute the command without framework source patches."""
    payload = AdapterRuntimePayload.read(payload_path, verify_imports=True)
    actual_command = list(command or payload.base_command)
    if not actual_command:
        raise ValueError("runtime entrypoint requires an executable command")
    env = {**os.environ, **payload.env}
    env["YOLO_AGENT_RUNTIME_PAYLOAD"] = str(Path(payload_path).resolve())
    for reference in payload.plugin_references:
        plugin_type = reference.resolve()
        plugin = plugin_type(**reference.options) if isinstance(plugin_type, type) else plugin_type
        prepare = getattr(plugin, "prepare_command", None)
        if callable(prepare):
            prepared = prepare(payload=payload, command=actual_command, env=env)
            if prepared is not None:
                actual_command, env = prepared
    completed = subprocess.run(
        actual_command,
        cwd=payload.cwd,
        env=env,
        check=False,
    )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint kept spawn-safe for Windows and Ultralytics DDP."""
    parser = argparse.ArgumentParser(description="Run a verified YOLO Agent adapter payload")
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    return run_payload(args.payload, command)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    multiprocessing.freeze_support()
    raise SystemExit(main())


__all__ = ["main", "run_payload"]
