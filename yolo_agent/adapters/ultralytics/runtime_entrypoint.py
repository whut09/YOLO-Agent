"""Isolated entrypoint for component-enabled Ultralytics commands."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import yaml

from yolo_agent.components.adapters.runtime import AdapterRuntimePayload


def run_payload(payload_path: Path | str, command: list[str]) -> int:
    """Validate plugins and execute the command without framework source patches."""
    payload = AdapterRuntimePayload.read(payload_path, verify_imports=True)
    actual_command = list(command or payload.base_command)
    if not actual_command:
        raise ValueError("runtime entrypoint requires an executable command")
    env = {**os.environ, **payload.env}
    env["YOLO_AGENT_RUNTIME_PAYLOAD"] = str(Path(payload_path).resolve())
    if _is_ultralytics_train_command(actual_command):
        return run_ultralytics_training(payload_path, actual_command, env=env)
    if _has_training_plugins(payload):
        raise ValueError(
            "training adapter payload requires a 'yolo detect train' command; "
            "refusing uninstrumented subprocess fallback"
        )
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


def run_ultralytics_training(
    payload_path: Path | str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> int:
    """Execute detect training through the stable local plugin Trainer class."""
    from ultralytics import YOLO

    from yolo_agent.adapters.ultralytics.plugin_bridge import (
        PluginDetectionTrainer,
        UltralyticsTrainerPluginBridge,
    )

    runtime_env = {**os.environ, **(env or {})}
    runtime_env["YOLO_AGENT_RUNTIME_PAYLOAD"] = str(Path(payload_path).resolve())
    os.environ.update(runtime_env)
    bridge = UltralyticsTrainerPluginBridge(payload_path)
    try:
        prepared_command, prepared_env = bridge.prepare_command(command, runtime_env)
        os.environ.update(prepared_env)
        task, arguments = parse_ultralytics_train_command(prepared_command)
        bridge.validate_training_args(arguments)
        model_value = arguments.pop("model", None)
        if model_value in {None, ""}:
            raise ValueError("Ultralytics plugin training requires model=<checkpoint-or-yaml>")
        model = YOLO(model_value, task=task)
        model.train(trainer=PluginDetectionTrainer, **arguments)
        bridge.verify_required_hooks()
    except Exception as exc:
        bridge.context.record_failure("runtime_entrypoint", "train", exc)
        try:
            _write_runtime_failure_artifact(payload_path, bridge, exc)
        except OSError:
            # Preserve the original adapter exception if artifact storage is unavailable.
            pass
        raise
    return 0


def _write_runtime_failure_artifact(
    payload_path: Path | str,
    bridge: Any,
    error: Exception,
) -> Path:
    """Persist a structured adapter failure for queue-level candidate isolation."""
    payload = bridge.payload
    output = Path(payload_path).with_name("adapter_runtime_failure.json")
    message = str(error)
    artifact = {
        "schema_version": "adapter_runtime_failure.v1",
        "component_ids": list(payload.component_ids),
        "plugin": message.split(":", 2)[1] if message.startswith("plugin hook failed:") else None,
        "exception_type": type(error).__name__,
        "message": message,
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "payload_hash": payload.payload_hash,
        "protocol_hash": payload.protocol_hash,
        "payload_path": str(Path(payload_path).resolve()),
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def parse_ultralytics_train_command(command: list[str]) -> tuple[str, dict[str, Any]]:
    """Parse the typed CLI command without discarding original train overrides."""
    if not _is_ultralytics_train_command(command):
        raise ValueError("expected 'yolo detect train key=value ...'")
    task = command[1]
    arguments: dict[str, Any] = {}
    for token in command[3:]:
        key, separator, raw_value = token.partition("=")
        if not separator or not key:
            raise ValueError(f"unsupported positional Ultralytics train argument: {token}")
        arguments[key] = yaml.safe_load(raw_value)
    return task, arguments


def _is_ultralytics_train_command(command: list[str]) -> bool:
    if len(command) < 3:
        return False
    executable = Path(command[0]).stem.lower()
    return executable == "yolo" and command[1] == "detect" and command[2] == "train"


def _has_training_plugins(payload: AdapterRuntimePayload) -> bool:
    return any(
        (
            payload.dataloader_plugin,
            payload.trainer_plugin,
            payload.model_graph_plugin,
            payload.loss_plugin,
            payload.assigner_plugin,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint kept spawn-safe for Windows and Ultralytics DDP."""
    parser = argparse.ArgumentParser(description="Run a verified YOLO Agent adapter payload")
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        return run_payload(args.payload, command)
    except Exception as exc:
        print(
            f"adapter_runtime_failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 86


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    multiprocessing.freeze_support()
    raise SystemExit(main())


__all__ = [
    "main",
    "parse_ultralytics_train_command",
    "run_payload",
    "run_ultralytics_training",
    "_write_runtime_failure_artifact",
]
