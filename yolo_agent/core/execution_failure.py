"""Structured classification and recovery for execution infrastructure failures."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.core.command_spec import CommandSpec


FailureKind = Literal["gpu_memory_exhausted", "host_memory_exhausted", "unknown"]


class ExecutionFailure(BaseModel):
    """One classified execution failure and its bounded recovery action."""

    kind: FailureKind
    summary: str
    root_cause: str
    recoverable: bool = False
    recovery_attempt: int = 0
    max_recovery_attempts: int = 2
    failed_settings: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    recovery_overrides: dict[str, str | int | float | bool] = Field(default_factory=dict)
    evidence_patterns: list[str] = Field(default_factory=list)


def classify_execution_failure(
    *,
    stdout: str,
    stderr: str,
    command: CommandSpec,
) -> ExecutionFailure | None:
    """Classify known infrastructure failures from merged process output."""
    output = f"{stdout}\n{stderr}"
    lowered = output.lower()
    attempt = _positive_int(command.metadata.get("resource_recovery_attempt"), default=0)
    workers = _positive_int(_arg_value(command, "workers"), default=8, allow_zero=True)
    batch = _positive_int(_arg_value(command, "batch"), default=None)
    cache = _arg_value(command, "cache")
    failed_settings = {"batch": batch, "workers": workers, "cache": cache}
    cuda_patterns = [
        pattern
        for pattern in (
            "CUDA error: out of memory" if "cuda error: out of memory" in lowered else "",
            "CUDA out of memory" if "cuda out of memory" in lowered else "",
            "torch.cuda.OutOfMemoryError" if "torch.cuda.outofmemoryerror" in lowered else "",
            "cudaErrorMemoryAllocation" if "cudaerrormemoryallocation" in lowered else "",
        )
        if pattern
    ]
    if cuda_patterns:
        overrides: dict[str, str | int | float | bool] = {}
        if attempt < 2 and batch is not None and batch > 1:
            overrides["batch"] = max(1, batch // 2)
        return ExecutionFailure(
            kind="gpu_memory_exhausted",
            summary="GPU memory was exhausted during Ultralytics training or validation.",
            root_cause=(
                "CUDA could not allocate enough VRAM for the current batch. "
                "This is an execution-resource failure, not a candidate quality result."
            ),
            recoverable=attempt < 2 and bool(overrides),
            recovery_attempt=attempt,
            failed_settings=failed_settings,
            recovery_overrides=overrides,
            evidence_patterns=cuda_patterns,
        )
    patterns = [
        pattern
        for pattern in (
            "_ArrayMemoryError" if "_arraymemoryerror" in lowered else "",
            "DataLoader worker" if "dataloader worker" in lowered else "",
            "Unable to allocate" if "unable to allocate" in lowered else "",
            "cv2.warpAffine" if "warpaffine" in lowered else "",
        )
        if pattern
    ]
    host_memory_failure = (
        "_arraymemoryerror" in lowered
        or (
            "dataloader worker" in lowered
            and ("unable to allocate" in lowered or "not enough memory" in lowered)
        )
    )
    if not host_memory_failure:
        return None

    overrides: dict[str, str | int | float | bool] = {}
    if attempt < 2:
        if attempt == 0:
            overrides["workers"] = min(2, workers) if workers > 0 else 0
        else:
            overrides["workers"] = 0
            if batch is not None and batch > 1:
                overrides["batch"] = max(1, batch // 2)

    recoverable = attempt < 2 and bool(overrides)
    return ExecutionFailure(
        kind="host_memory_exhausted",
        summary="System RAM was exhausted in an Ultralytics DataLoader worker.",
        root_cause=(
            "NumPy could not allocate an augmented image buffer; OpenCV then failed in warpAffine. "
            "This is a host-memory/input-pipeline failure, not a measured model regression."
        ),
        recoverable=recoverable,
        recovery_attempt=attempt,
        failed_settings=failed_settings,
        recovery_overrides=overrides,
        evidence_patterns=patterns,
    )


def apply_execution_recovery(command: CommandSpec, failure: ExecutionFailure) -> CommandSpec:
    """Apply one classified recovery without changing the method recipe."""
    if not failure.recoverable or not failure.recovery_overrides:
        return command
    updated = _upsert_args(command, failure.recovery_overrides)
    metadata = {
        **updated.metadata,
        "resource_recovery_attempt": failure.recovery_attempt + 1,
        "resource_recovery_kind": failure.kind,
        "resource_recovery_is_infrastructure_only": True,
        "resource_recovery_excluded_from_model_evidence": True,
        "resource_recovery_original_settings": json.dumps(failure.failed_settings, sort_keys=True),
        "resource_recovery_overrides": json.dumps(failure.recovery_overrides, sort_keys=True),
    }
    if "batch" in failure.recovery_overrides:
        metadata["batch_tuned"] = True
        metadata["batch_tuning_selected_batch"] = int(failure.recovery_overrides["batch"])
    return updated.model_copy(update={"metadata": metadata})


def resource_policy_cache_key(command: CommandSpec) -> str:
    """Return a machine-local identity for a proven host-memory-safe policy."""
    payload = {
        "schema": "host_memory_policy.v1",
        "machine": {
            "node": platform.node(),
            "system": platform.system(),
            "processor": platform.processor(),
        },
        "command": {
            "model": _arg_value(command, "model"),
            "data": _arg_value(command, "data"),
            "imgsz": _arg_value(command, "imgsz"),
            "device": _arg_value(command, "device"),
            "cache": _arg_value(command, "cache"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_cached_resource_policy(command: CommandSpec) -> tuple[CommandSpec, bool]:
    """Apply a previously proven host-memory cap after normal batch tuning."""
    path = _resource_policy_cache_path()
    data = _read_policy_cache(path)
    raw = data.get(resource_policy_cache_key(command))
    if not isinstance(raw, dict):
        return command, False
    updates: dict[str, str | int | float | bool] = {}
    safe_workers = _positive_int(raw.get("workers"), default=None, allow_zero=True)
    current_workers = _positive_int(_arg_value(command, "workers"), default=None, allow_zero=True)
    if safe_workers is not None and (current_workers is None or current_workers > safe_workers):
        updates["workers"] = safe_workers
    safe_batch = _positive_int(raw.get("batch"), default=None)
    current_batch = _positive_int(_arg_value(command, "batch"), default=None)
    if safe_batch is not None and current_batch is not None and current_batch > safe_batch:
        updates["batch"] = safe_batch
    if not updates:
        return command, False
    updated = _upsert_args(command, updates)
    metadata = {
        **updated.metadata,
        "host_memory_policy_applied": True,
        "host_memory_policy_cache_key": resource_policy_cache_key(command),
        "host_memory_policy_overrides": json.dumps(updates, sort_keys=True),
    }
    if "batch" in updates:
        metadata["batch_tuned"] = True
        metadata["batch_tuning_selected_batch"] = int(updates["batch"])
    return updated.model_copy(update={"metadata": metadata}), True


def save_successful_resource_policy(command: CommandSpec) -> Path | None:
    """Persist settings only after a recovered training command completes."""
    if _positive_int(command.metadata.get("resource_recovery_attempt"), default=0) <= 0:
        return None
    workers = _positive_int(_arg_value(command, "workers"), default=None, allow_zero=True)
    batch = _positive_int(_arg_value(command, "batch"), default=None)
    if workers is None:
        return None
    path = _resource_policy_cache_path()
    data = _read_policy_cache(path)
    data[resource_policy_cache_key(command)] = {
        "schema": "host_memory_policy.v1",
        "workers": workers,
        "batch": batch,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _resource_policy_cache_path() -> Path:
    override = os.environ.get("YOLO_AGENT_RESOURCE_POLICY_CACHE")
    return Path(override) if override else Path.home() / ".yolo_agent" / "host_memory_policy_cache.json"


def _read_policy_cache(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _upsert_args(command: CommandSpec, updates: dict[str, str | int | float | bool]) -> CommandSpec:
    argv = list(command.argv or [command.command, *command.args])
    seen: set[str] = set()
    updated_argv: list[str] = []
    for item in argv:
        key = item.split("=", 1)[0] if "=" in item else ""
        if key in updates:
            updated_argv.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            updated_argv.append(item)
    for key, value in updates.items():
        if key not in seen:
            updated_argv.append(f"{key}={value}")
    return command.model_copy(
        update={"command": updated_argv[0], "args": updated_argv[1:], "argv": updated_argv}
    )


def _arg_value(command: CommandSpec, key: str) -> str | None:
    for item in command.argv or [command.command, *command.args]:
        if item.startswith(f"{key}="):
            return item.split("=", 1)[1]
    return None


def _positive_int(value: object, *, default: int | None, allow_zero: bool = False) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    if parsed > 0 or (allow_zero and parsed == 0):
        return parsed
    return default
