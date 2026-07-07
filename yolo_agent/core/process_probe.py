"""Best-effort local process probes for queued training commands."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Literal

from pydantic import BaseModel

from yolo_agent.core.command_spec import CommandSpec


ProcessProbeStatus = Literal["found", "not_found", "unknown"]


class ProcessProbeResult(BaseModel):
    """Result of checking whether a command appears to be running locally."""

    status: ProcessProbeStatus
    detail: str = ""
    pid: int | None = None
    name: str = ""


def probe_command_process(command: CommandSpec) -> ProcessProbeResult:
    """Return whether a command appears in the local process table."""
    marker = _command_marker(command)
    if not marker:
        return ProcessProbeResult(status="unknown", detail="no stable command marker")
    try:
        processes = _windows_processes() if platform.system().lower() == "windows" else _posix_processes()
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return ProcessProbeResult(status="unknown", detail=f"process probe failed: {exc}")
    marker_lower = marker.lower()
    for process in processes:
        command_line = str(process.get("command_line") or "")
        name = str(process.get("name") or "")
        pid = _int_or_none(process.get("pid"))
        if pid == os.getpid():
            continue
        if name.lower() in {"powershell.exe", "pwsh.exe", "cmd.exe"}:
            continue
        if marker_lower in command_line.lower():
            return ProcessProbeResult(
                status="found",
                detail=f"pid={pid} name={name}",
                pid=pid,
                name=name,
            )
    return ProcessProbeResult(status="not_found", detail=f"no process matched marker {marker!r}")


def _command_marker(command: CommandSpec) -> str:
    for arg in command.argv:
        if arg.startswith("name="):
            return arg.split("=", 1)[1]
    for key in ("node_id", "candidate_id"):
        value = command.metadata.get(key)
        if value:
            return str(value)
    return ""


def _windows_processes() -> list[dict[str, object]]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,Name,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.SubprocessError(completed.stderr.strip() or "Win32_Process query failed")
    raw = json.loads(completed.stdout or "[]")
    if isinstance(raw, dict):
        raw = [raw]
    return [
        {
            "pid": item.get("ProcessId"),
            "name": item.get("Name"),
            "command_line": item.get("CommandLine"),
        }
        for item in raw
        if isinstance(item, dict)
    ]


def _posix_processes() -> list[dict[str, object]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.SubprocessError(completed.stderr.strip() or "ps query failed")
    processes: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        processes.append({"pid": parts[0], "name": parts[1], "command_line": parts[2]})
    return processes


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
