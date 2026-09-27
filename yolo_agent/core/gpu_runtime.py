"""Best-effort GPU occupancy inspection for safe training recovery."""

from __future__ import annotations

import csv
import os
import subprocess
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from yolo_agent.core.command_spec import CommandSpec


class GPUProcessInfo(BaseModel):
    """One process reported by the NVIDIA driver."""

    pid: int
    process_name: str
    command_line: str = ""
    used_memory_mb: int | None = None
    belongs_to_run: bool = False


class GPURuntimeSnapshot(BaseModel):
    """Current memory pressure and relevant compute processes for one GPU."""

    device: int = 0
    used_memory_mb: int | None = None
    total_memory_mb: int | None = None
    processes: list[GPUProcessInfo] = Field(default_factory=list)
    inspection_error: str = ""

    @property
    def free_memory_mb(self) -> int | None:
        if self.used_memory_mb is None or self.total_memory_mb is None:
            return None
        return max(0, self.total_memory_mb - self.used_memory_mb)

    @property
    def external_processes(self) -> list[GPUProcessInfo]:
        return [process for process in self.processes if not process.belongs_to_run]

    @property
    def has_external_training_conflict(self) -> bool:
        """True only when *external* GPU pressure genuinely blocks training.

        Memory attribution rules, in order:

        1. No external compute processes -> no conflict.
        2. Every external process reports its own GPU memory -> decide on the
           sum of external memory only.  Total ``used_memory_mb`` also counts
           run-owned processes (the active training child, or an orphan from a
           dead CLI), which stale-run recovery terminates before launch;
           attributing their memory to "external" produced permanent false
           conflicts.
        3. External memory unknown but run-owned processes with unknown memory
           exist -> the total cannot be attributed to anyone, and every launch
           path terminates claimed processes and re-checks before running, so
           do not block here.
        4. Otherwise (no claimable process to clean up) fall back to the total
           used-memory heuristic, fail-closed.
        """
        externals = self.external_processes
        if not externals:
            return False
        external_memory = [process.used_memory_mb for process in externals]
        if all(memory is not None for memory in external_memory):
            used_memory_mb: int | None = sum(memory or 0 for memory in external_memory)
        elif any(
            process.belongs_to_run and process.used_memory_mb is None
            for process in self.processes
        ):
            return False
        else:
            used_memory_mb = self.used_memory_mb
        if used_memory_mb is None or self.total_memory_mb in {None, 0}:
            return True
        return used_memory_mb >= 4096 and used_memory_mb / self.total_memory_mb >= 0.25


def inspect_gpu_runtime(
    command: CommandSpec,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> GPURuntimeSnapshot:
    """Inspect one GPU without requiring NVML Python bindings."""
    device = _device_index(command)
    try:
        memory_result = runner(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if memory_result.returncode != 0:
            return GPURuntimeSnapshot(
                device=device,
                inspection_error=(memory_result.stderr or "nvidia-smi returned an error").strip(),
            )
        used_memory_mb, total_memory_mb = _parse_memory(memory_result.stdout)
        process_result = runner(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        processes = (
            _parse_processes(process_result.stdout, command)
            if process_result.returncode == 0
            else []
        )
        return GPURuntimeSnapshot(
            device=device,
            used_memory_mb=used_memory_mb,
            total_memory_mb=total_memory_mb,
            processes=processes,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return GPURuntimeSnapshot(device=device, inspection_error=str(exc))


def terminate_stale_run_processes(snapshot: GPURuntimeSnapshot) -> list[int]:
    """Terminate only processes whose command line exactly identifies this run."""
    terminated: list[int] = []
    try:
        import psutil
    except ImportError:
        return terminated
    for process_info in snapshot.processes:
        if not process_info.belongs_to_run or process_info.pid == os.getpid():
            continue
        try:
            process = psutil.Process(process_info.pid)
            descendants = process.children(recursive=True)
            for child in reversed(descendants):
                child.kill()
            process.kill()
            psutil.wait_procs([*descendants, process], timeout=3)
            terminated.append(process_info.pid)
        except (psutil.Error, OSError):
            continue
    return terminated


def _parse_memory(output: str) -> tuple[int, int]:
    row = next(csv.reader(line for line in output.splitlines() if line.strip()), None)
    if row is None or len(row) < 2:
        raise ValueError("nvidia-smi did not return GPU memory values")
    return int(row[0].strip()), int(row[1].strip())


def _parse_processes(output: str, command: CommandSpec) -> list[GPUProcessInfo]:
    processes: list[GPUProcessInfo] = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) < 2:
            continue
        try:
            pid = int(row[0].strip())
        except ValueError:
            continue
        process_name = row[1].strip()
        command_line = _process_command_line(pid)
        if not _is_relevant_compute_process(process_name, command_line):
            continue
        used_memory = None
        if len(row) >= 3 and row[2].strip().isdigit():
            used_memory = int(row[2].strip())
        processes.append(
            GPUProcessInfo(
                pid=pid,
                process_name=process_name,
                command_line=command_line,
                used_memory_mb=used_memory,
                belongs_to_run=(
                    _belongs_to_current_process_tree(pid)
                    or _belongs_to_command(command_line, command)
                    or _belongs_to_workspace(command_line)
                ),
            )
        )
    return processes


def _process_command_line(pid: int) -> str:
    try:
        import psutil

        return subprocess.list2cmdline(psutil.Process(pid).cmdline())
    except (ImportError, OSError):
        return ""
    except Exception as exc:  # psutil exceptions are optional at import time.
        if exc.__class__.__module__.startswith("psutil"):
            return ""
        raise


def _is_relevant_compute_process(process_name: str, command_line: str) -> bool:
    text = f"{process_name} {command_line}".lower()
    return any(
        marker in text
        for marker in ("python", "yolo", "ultralytics", "torchrun", "triton", "indextts")
    )


def _belongs_to_command(command_line: str, command: CommandSpec) -> bool:
    if not command_line:
        return False
    project = _arg_value(command, "project")
    name = _arg_value(command, "name")
    if not project or not name:
        return False
    normalized = command_line.replace("\\", "/").lower()
    project_marker = f"project={project}".replace("\\", "/").lower()
    name_marker = f"name={name}".lower()
    return project_marker in normalized and name_marker in normalized


def _belongs_to_current_process_tree(pid: int) -> bool:
    """Treat the invoking yolo-agent process and its ancestors as run-owned."""
    current_pid = os.getpid()
    if pid == current_pid:
        return True
    try:
        import psutil

        return pid in {parent.pid for parent in psutil.Process(current_pid).parents()}
    except (ImportError, OSError):
        return False
    except Exception as exc:  # psutil exceptions are optional at import time.
        if exc.__class__.__module__.startswith("psutil"):
            return False
        raise


_WORKSPACE_PROCESS_MARKERS = (
    "yolo_agent",  # adapter runtime entrypoint: project/name live in the payload file
    "runtime_entrypoint",
    "runs/ultralytics",  # this workspace's ultralytics output project
)


def _belongs_to_workspace(command_line: str) -> bool:
    """Claim processes launched for this workspace's training pipeline.

    Paper-candidate training runs through the adapter runtime entrypoint,
    whose command line carries no ``project=``/``name=`` literals (they live
    inside the payload YAML).  A CLI that died mid-run therefore left its
    training child behind as permanent ``unrelated`` GPU pressure: every new
    invocation blocked on it until the orphan finished by itself.  Command
    lines identifying this workspace's own training pipeline are run-owned,
    so stale-run recovery can terminate them instead of waiting on them.
    """
    if not command_line:
        return False
    normalized = command_line.replace("\\", "/").lower()
    return any(marker in normalized for marker in _WORKSPACE_PROCESS_MARKERS)


def _device_index(command: CommandSpec) -> int:
    raw = _arg_value(command, "device") or str(command.resource_requirements.preferred_gpu_id or 0)
    try:
        return int(raw.split(",", 1)[0])
    except ValueError:
        return 0


def _arg_value(command: CommandSpec, key: str) -> str | None:
    for item in command.argv or [command.command, *command.args]:
        if item.startswith(f"{key}="):
            return item.split("=", 1)[1]
    return None
