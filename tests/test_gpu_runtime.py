"""GPU process inspection and ownership tests."""

from __future__ import annotations

from dataclasses import dataclass

from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.gpu_runtime import GPURuntimeSnapshot, GPUProcessInfo, inspect_gpu_runtime


@dataclass
class _Result:
    stdout: str
    returncode: int = 0
    stderr: str = ""


def _command() -> CommandSpec:
    return CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name="improve-map-r10_baseline",
        batch=48,
        device=0,
    )


def test_gpu_snapshot_finds_external_python_pressure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    results = iter(
        [
            _Result("8325, 24564\n"),
            _Result("15584, python.exe, [N/A]\n3780, explorer.exe, [N/A]\n"),
        ]
    )
    monkeypatch.setattr(
        "yolo_agent.core.gpu_runtime._process_command_line",
        lambda pid: "python E:/codex/scene_gen/scripts/indextts-worker.py" if pid == 15584 else "",
    )

    snapshot = inspect_gpu_runtime(_command(), runner=lambda *args, **kwargs: next(results))

    assert snapshot.used_memory_mb == 8325
    assert snapshot.free_memory_mb == 16239
    assert [process.pid for process in snapshot.external_processes] == [15584]
    assert snapshot.has_external_training_conflict is True


def test_gpu_snapshot_marks_only_exact_run_command_as_owned(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    results = iter(
        [
            _Result("7000, 24564\n"),
            _Result("111, python.exe, [N/A]\n222, python.exe, [N/A]\n"),
        ]
    )
    commands = {
        111: "python train.py project=runs/ultralytics name=improve-map-r10_baseline",
        222: "python E:/codex/scene_gen/scripts/indextts-worker.py",
    }
    monkeypatch.setattr(
        "yolo_agent.core.gpu_runtime._process_command_line",
        lambda pid: commands[pid],
    )

    snapshot = inspect_gpu_runtime(_command(), runner=lambda *args, **kwargs: next(results))

    assert snapshot.processes[0].belongs_to_run is True
    assert snapshot.processes[1].belongs_to_run is False


def test_gpu_snapshot_treats_invoking_agent_as_run_owned(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    results = iter(
        [
            _Result("9800, 24564\n"),
            _Result("40152, python.exe, [N/A]\n34020, python.exe, [N/A]\n"),
        ]
    )
    commands = {
        40152: "python yolo-agent.exe train --run-id improve-map-5-v3",
        34020: "python E:/codex/scene_gen/scripts/indextts-worker.py",
    }
    monkeypatch.setattr("yolo_agent.core.gpu_runtime.os.getpid", lambda: 40152)
    monkeypatch.setattr(
        "yolo_agent.core.gpu_runtime._process_command_line",
        lambda pid: commands[pid],
    )

    snapshot = inspect_gpu_runtime(_command(), runner=lambda *args, **kwargs: next(results))

    assert snapshot.processes[0].belongs_to_run is True
    assert snapshot.processes[1].belongs_to_run is False
    assert [process.pid for process in snapshot.external_processes] == [34020]


def test_low_desktop_gpu_usage_is_not_a_training_conflict() -> None:
    snapshot = GPURuntimeSnapshot(
        used_memory_mb=1595,
        total_memory_mb=24564,
        processes=[GPUProcessInfo(pid=1, process_name="python.exe")],
    )

    assert snapshot.has_external_training_conflict is False
