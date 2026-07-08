"""Process probe and termination tests."""

from __future__ import annotations

import types

from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.process_probe import (
    ProcessProbeResult,
    terminate_command_process,
    terminate_run_processes,
)


def test_terminate_command_process_uses_windows_process_tree(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Windows termination should kill the process tree, not only the parent shell."""
    seen: dict[str, object] = {}
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name="exp_node",
    )

    monkeypatch.setattr(
        "yolo_agent.core.process_probe.probe_command_process",
        lambda command: ProcessProbeResult(status="found", pid=1234, name="python.exe", detail="pid=1234"),
    )
    monkeypatch.setattr("yolo_agent.core.process_probe.platform.system", lambda: "Windows")

    def fake_run(argv: list[str], **kwargs: object) -> object:
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return types.SimpleNamespace(returncode=0, stdout="SUCCESS", stderr="")

    monkeypatch.setattr("yolo_agent.core.process_probe.subprocess.run", fake_run)

    result = terminate_command_process(command)

    assert result.terminated is True
    assert result.pid == 1234
    assert seen["argv"] == ["taskkill", "/PID", "1234", "/T", "/F"]
    assert seen["kwargs"]["encoding"] == "utf-8"
    assert seen["kwargs"]["errors"] == "replace"


def test_terminate_run_processes_matches_outer_optimize_process(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Run-level stop should catch the outer yolo-agent optimize process too."""
    killed: list[int] = []

    monkeypatch.setattr("yolo_agent.core.process_probe.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "yolo_agent.core.process_probe._windows_processes",
        lambda: [
            {
                "pid": 111,
                "name": "yolo-agent.exe",
                "command_line": "yolo-agent optimize coco --run-id coco-yolo26n --execute",
            },
            {
                "pid": 222,
                "name": "python.exe",
                "command_line": "yolo detect train name=coco-yolo26n_node_yolo26n_coco_pilot",
            },
            {
                "pid": 333,
                "name": "powershell.exe",
                "command_line": "powershell yolo-agent loop stop --run runs\\coco-yolo26n",
            },
            {
                "pid": 444,
                "name": "yolo-agent.exe",
                "command_line": "yolo-agent loop stop --run runs\\coco-yolo26n",
            },
        ],
    )

    def fake_run(argv: list[str], **kwargs: object) -> object:
        killed.append(int(argv[2]))
        return types.SimpleNamespace(returncode=0, stdout="SUCCESS", stderr="")

    monkeypatch.setattr("yolo_agent.core.process_probe.subprocess.run", fake_run)

    results = terminate_run_processes("coco-yolo26n")

    assert [result.pid for result in results] == [111, 222]
    assert killed == [111, 222]
