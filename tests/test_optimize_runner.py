"""One-command optimize runner tests."""

from __future__ import annotations

from pathlib import Path

import yaml

import yolo_agent.agents.optimize_runner as optimize_module
from yolo_agent.agents.optimize_runner import OptimizeRunner
from yolo_agent.agents.orchestrator import LoopOrchestrator, TrainingLoopResult
from yolo_agent.cli import COMMANDS, _print_event_progress, _run_with_event_progress, main
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueStore
from yolo_agent.core.process_probe import ProcessProbeResult, ProcessTerminateResult
from yolo_agent.core.resource_scheduler import ResourceDecision


def _make_dataset(root: Path) -> Path:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "img1.jpg").write_bytes(b"image")
    (label_dir / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "names:",
                "  0: object",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_optimize_coco_prepares_debug_queue_without_execute(tmp_path: Path) -> None:
    """optimize coco should prepare a safe debug run without starting training by default."""
    data_yaml = _make_dataset(tmp_path / "dataset")

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )

    assert result.ok is True
    assert result.executed is False
    assert result.profile == "debug"
    assert result.task_path.exists()
    assert result.experiment_plan_path.exists()
    assert result.queue_path.exists()
    assert result.report_path is not None and result.report_path.exists()
    assert result.training_loop is not None
    assert result.training_loop.stopped_reason == "next_round_blocked"
    assert result.queue_counts["completed"] == 1
    assert "Rerun with --execute" in result.next_action
    plan = yaml.safe_load(result.experiment_plan_path.read_text(encoding="utf-8-sig"))
    assert plan["metadata"]["preset"] is None
    queue = ExecutionQueue.from_yaml(result.queue_path)
    assert queue.items[0].status == "completed"
    assert queue.items[0].command.command_type == "train"
    assert queue.items[0].command.metadata["training_budget_profile"] == "debug"
    assert queue.metadata["queue_source_plan_hash"] == plan["metadata"]["plan_hash"]


def test_optimize_ctrl_c_marks_running_queue_as_needs_resume(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    """Ctrl+C should stop the progress watcher and make recovery explicit."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )
    store = ExecutionQueueStore(result.run_dir)
    queue = store.load()
    queue.items[0].mark_running()
    store.update_item(queue.items[0])

    monkeypatch.setattr(
        "yolo_agent.cli.terminate_command_process",
        lambda command: ProcessTerminateResult(terminated=True, pid=1234, detail="terminated"),
    )

    def action() -> None:
        raise KeyboardInterrupt

    try:
        _run_with_event_progress(result.run_dir, action, enabled=True)
    except KeyboardInterrupt:
        pass

    updated = store.load()
    output = capsys.readouterr().out
    assert updated.items[0].status == "needs_resume"
    assert updated.items[0].resource_blockers == ["interrupted_by_user"]
    assert "Ctrl+C received" in output
    assert "queue-refresh" in output


def test_loop_stop_marks_running_queue_and_prints_recovery(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    """loop stop should be a reliable fallback when Ctrl+C is not trusted."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )
    store = ExecutionQueueStore(result.run_dir)
    queue = store.load()
    queue.items[0].mark_running()
    store.update_item(queue.items[0])

    monkeypatch.setattr(
        "yolo_agent.cli.terminate_run_processes",
        lambda run_id: [ProcessTerminateResult(terminated=True, pid=111, name="yolo-agent.exe", detail="stopped")],
    )
    monkeypatch.setattr(
        "yolo_agent.cli.terminate_command_process",
        lambda command: ProcessTerminateResult(terminated=False, pid=None, detail="already stopped"),
    )

    code = main(["loop", "stop", "--run", str(result.run_dir)])

    updated = store.load()
    output = capsys.readouterr().out
    assert code == 0
    assert updated.items[0].status == "needs_resume"
    assert updated.items[0].resource_blockers == ["interrupted_by_user"]
    assert "stopped_processes=1" in output
    assert "marked_running_items=1" in output
    assert "queue-refresh" in output

    refresh_code = main(["loop", "queue-refresh", "--run", str(result.run_dir)])
    refreshed = store.load()
    assert refresh_code == 0
    assert refreshed.items[0].status == "queued"


def test_optimize_rebuilds_stale_queue_when_profile_changes(tmp_path: Path) -> None:
    """Changing profile for an existing run should not reuse the old completed queue."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    runner = OptimizeRunner()

    debug = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )
    debug_queue = ExecutionQueue.from_yaml(debug.queue_path)
    debug_hash = str(debug_queue.metadata["queue_source_plan_hash"])
    assert debug_queue.items[0].status == "completed"
    assert debug_queue.items[0].command.metadata["training_budget_profile"] == "debug"

    pilot = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )

    pilot_plan = yaml.safe_load(pilot.experiment_plan_path.read_text(encoding="utf-8-sig"))
    pilot_queue = ExecutionQueue.from_yaml(pilot.queue_path)
    assert pilot.profile == "pilot"
    assert pilot_queue.metadata["queue_source_plan_hash"] == pilot_plan["metadata"]["plan_hash"]
    assert pilot_queue.metadata["queue_source_plan_hash"] != debug_hash
    assert pilot_queue.items[0].command.metadata["training_budget_profile"] == "pilot"
    assert pilot_queue.items[0].command.metadata["training_budget_epochs"] == 10


def test_optimize_does_not_rewrite_plan_while_queue_is_running(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A rerun during active training should report the active queue instead of advancing profiles."""
    monkeypatch.setattr(
        optimize_module,
        "probe_command_process",
        lambda command: ProcessProbeResult(status="found", detail="pid=123 yolo.EXE", pid=123, name="yolo.EXE"),
    )
    data_yaml = _make_dataset(tmp_path / "dataset")
    runner = OptimizeRunner()
    debug = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )
    plan_before = yaml.safe_load(debug.experiment_plan_path.read_text(encoding="utf-8-sig"))
    queue = ExecutionQueue.from_yaml(debug.queue_path)
    queue.items[0].mark_running()
    queue.to_yaml(debug.queue_path)

    monkeypatch.setattr(
        optimize_module,
        "optimize_preflight",
        lambda kind, data_yaml, execute=False: [
            optimize_module.PreflightCheck(name="test_preflight", ok=True, level="info", message="ok")
        ],
    )

    result = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=True,
    )

    plan_after = yaml.safe_load(debug.experiment_plan_path.read_text(encoding="utf-8-sig"))
    queue_after = ExecutionQueue.from_yaml(debug.queue_path)
    assert result.profile == "debug"
    assert result.profile_history == ["debug"]
    assert result.training_loop is not None
    assert result.training_loop.stopped_reason == "queue_running"
    assert result.queue_counts["running"] == 1
    assert "already running" in result.next_action
    assert plan_after["metadata"]["profile"] == "debug"
    assert plan_after["metadata"]["plan_hash"] == plan_before["metadata"]["plan_hash"]
    assert queue_after.items[0].status == "running"


def test_optimize_blocks_profile_advance_when_running_queue_is_stale(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A stale debug queue should be recovered as debug before advancing to pilot."""
    monkeypatch.setattr(
        optimize_module,
        "probe_command_process",
        lambda command: ProcessProbeResult(status="not_found", detail="no matching process"),
    )
    monkeypatch.setattr(
        optimize_module,
        "optimize_preflight",
        lambda kind, data_yaml, execute=False: [
            optimize_module.PreflightCheck(name="test_preflight", ok=True, level="info", message="ok")
        ],
    )
    data_yaml = _make_dataset(tmp_path / "dataset")
    runner = OptimizeRunner()
    debug = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )
    queue = ExecutionQueue.from_yaml(debug.queue_path)
    queue.items[0].mark_running()
    queue.to_yaml(debug.queue_path)

    result = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=True,
    )

    assert result.profile == "debug"
    assert result.training_loop is not None
    assert result.training_loop.stopped_reason == "queue_stale"
    assert "Rerun optimize with --profile debug" in result.next_action
    assert yaml.safe_load(debug.experiment_plan_path.read_text(encoding="utf-8-sig"))["metadata"]["profile"] == "debug"


def test_optimize_reports_existing_blocked_profile_without_rewriting_plan(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """Rerunning debug should report an active blocked pilot queue instead of hiding the real blocker."""
    monkeypatch.setattr(
        optimize_module,
        "optimize_preflight",
        lambda kind, data_yaml, execute=False: [
            optimize_module.PreflightCheck(name="test_preflight", ok=True, level="info", message="ok")
        ],
    )
    data_yaml = _make_dataset(tmp_path / "dataset")
    runner = OptimizeRunner()
    pilot = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )
    plan_before = yaml.safe_load(pilot.experiment_plan_path.read_text(encoding="utf-8-sig"))
    queue = ExecutionQueue.from_yaml(pilot.queue_path)
    queue.items[0].mark_resource_decision(
        ResourceDecision(
            status="blocked_by_resource",
            reasons=["missing_batch_tuning_result"],
            message="Execution blocked by missing resource preparation evidence.",
        )
    )
    queue.to_yaml(pilot.queue_path)

    result = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=True,
    )

    plan_after = yaml.safe_load(pilot.experiment_plan_path.read_text(encoding="utf-8-sig"))
    assert result.profile == "pilot"
    assert result.training_loop is not None
    assert result.training_loop.stopped_reason == "queue_blocked"
    assert result.queue_counts["blocked_by_resource"] == 1
    assert "batch tuning" in result.next_action
    assert plan_after["metadata"]["profile"] == plan_before["metadata"]["profile"]
    assert plan_after["metadata"]["plan_hash"] == plan_before["metadata"]["plan_hash"]


def test_optimize_advance_reuses_existing_run_context(tmp_path: Path) -> None:
    """advance should move an existing run to a new profile without restating data/model."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    runner = OptimizeRunner()
    debug = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )
    debug_queue = ExecutionQueue.from_yaml(debug.queue_path)

    pilot = runner.advance(
        run_dir=tmp_path / "runs" / "coco-yolo26n",
        to_profile="pilot",
        execute=False,
    )

    pilot_plan = yaml.safe_load(pilot.experiment_plan_path.read_text(encoding="utf-8-sig"))
    pilot_queue = ExecutionQueue.from_yaml(pilot.queue_path)
    assert pilot.profile == "pilot"
    assert pilot.executed is False
    assert pilot_plan["metadata"]["model"] == "yolo26n.pt"
    assert pilot_plan["metadata"]["data_yaml"] == data_yaml.as_posix()
    assert pilot_queue.metadata["queue_source_plan_hash"] != debug_queue.metadata["queue_source_plan_hash"]
    assert pilot_queue.items[0].command.metadata["training_budget_profile"] == "pilot"


def test_optimize_execute_auto_advances_debug_to_pilot(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A successful debug execution should automatically continue to pilot."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    calls: list[str] = []

    monkeypatch.setattr(
        optimize_module,
        "optimize_preflight",
        lambda kind, data_yaml, execute=False: [
            optimize_module.PreflightCheck(name="test_preflight", ok=True, level="info", message="ok")
        ],
    )

    def fake_training_loop(
        self: LoopOrchestrator,
        profile: str,
        executor: str,
        max_steps: int = 8,
        auto_import: bool = True,
    ) -> TrainingLoopResult:
        calls.append(profile)
        return TrainingLoopResult(
            run_id=self.context.run_id,
            profile=profile,
            executor=executor,
            auto_import=auto_import,
            max_steps=max_steps,
            steps=[],
            queue_counts={"completed": 1},
            stopped_reason="next_round_blocked",
            completed=True,
        )

    monkeypatch.setattr(LoopOrchestrator, "run_training_loop", fake_training_loop)

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=True,
    )

    assert calls == ["debug", "pilot"]
    assert result.profile == "pilot"
    assert result.profile_history == ["debug", "pilot"]
    assert "Full COCO is blocked" in result.next_action
    plan = yaml.safe_load(result.experiment_plan_path.read_text(encoding="utf-8-sig"))
    assert plan["metadata"]["profile"] == "pilot"


def test_optimize_execute_can_disable_auto_advance(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Users should still be able to stop after the requested profile."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    calls: list[str] = []

    monkeypatch.setattr(
        optimize_module,
        "optimize_preflight",
        lambda kind, data_yaml, execute=False: [
            optimize_module.PreflightCheck(name="test_preflight", ok=True, level="info", message="ok")
        ],
    )

    def fake_training_loop(
        self: LoopOrchestrator,
        profile: str,
        executor: str,
        max_steps: int = 8,
        auto_import: bool = True,
    ) -> TrainingLoopResult:
        calls.append(profile)
        return TrainingLoopResult(
            run_id=self.context.run_id,
            profile=profile,
            executor=executor,
            auto_import=auto_import,
            max_steps=max_steps,
            steps=[],
            queue_counts={"completed": 1},
            stopped_reason="next_round_blocked",
            completed=True,
        )

    monkeypatch.setattr(LoopOrchestrator, "run_training_loop", fake_training_loop)

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=True,
        auto_advance=False,
    )

    assert calls == ["debug"]
    assert result.profile == "debug"
    assert result.profile_history == ["debug"]
    assert "Auto-advance" in result.next_action


def test_optimize_full_profile_execute_requires_confirmation(tmp_path: Path) -> None:
    """Full COCO profiles should not execute unless the user confirms the budget."""
    data_yaml = _make_dataset(tmp_path / "dataset")

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="baseline_full",
        execute=True,
    )

    assert result.ok is False
    assert result.executed is False
    assert any(check.name == "confirm_full_run" and not check.ok for check in result.preflight)
    assert "--confirm-full-run" in result.next_action
    assert not result.experiment_plan_path.exists()


def test_optimize_full_profile_dry_run_does_not_require_confirmation(tmp_path: Path) -> None:
    """Dry-run planning for full profiles should remain available without confirmation."""
    data_yaml = _make_dataset(tmp_path / "dataset")

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="baseline_full",
        execute=False,
    )

    assert result.ok is True
    assert result.executed is False
    assert not any(check.name == "confirm_full_run" and not check.ok for check in result.preflight)
    assert result.experiment_plan_path.exists()


def test_optimize_advance_cli_runs_existing_run(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The optimize advance CLI should expose a short profile transition command."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    assert main(
        [
            "optimize",
            "coco",
            "--data",
            str(data_yaml),
            "--run-id",
            "cli-coco",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    ) == 0
    capsys.readouterr()

    assert main(
        [
            "optimize",
            "advance",
            "--run",
            str(tmp_path / "runs" / "cli-coco"),
            "--to-profile",
            "pilot",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "YOLO Agent Optimize" in output
    assert "Profile:  pilot" in output
    assert "Mode:     dry-run" in output
    assert "Queue:" in output
    assert f"Status:   yolo-agent loop status --run {tmp_path / 'runs' / 'cli-coco'}" in output
    queue = ExecutionQueue.from_yaml(tmp_path / "runs" / "cli-coco" / "execution_queue.yaml")
    assert queue.items[0].command.metadata["training_budget_profile"] == "pilot"


def test_optimize_cli_blocks_full_execute_without_confirmation(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The CLI should make full COCO execution an explicit opt-in."""
    data_yaml = _make_dataset(tmp_path / "dataset")

    assert main(
        [
            "optimize",
            "coco",
            "--data",
            str(data_yaml),
            "--run-id",
            "cli-coco",
            "--run-root",
            str(tmp_path / "runs"),
            "--profile",
            "baseline_full",
            "--execute",
        ]
    ) == 1

    output = capsys.readouterr().out
    assert "State:    preflight failed" in output
    assert "confirm_full_run: error" in output
    assert "--confirm-full-run" in output
    assert not (tmp_path / "runs" / "cli-coco" / "artifacts" / "experiment_plan.yaml").exists()


def test_optimize_advance_cli_blocks_full_execute_without_confirmation(
    tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    """The advance shortcut should not bypass full-run confirmation."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    assert main(
        [
            "optimize",
            "coco",
            "--data",
            str(data_yaml),
            "--run-id",
            "cli-coco",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    ) == 0
    capsys.readouterr()

    assert main(
        [
            "optimize",
            "advance",
            "--run",
            str(tmp_path / "runs" / "cli-coco"),
            "--to-profile",
            "candidate_full",
            "--execute",
        ]
    ) == 1

    output = capsys.readouterr().out
    assert "Profile:  candidate_full" in output
    assert "confirm_full_run: error" in output
    assert "--confirm-full-run" in output


def test_optimize_cli_runs_coco_dry_run(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The top-level optimize CLI should be available and user-facing."""
    data_yaml = _make_dataset(tmp_path / "dataset")

    assert "optimize" in COMMANDS
    assert main(
        [
            "optimize",
            "coco",
            "--data",
            str(data_yaml),
            "--run-id",
            "cli-coco",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Starting YOLO Agent optimize" in output
    assert "Run: cli-coco  Profile: debug  Mode: dry-run" in output
    assert "Preset:   coco_yolo26_auto" in output
    assert "Profile:  debug" in output
    assert "Mode:     dry-run" in output
    assert "Queue:" in output
    assert f"Status:   yolo-agent loop status --run {tmp_path / 'runs' / 'cli-coco'}" in output
    assert (tmp_path / "runs" / "cli-coco" / "task.yaml").exists()
    task = yaml.safe_load((tmp_path / "runs" / "cli-coco" / "task.yaml").read_text(encoding="utf-8-sig"))
    assert task["primary_metric"]["name"] == "map50_95"
    plan = yaml.safe_load(
        (tmp_path / "runs" / "cli-coco" / "artifacts" / "experiment_plan.yaml").read_text(encoding="utf-8-sig")
    )
    assert plan["metadata"]["preset"] == "coco_yolo26_auto"


def test_optimize_event_progress_renders_stage_events(capsys) -> None:  # type: ignore[no-untyped-def]
    """Event log lines should render immediately useful progress output."""
    _print_event_progress(
        '{"event_type":"stage_started","stage":"profile_data","status":"running",'
        '"message":"Running profile_data (attempt 1/1)."}'
    )

    output = capsys.readouterr().out
    assert "progress: stage_started stage=profile_data status=running" in output
    assert "Running profile_data" in output


def test_optimize_event_progress_renders_training_logs(capsys) -> None:  # type: ignore[no-untyped-def]
    """Executor log events should show live train/val progress instead of waiting text."""
    _print_event_progress(
        '{"event_type":"executor_log","message":"\\u001b[K                 Class     Images  Instances      '
        'Box(P          R      mAP50  mAP50-95): 68% ━━━━━━━━──── 60/87 2.5it/s 40.3s<10.8s",'
        '"details":{"node_id":"node_yolo26n_coco_debug"}}'
    )

    output = capsys.readouterr().out
    assert "training:" in output
    assert "Class Images" in output
    assert "68%" in output
    assert "60/87" in output
    assert "\x1b" not in output
    assert "━" not in output
