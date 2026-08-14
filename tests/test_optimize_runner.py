"""One-command optimize runner tests."""

from __future__ import annotations

import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml
import pytest

import yolo_agent.agents.optimize_runner as optimize_module
from yolo_agent.agents.auto_optimization_loop import (
    AutoOptimizationLoopDriver,
    AutoOptimizationResult,
    AutoRoundResult,
    CandidateExecutionAssessment,
)
from yolo_agent.agents.optimize_runner import OptimizeRunner
from yolo_agent.agents.orchestrator import LoopOrchestrator, TrainingLoopResult
from yolo_agent.cli import (
    COMMANDS,
    _auto_optimization_decision_lines,
    _exhausted_search_result_lines,
    _auto_round_outcome,
    _auto_round_comparison_lines,
    _auto_round_paper_lines,
    _auto_round_state_label,
    _gpu_certification_command,
    _optimize_user_summary_lines,
    _optimize_reason,
    _research_snapshot_missing_automatic_components,
    _research_snapshot_needs_recipe_refresh,
    _optimize_state,
    _optimize_training_state,
    _print_event_progress,
    _print_live_status_progress,
    _print_optimize_summary,
    _run_with_event_progress,
    main,
)
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueStore
from yolo_agent.core.execution_failure import ExecutionFailure
from yolo_agent.core.executor import ExecutionResult
from yolo_agent.core.gpu_runtime import GPURuntimeSnapshot
from yolo_agent.core.loop_status import LoopRunStatus
from yolo_agent.core.optimization_objective import OptimizationObjectiveStatus
from yolo_agent.core.process_probe import ProcessProbeResult, ProcessTerminateResult
from yolo_agent.core.resource_scheduler import ResourceDecision
from yolo_agent.core.run_allocation import allocate_base_run_id


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


def test_train_detects_legacy_paper_neck_fact_bindings(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    recipes_path = snapshot_dir / "recipes.yaml"
    recipes_path.write_text(
        yaml.safe_dump(
            {
                "recipes": [
                    {
                        "recipe_id": "paper.neck.multi_scale_fusion",
                        "target_error_facts": [
                            {"fact_type": "scale_variation"},
                            {"fact_type": "small_object_false_negative"},
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert _research_snapshot_needs_recipe_refresh(snapshot_dir) is True

    recipes_path.write_text(
        yaml.safe_dump(
            {
                "recipes": [
                    {
                        "recipe_id": "paper.neck.multi_scale_fusion",
                        "target_error_facts": [
                            {"fact_type": "scale_variation"},
                            {
                                "fact_type": "area_metric",
                                "area": "small",
                                "metric_name": "ap_small",
                            },
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert _research_snapshot_needs_recipe_refresh(snapshot_dir) is False


def test_train_detects_implemented_paper_adapters_missing_runtime_maturity(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "component_contracts.yaml").write_text(
        yaml.safe_dump(
            {
                "components": {
                    "sampling.small_object": {"maturity": "adapter_implemented"},
                    "head.p2_small_object": {"maturity": "smoke_passed"},
                    "loss.quality.correlation": {"maturity": "gpu_certified"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert _research_snapshot_missing_automatic_components(snapshot_dir) == [
        "sampling.small_object"
    ]


def test_train_detects_stale_frozen_adapter_identity(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "component_contracts.yaml").write_text(
        yaml.safe_dump(
            {
                "components": {
                    "sampling.small_object": {"maturity": "smoke_passed"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (snapshot_dir / "effective_component_maturity.yaml").write_text(
        yaml.safe_dump(
            {
                "entries": [
                    {
                        "component_id": "sampling.small_object",
                        "adapter_hash": "stale",
                        "ultralytics_version": "stale",
                        "protocol_hash": "stale",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert _research_snapshot_missing_automatic_components(snapshot_dir) == [
        "sampling.small_object"
    ]


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
    assert result.training_loop.stopped_reason == "complete"
    assert result.queue_counts["completed"] == 1
    assert "Rerun with --execute" in result.next_action
    plan = yaml.safe_load(result.experiment_plan_path.read_text(encoding="utf-8-sig"))
    assert plan["metadata"]["preset"] is None
    queue = ExecutionQueue.from_yaml(result.queue_path)
    assert queue.items[0].status == "completed"
    assert queue.items[0].command.command_type == "train"
    assert queue.items[0].command.metadata["training_budget_profile"] == "debug"
    assert queue.metadata["queue_source_plan_hash"] == plan["metadata"]["plan_hash"]
    context = LoopOrchestrator.from_run_dir(result.run_dir).context
    initialization_status = Path(context.metadata["run_initialization_status_path"])
    assert initialization_status.is_file()


def test_failed_orchestrator_initialization_archives_partial_run(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"

    def fail_initialization(**kwargs):  # type: ignore[no-untyped-def]
        run_dir = Path(kwargs["run_root"]) / kwargs["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_context.yaml").write_text("partial: true\n", encoding="utf-8")
        raise RuntimeError("initializer failed")

    monkeypatch.setattr(LoopOrchestrator, "initialize", fail_initialization)

    with pytest.raises(RuntimeError, match="initializer failed"):
        OptimizeRunner().run(
            kind="coco",
            model="yolo26n.pt",
            data_yaml=data_yaml,
            run_id="failed-init",
            run_root=run_root,
            profile="debug",
            execute=False,
        )

    assert not (run_root / "failed-init").exists()
    reports = list((run_root / "initialization_failures").glob("failed-init*.yaml"))
    assert len(reports) == 1
    report = yaml.safe_load(reports[0].read_text(encoding="utf-8-sig"))
    assert report["status"] == "failed"
    assert report["action"] == "archived_failed_initialization"


def test_optimize_persists_fresh_run_allocation_metadata(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    (run_root / "coco-yolo26n").mkdir(parents=True)
    allocation = allocate_base_run_id(run_root, "coco-yolo26n")

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id=allocation.allocated_run_id,
        run_root=run_root,
        profile="debug",
        execute=False,
        run_allocation=allocation,
    )

    context = LoopOrchestrator.from_run_dir(result.run_dir).context
    assert context.run_id == "coco-yolo26n-1"
    assert context.metadata["requested_run_id"] == "coco-yolo26n"
    assert context.metadata["allocated_run_id"] == "coco-yolo26n-1"
    assert context.metadata["run_sequence"] == 1
    assert context.metadata["fresh_run_reason"] == "existing_run_directory"


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
    assert "next: yolo-agent train" in output


def test_execute_reenters_external_gpu_wait_for_live_recovery(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    initialized = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="gpu-wait",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )
    queue = ExecutionQueue.from_yaml(initialized.queue_path)
    item = queue.items[0]
    item.status = "running"
    item.mark_result(
        ExecutionResult(
            run_id="gpu-wait",
            node_id=item.node_id,
            candidate_id=item.candidate_id,
            status="failed",
            command=item.command,
            failure=ExecutionFailure(
                kind="gpu_memory_exhausted",
                summary="GPU busy.",
                root_cause="An unrelated GPU process is active.",
                recoverable=True,
                failed_settings={"batch": -1},
                waiting_for_external_gpu=True,
                recovery_strategy="wait_for_external_gpu_then_retry_same_batch",
                gpu_snapshot=GPURuntimeSnapshot(
                    used_memory_mb=10153,
                    total_memory_mb=24564,
                ),
            ),
        )
    )
    queue.to_yaml(initialized.queue_path)

    execute_result = optimize_module._existing_running_queue_result(
        kind="coco",
        run_id="gpu-wait",
        run_dir=initialized.run_dir,
        requested_profile="debug",
        executor="ultralytics-train",
        preflight=[],
        task_path=initialized.task_path,
        plan_path=initialized.experiment_plan_path,
        queue_path=initialized.queue_path,
        execute=True,
    )
    dry_run_result = optimize_module._existing_running_queue_result(
        kind="coco",
        run_id="gpu-wait",
        run_dir=initialized.run_dir,
        requested_profile="debug",
        executor="dry-run",
        preflight=[],
        task_path=initialized.task_path,
        plan_path=initialized.experiment_plan_path,
        queue_path=initialized.queue_path,
        execute=False,
    )

    assert execute_result is None
    assert dry_run_result is not None
    assert dry_run_result.queue_counts["needs_resume"] == 1


def test_stop_marks_running_queue_and_prints_recovery(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    """stop should be a reliable fallback when Ctrl+C is not trusted."""
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

    code = main(["stop", "--run", str(result.run_dir)])

    updated = store.load()
    output = capsys.readouterr().out
    assert code == 0
    assert updated.items[0].status == "needs_resume"
    assert updated.items[0].resource_blockers == ["interrupted_by_user"]
    assert "stopped_processes=1" in output
    assert "marked_running_items=1" in output
    assert "next: yolo-agent train" in output

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
    assert "Rerun yolo-agent train for the same run" in result.next_action
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
    assert "pilot-only candidate proposals" in result.next_action
    plan = yaml.safe_load(result.experiment_plan_path.read_text(encoding="utf-8-sig"))
    assert plan["metadata"]["profile"] == "pilot"
    context = LoopOrchestrator.from_run_dir(result.run_dir).context
    study = optimize_module.ASHAStudy.from_yaml(result.run_dir / "artifacts" / "asha_state.yaml")
    assert study.run_protocol_hash == context.run_protocol_hash


def test_fast_baseline_block_is_explained_as_missing_debug_sanity(
    tmp_path: Path,
) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="pilot-gate-output",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )
    queue = ExecutionQueue.from_yaml(result.queue_path)
    queue.items[0].status = "skipped"
    queue.items[0].message = "Fast Baseline Gate blocked this run."
    ExecutionQueueStore(result.run_dir).save(queue)

    assert _optimize_state(result) == "BLOCKED before training: debug sanity is missing"
    assert _optimize_training_state(result) == (
        "no; pilot did not start and no candidate metrics were produced"
    )
    assert _optimize_reason(result) == "pilot requires a completed debug sanity run"
    assert _optimize_user_summary_lines(result, []) == [
        "BLOCKED - training did not start.",
        "Problem: this fresh run started at pilot, but the required debug sanity run is missing.",
        "Measured result: none; no baseline or candidate mAP was produced in this attempt.",
        "Action: rerun the same train command; YOLO Agent will recover with debug, then continue to pilot automatically.",
    ]


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
    assert "Status:   READY - dry run completed" in output
    assert "Training: not started; execution was not requested" in output
    assert "Tried:    0 candidates" in output
    queue = ExecutionQueue.from_yaml(tmp_path / "runs" / "cli-coco" / "execution_queue.yaml")
    assert queue.items[0].command.metadata["training_budget_profile"] == "pilot"


def test_optimize_auto_loop_cli_runs_existing_run_without_baseline_rerun(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    """The auto-loop shortcut should continue from an existing run directory."""
    from yolo_agent.agents.auto_optimization_loop import AutoOptimizationLoopDriver

    run_dir = tmp_path / "runs" / "coco-yolo26n"
    run_dir.mkdir(parents=True)
    calls: list[tuple[Path, int, bool]] = []

    def fake_run(
        self: AutoOptimizationLoopDriver,
        base_run_dir: Path,
        auto_rounds: int,
        *,
        execute: bool,
        executor: str,
        max_steps: int = 8,
        auto_import: bool = True,
        profile: str = "pilot",
    ) -> AutoOptimizationResult:
        calls.append((Path(base_run_dir), auto_rounds, execute))
        return AutoOptimizationResult(
            base_run_id="coco-yolo26n",
            base_run_dir=run_dir,
            requested_rounds=auto_rounds,
            executed=execute,
            rounds=[],
            stopped_reason="requested_rounds_completed",
            summary_path=run_dir / "artifacts" / "auto_optimization_summary.md",
            full_candidate_recommendations_path=run_dir / "artifacts" / "full_candidate_recommendations.yaml",
        )

    monkeypatch.setattr(AutoOptimizationLoopDriver, "run", fake_run)

    assert main(
        [
            "optimize",
            "auto-loop",
            "--run",
            str(run_dir),
            "--auto-rounds",
            "2",
            "--execute",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert calls == [(run_dir, 2, True)]
    assert "YOLO Agent Auto Loop" in output
    assert "Rounds:   0/2" in output


def test_one_command_train_enables_automatic_gpu_certification(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """The beginner execute path should internalize readiness certification."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        optimize_module,
        "optimize_preflight",
        lambda kind, data_yaml, execute=False: [
            optimize_module.PreflightCheck(
                name="test_preflight",
                ok=True,
                level="info",
                message="ok",
            )
        ],
    )

    def fake_training_loop(
        self: LoopOrchestrator,
        profile: str,
        executor: str,
        max_steps: int = 8,
        auto_import: bool = True,
    ) -> TrainingLoopResult:
        return TrainingLoopResult(
            run_id=self.context.run_id,
            profile=profile,
            executor=executor,
            auto_import=auto_import,
            max_steps=max_steps,
            queue_counts={"completed": 1},
            stopped_reason="complete",
            completed=True,
        )

    def fake_auto_run(
        self: AutoOptimizationLoopDriver,
        base_run_dir: Path,
        auto_rounds: int,
        **kwargs: object,
    ) -> AutoOptimizationResult:
        observed["auto_certify_gpu"] = self.auto_certify_gpu
        observed["certification_model"] = self.certification_model
        observed["execute"] = kwargs["execute"]
        run_dir = Path(base_run_dir)
        return AutoOptimizationResult(
            base_run_id=run_dir.name,
            base_run_dir=run_dir,
            requested_rounds=auto_rounds,
            executed=True,
            stopped_reason="requested_rounds_completed",
            summary_path=run_dir / "artifacts" / "summary.md",
            full_candidate_recommendations_path=run_dir / "artifacts" / "recommendations.yaml",
        )

    monkeypatch.setattr(LoopOrchestrator, "run_training_loop", fake_training_loop)
    monkeypatch.setattr(AutoOptimizationLoopDriver, "run", fake_auto_run)

    OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="one-command-auto-cert",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=True,
        auto_rounds=1,
    )

    assert observed == {
        "auto_certify_gpu": True,
        "certification_model": "yolo26n.pt",
        "execute": True,
    }


def test_auto_optimization_decision_rejects_not_promoted_full_candidate(tmp_path: Path) -> None:
    """The final optimize panel should say when a pilot candidate is not ready for full COCO."""
    run_dir = tmp_path / "runs" / "coco-yolo26n"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    recommendations_path = artifacts / "full_candidate_recommendations.yaml"
    recommendations_path.write_text(
        yaml.safe_dump(
            {
                "recommendations": [
                    {
                        "candidate_id": "next_augmentation_reduce_mosaic_strength",
                        "promotion_status": "pilot_only_evidence_required",
                    },
                    {
                        "candidate_id": "next_augmentation_reduce_mosaic_strength",
                        "promotion_status": "pilot_only_evidence_required",
                    },
                ],
                "recommendation_only": [
                    {"action_id": "benchmark_latency"},
                    {"action_id": "benchmark_latency"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    auto = AutoOptimizationResult(
        base_run_id="coco-yolo26n",
        base_run_dir=run_dir,
        requested_rounds=30,
        executed=True,
        stopped_reason="requested_rounds_completed",
        summary_path=artifacts / "auto_optimization_summary.md",
        full_candidate_recommendations_path=recommendations_path,
    )

    lines = _auto_optimization_decision_lines(auto)

    assert "full_candidate=not_ready; do not start full COCO for current candidates" in lines
    assert "blocked_candidates=next_augmentation_reduce_mosaic_strength" in lines
    assert "evidence_first=benchmark_latency" in lines
    assert all(", next_augmentation_reduce_mosaic_strength" not in line for line in lines)


def test_auto_summary_distinguishes_empty_planning_from_model_regression(tmp_path: Path) -> None:
    """A zero-candidate round should never be presented as a failed model experiment."""
    run_dir = tmp_path / "runs" / "coco-yolo26n"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    round_result = AutoRoundResult(
        round_index=1,
        run_id="coco-yolo26n-r1",
        run_dir=run_dir / "coco-yolo26n-r1",
        parent_run_id="coco-yolo26n",
        status="blocked",
        stop_reason="no_guarded_candidates",
        auto_round_summary_path=artifacts / "auto_round_summary.yaml",
    )
    auto = AutoOptimizationResult(
        base_run_id="coco-yolo26n",
        base_run_dir=run_dir,
        requested_rounds=30,
        executed=True,
        rounds=[round_result],
        stopped_reason="no_guarded_candidates",
        summary_path=artifacts / "auto_optimization_summary.md",
        full_candidate_recommendations_path=artifacts / "full_candidate_recommendations.yaml",
    )

    assert _auto_round_state_label(round_result) == "auto round 1 blocked during candidate planning"
    assert "candidate_training=not_started" in _auto_round_outcome(round_result)
    assert _auto_optimization_decision_lines(auto) == [
        "candidate_training=not_started",
        "why=baseline pilot completed, but candidate planning selected zero proposals; this is not a negative optimization result",
        "next=repair or rerun candidate planning before spending additional GPU budget",
    ]


def test_auto_summary_explains_method_exhaustion_without_scalar_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "method-exhausted"
    asha_path = run_dir / "asha_state.yaml"
    asha_path.parent.mkdir(parents=True)
    asha_path.write_text(
        yaml.safe_dump(
            {
                "trials": [
                    {
                        "candidate_id": "next_augmentation_scale_aug_0_7",
                        "observations": [
                            {
                                "stage_id": "pilot_3",
                                "paired_delta": 0.00111588,
                                "paired_result_verified": True,
                            },
                            {
                                "stage_id": "pilot_10",
                                "paired_delta": -0.00471270,
                                "paired_result_verified": True,
                            },
                        ],
                    },
                    {
                        "candidate_id": "next_augmentation_copy_paste_0_1",
                        "observations": [
                            {
                                "stage_id": "pilot_3",
                                "paired_delta": 0.00020836,
                                "paired_result_verified": True,
                            },
                            {
                                "stage_id": "pilot_10",
                                "paired_delta": 0.0,
                                "paired_result_verified": True,
                            },
                        ],
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    round_result = AutoRoundResult(
        round_index=4,
        run_id="method-exhausted-r4",
        run_dir=run_dir / "method-exhausted-r4",
        parent_run_id="method-exhausted",
        status="blocked",
        stop_reason="method_candidates_exhausted",
        auto_round_summary_path=run_dir / "method-exhausted-r4" / "summary.yaml",
    )
    auto = AutoOptimizationResult(
        base_run_id="method-exhausted",
        base_run_dir=run_dir,
        requested_rounds=12,
        executed=True,
        rounds=[round_result],
        stopped_reason="method_candidates_exhausted",
        summary_path=run_dir / "summary.md",
        full_candidate_recommendations_path=run_dir / "recommendations.yaml",
        asha_state_path=asha_path,
        objective_status=OptimizationObjectiveStatus(
            objective_hash="objective",
            primary_metric="map50_95",
            baseline_value=0.3932752179,
            best_value=0.3933794001,
            observed_delta=0.0001041822,
            required_delta=0.02,
        ),
    )

    assert _auto_round_state_label(round_result) == (
        "auto round 4 stopped after method candidates were exhausted"
    )
    assert "scalar HPO is disabled" in _auto_round_outcome(round_result)
    assert any(
        "optimizer/lr/weight-decay fallback is disabled" in line
        for line in _auto_optimization_decision_lines(auto)
    )
    assert _exhausted_search_result_lines(auto) == [
        "baseline_mAP50-95=0.393275",
        "required_delta=+0.020000",
        "candidates_tested=2; training_observations=4",
        "best_screening=scale_aug_0_7 pilot_3_delta=+0.001116",
        "promotion_check=scale_aug_0_7 pilot_10_delta=-0.004713 (rejected)",
        "confirmed_objective_delta=+0.000104 (best result that passed promotion)",
    ]
    result = optimize_module.OptimizeResult(
        kind="coco",
        run_id="method-exhausted",
        run_dir=run_dir,
        model="yolo26n.pt",
        data_yaml=tmp_path / "coco.yaml",
        profile="pilot",
        executor="ultralytics-train",
        executed=True,
        task_path=run_dir / "task.yaml",
        experiment_plan_path=run_dir / "plan.yaml",
        queue_path=run_dir / "queue.yaml",
        training_loop=TrainingLoopResult(
            run_id="method-exhausted",
            profile="pilot",
            executor="ultralytics-train",
            max_steps=8,
            stopped_reason="complete",
        ),
        auto_optimization=auto,
    )
    assert _optimize_reason(result) == (
        "search finished without a candidate that reached the requested improvement"
    )
    assert _optimize_user_summary_lines(result, [])[0] == (
        "SEARCH FINISHED - no candidate passed guarded promotion."
    )
    assert "routine scalar HPO remains disabled" in optimize_module._auto_optimization_next_action(
        auto,
        "fallback",
    )


def test_auto_summary_explains_candidates_planned_but_not_registered(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "runs" / "improve-map"
    asha_path = run_dir / "artifacts" / "asha_state.yaml"
    asha_path.parent.mkdir(parents=True)
    asha_path.write_text(
        yaml.safe_dump(
            {
                "trials": [
                    {"candidate_id": f"tested-{index}", "observations": [{"stage_id": "pilot_3"}]}
                    for index in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )
    round_result = AutoRoundResult(
        round_index=1,
        run_id="improve-map-r1",
        run_dir=run_dir / "improve-map-r1",
        parent_run_id="improve-map",
        status="blocked",
        stop_reason="no_new_asha_trials",
        auto_round_summary_path=run_dir / "improve-map-r1" / "summary.yaml",
        candidate_assessments=[
            CandidateExecutionAssessment(
                policy_id=f"method-{index}",
                candidate_id=f"method-{index}",
                execution_class="executable",
            )
            for index in range(3)
        ],
    )
    auto = AutoOptimizationResult(
        base_run_id="improve-map",
        base_run_dir=run_dir,
        requested_rounds=12,
        executed=True,
        rounds=[round_result],
        stopped_reason="no_new_asha_trials",
        summary_path=run_dir / "summary.md",
        full_candidate_recommendations_path=run_dir / "recommendations.yaml",
        asha_state_path=asha_path,
    )

    assert _auto_round_state_label(round_result) == (
        "auto round 1 blocked before ASHA registration"
    )
    assert _auto_round_outcome(round_result) == (
        "candidate_training=not_started; candidates_planned=3; ASHA_trials_registered=0"
    )
    assert _auto_optimization_decision_lines(auto)[1] == (
        "candidates_planned=3; candidates_trained=0"
    )
    assert _optimize_user_summary_lines(
        optimize_module.OptimizeResult(
            kind="coco",
            run_id="improve-map",
            run_dir=run_dir,
            model="yolo26n.pt",
            data_yaml=tmp_path / "coco.yaml",
            profile="pilot",
            executor="ultralytics-train",
            executed=True,
            task_path=run_dir / "task.yaml",
            experiment_plan_path=run_dir / "plan.yaml",
            queue_path=run_dir / "queue.yaml",
            auto_optimization=auto,
        ),
        ["metrics mAP50-95=0.39272"],
    )[:4] == [
        "BLOCKED - candidates were planned, but candidate training did not start.",
        "Candidates planned: 3.",
        "Candidates trained: 0.",
        "mAP improvement: not measured; there is no candidate result to compare.",
    ]

    _print_optimize_summary(
        optimize_module.OptimizeResult(
            kind="coco",
            run_id="improve-map",
            run_dir=run_dir,
            model="yolo26n.pt",
            data_yaml=tmp_path / "coco.yaml",
            profile="pilot",
            executor="ultralytics-train",
            executed=True,
            task_path=run_dir / "task.yaml",
            experiment_plan_path=run_dir / "plan.yaml",
            queue_path=run_dir / "queue.yaml",
            auto_optimization=auto,
        ),
        "coco_yolo26_auto",
    )
    output = capsys.readouterr().out
    assert "Status:   BLOCKED - candidate optimization did not start" in output
    assert "Training: baseline pilot completed; candidate training did not start" in output
    assert "Tried:    4 candidates tested in 0 paired runs; this round trained 0 (3 candidate planned)" in output
    assert "Result:   mAP improvement not measured" in output


def test_old_no_trial_summary_reports_exhausted_when_only_small_object_method_remains(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "runs" / "improve-map"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "optimization_objective.yaml").write_text(
        yaml.safe_dump(
            {
                "goal_expression": "+2map",
                "goal_description": "Improve overall mAP",
                "primary_metric": "map50_95",
            }
        ),
        encoding="utf-8",
    )
    asha_path = artifacts / "asha_state.yaml"
    asha_path.write_text(
        yaml.safe_dump(
            {
                "trials": [
                    {
                        "candidate_id": "paper.neck.gold",
                        "observations": [
                            {
                                "stage_id": "pilot_3",
                                "paired_result_verified": True,
                                "paired_delta": 0.000019,
                            },
                            {
                                "stage_id": "pilot_10",
                                "paired_result_verified": True,
                                "paired_delta": -0.001266,
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    round_result = AutoRoundResult(
        round_index=11,
        run_id="improve-map-r11",
        run_dir=run_dir.parent / "improve-map-r11",
        parent_run_id="improve-map-r10",
        status="blocked",
        stop_reason="no_new_asha_trials",
        auto_round_summary_path=artifacts / "round.yaml",
        candidate_assessments=[
            CandidateExecutionAssessment(
                policy_id="small-p2",
                candidate_id="paper_recipe_yolo26_small_object_p2_v1_0_0",
                action_id="yolo26_small_object_p2",
                adapter_ids=["head.p2_small_object"],
                execution_class="executable",
            )
        ],
    )
    auto = AutoOptimizationResult(
        base_run_id="improve-map",
        base_run_dir=run_dir,
        requested_rounds=12,
        executed=True,
        rounds=[round_result],
        stopped_reason="no_new_asha_trials",
        summary_path=artifacts / "summary.md",
        full_candidate_recommendations_path=artifacts / "recommendations.yaml",
        asha_state_path=asha_path,
    )
    result = optimize_module.OptimizeResult(
        kind="coco",
        run_id="improve-map",
        run_dir=run_dir,
        model="yolo26n.pt",
        data_yaml=tmp_path / "coco.yaml",
        profile="pilot",
        executor="ultralytics-train",
        executed=True,
        task_path=run_dir / "task.yaml",
        experiment_plan_path=run_dir / "plan.yaml",
        queue_path=run_dir / "queue.yaml",
        auto_optimization=auto,
    )

    _print_optimize_summary(result, "coco_yolo26_auto")
    output = capsys.readouterr().out

    assert "Status:   COMPLETED - search finished" in output
    assert "Tried:    1 candidate in 2 verified paired runs" in output
    assert "pilot_3 +0.000019, pilot_10 -0.001266 (rejected)" in output
    assert "remaining planned method targets small objects, not overall mAP" in output


def test_compact_summary_counts_coupled_ablation_as_one_method_cohort(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "runs" / "coupled-search"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    asha_path = artifacts / "asha_state.yaml"
    trials = []
    for combination in ("A", "B", "A+B"):
        trials.append(
            {
                "candidate_id": f"hard-negative-{combination}",
                "source_node": {
                    "candidate_config": {"action_id": "yolo26_hard_negative_pair"},
                    "command_spec": {
                        "metadata": {
                            "guarded_coupled_ablation_member": True,
                            "ablation_combination_id": combination,
                        }
                    },
                },
                "observations": [
                    {
                        "stage_id": "pilot_3",
                        "paired_result_verified": True,
                        "paired_delta": -0.001,
                    }
                ],
            }
        )
    trials.append(
        {
            "candidate_id": "task-aligned",
            "observations": [
                {
                    "stage_id": "pilot_3",
                    "paired_result_verified": True,
                    "paired_delta": -0.002,
                }
            ],
        }
    )
    asha_path.write_text(yaml.safe_dump({"trials": trials}), encoding="utf-8")
    round_result = AutoRoundResult(
        round_index=4,
        run_id="coupled-search-r4",
        run_dir=run_dir.parent / "coupled-search-r4",
        parent_run_id="coupled-search-r3",
        status="completed",
        stop_reason="asha_assignment_completed",
        auto_round_summary_path=artifacts / "round.yaml",
    )
    auto = AutoOptimizationResult(
        base_run_id="coupled-search",
        base_run_dir=run_dir,
        requested_rounds=12,
        executed=True,
        rounds=[round_result],
        stopped_reason="no_improvement_patience_reached",
        summary_path=artifacts / "summary.md",
        full_candidate_recommendations_path=artifacts / "recommendations.yaml",
        asha_state_path=asha_path,
    )
    result = optimize_module.OptimizeResult(
        kind="coco",
        run_id="coupled-search",
        run_dir=run_dir,
        model="yolo26n.pt",
        data_yaml=tmp_path / "coco.yaml",
        profile="pilot",
        executor="ultralytics-train",
        executed=True,
        task_path=run_dir / "task.yaml",
        experiment_plan_path=run_dir / "plan.yaml",
        queue_path=run_dir / "queue.yaml",
        auto_optimization=auto,
    )

    _print_optimize_summary(result, "coco_yolo26_auto")
    output = capsys.readouterr().out

    assert "Status:   COMPLETED - search finished" in output
    assert (
        "Tried:    1 independent candidate + 1 coupled method cohort "
        "(3 ablation arms) in 4 verified paired runs"
    ) in output
    assert "Auto budget:" not in output
    assert "Paper components:" not in output


def test_paper_summary_prioritizes_selected_and_eligible_components(tmp_path: Path) -> None:
    plan_path = tmp_path / "paper_recipe_plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "executable_pilot_policies": [],
                "paper_component_decisions": [
                    {
                        "paper_ids": ["paper-stale"],
                        "component_id": "assigner.stale",
                        "eligible": False,
                        "rejection_reasons": ["frozen_adapter_hash_mismatch:old:new"],
                    },
                    {
                        "paper_ids": ["paper-neck"],
                        "component_id": "neck.multi_scale_fusion",
                        "eligible": True,
                        "rejection_reasons": [],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    round_result = AutoRoundResult(
        round_index=1,
        run_id="run-r1",
        run_dir=tmp_path / "run-r1",
        parent_run_id="run",
        paper_recipe_plan_path=plan_path,
        auto_round_summary_path=tmp_path / "summary.yaml",
    )

    assert _auto_round_paper_lines(round_result) == [
        "paper recipes planned=0; current cohort uses local evidence-bound method recipes",
        "certified paper components available=1: neck.multi_scale_fusion",
        "paper component summary: eligible=1 rejected=1 stale=1",
    ]


def test_paper_summary_shows_executable_search_funnel(tmp_path: Path) -> None:
    plan_path = tmp_path / "paper_recipe_plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "executable_portfolio": {
                    "catalog_papers": 728,
                    "method_profiles": 237,
                    "recipe_definitions": 260,
                    "runtime_eligible_recipes": 41,
                    "diagnosis_matched_recipes": 18,
                    "planner_selected_recipes": 6,
                    "critic_accepted_recipes": 5,
                    "executable_policies": 5,
                },
                "executable_pilot_policies": [],
                "paper_component_decisions": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    round_result = AutoRoundResult(
        round_index=1,
        run_id="run-r1",
        run_dir=tmp_path / "run-r1",
        parent_run_id="run",
        paper_recipe_plan_path=plan_path,
        auto_round_summary_path=tmp_path / "summary.yaml",
    )

    assert _auto_round_paper_lines(round_result)[:2] == [
        "search funnel: catalog=728 papers, profiles=237, recipes=260, "
        "runtime-ready=41 recipes",
        "this diagnosis: matched=18, selected=6, critic-passed=5, materialized=5",
    ]


def test_paper_summary_explains_why_certified_component_did_not_train(tmp_path: Path) -> None:
    plan_path = tmp_path / "paper_recipe_plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "executable_pilot_policies": [],
                "paper_component_decisions": [
                    {
                        "paper_ids": ["paper-neck"],
                        "component_id": "neck.multi_scale_fusion",
                        "eligible": True,
                        "rejection_reasons": [],
                    }
                ],
                "recipe_critic_reports": [
                    {
                        "recipe_id": "paper.neck.multi_scale_fusion",
                        "findings": [{"code": "missing_bound_error_facts"}],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    round_result = AutoRoundResult(
        round_index=11,
        run_id="run-r11",
        run_dir=tmp_path / "run-r11",
        parent_run_id="run",
        status="blocked",
        stop_reason="method_candidates_exhausted",
        paper_recipe_plan_path=plan_path,
        auto_round_summary_path=tmp_path / "summary.yaml",
    )

    lines = _auto_round_paper_lines(round_result)

    assert lines[-1] == (
        "paper blocker=certified components did not enter training because "
        "paper.neck.multi_scale_fusion lacked a matching local error-fact binding"
    )


def test_paper_summary_distinguishes_local_and_paper_candidates(tmp_path: Path) -> None:
    plan_path = tmp_path / "paper_recipe_plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "executable_pilot_policies": [
                    {
                        "action_id": "paper.neck.multi_scale_fusion",
                        "components": ["neck.multi_scale_fusion"],
                        "expected_improvement": {"paper_ids": ["paper-neck"]},
                    },
                    {
                        "action_id": "yolo26_small_object_sampling",
                        "components": ["sampling.small_object"],
                        "expected_improvement": {"component_ids": ["sampling.small_object"]},
                    },
                ],
                "paper_component_decisions": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    round_result = AutoRoundResult(
        round_index=1,
        run_id="run-r1",
        run_dir=tmp_path / "run-r1",
        parent_run_id="run",
        paper_recipe_plan_path=plan_path,
        auto_round_summary_path=tmp_path / "summary.yaml",
        candidate_assessments=[
            CandidateExecutionAssessment(
                policy_id="paper-neck",
                candidate_id="paper-neck",
                execution_class="executable",
                action_id="paper.neck.multi_scale_fusion",
            ),
            CandidateExecutionAssessment(
                policy_id="local-sampling",
                candidate_id="local-sampling",
                execution_class="executable",
                action_id="yolo26_small_object_sampling",
            ),
        ],
    )

    lines = _auto_round_paper_lines(round_result)

    assert lines[:4] == [
        "candidate recipes planned=2 (paper=1, local=1)",
        "entered executable queue=2",
        "recipe=paper.neck.multi_scale_fusion source=paper:paper-neck component=neck.multi_scale_fusion",
        "recipe=yolo26_small_object_sampling source=local evidence component=sampling.small_object",
    ]


def test_paper_summary_reports_policy_rejection_before_asha(tmp_path: Path) -> None:
    plan_path = tmp_path / "paper_recipe_plan.yaml"
    policy_path = tmp_path / "policy_evaluation.yaml"
    policy_id = "paper_recipe_paper.neck.multi_scale_fusion_0_1_0"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "executable_pilot_policies": [
                    {
                        "policy_id": policy_id,
                        "action_id": "paper.neck.multi_scale_fusion",
                        "components": ["neck.multi_scale_fusion"],
                    }
                ],
                "paper_component_decisions": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    policy_path.write_text(
        yaml.safe_dump(
            {
                "evaluations": [
                    {
                        "policy_id": policy_id,
                        "decision": "rejected",
                        "errors": ["multi_variable_candidate_marked_single_variable"],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    round_result = AutoRoundResult(
        round_index=11,
        run_id="run-r11",
        run_dir=tmp_path,
        parent_run_id="run",
        paper_recipe_plan_path=plan_path,
        policy_evaluation_path=policy_path,
        auto_round_summary_path=tmp_path / "summary.yaml",
    )

    lines = _auto_round_paper_lines(round_result)

    assert "entered executable queue=0" in lines
    assert (
        f"not queued={policy_id} "
        "reason=multi_variable_candidate_marked_single_variable"
    ) in lines


def test_auto_summary_explains_readiness_block_before_candidate_training(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "runs" / "readiness-blocked"
    readiness = {
        "ready": False,
        "mode": "blocked",
        "blockers": [
            "gpu_certification_report_invalid:1 validation error; report hash does not match its payload"
        ],
    }
    auto = AutoOptimizationResult(
        base_run_id="readiness-blocked",
        base_run_dir=run_dir,
        requested_rounds=60,
        executed=True,
        stopped_reason="optimization_readiness_blocked",
        summary_path=run_dir / "artifacts" / "summary.md",
        full_candidate_recommendations_path=run_dir / "artifacts" / "recommendations.yaml",
        readiness=readiness,  # type: ignore[arg-type]
    )
    result = optimize_module.OptimizeResult(
        kind="coco",
        run_id="readiness-blocked",
        run_dir=run_dir,
        model="yolo26n.pt",
        data_yaml=tmp_path / "coco.yaml",
        profile="pilot",
        executor="ultralytics-train",
        executed=True,
        task_path=run_dir / "task.yaml",
        experiment_plan_path=run_dir / "plan.yaml",
        queue_path=run_dir / "queue.yaml",
        auto_optimization=auto,
    )

    lines = _optimize_user_summary_lines(
        result,
        ["metrics mAP50-95=0.39272 mAP50=0.54953"],
    )

    assert lines[0] == "BLOCKED - baseline pilot completed successfully; automatic optimization did not start."
    assert any("Baseline pilot: mAP50-95=0.39272" in line for line in lines)
    assert any("Optimization candidates: 0 trained" in line for line in lines)
    assert any("mAP improvement: not measured" in line for line in lines)
    assert any(
        "GPU certification report is invalid or stale (report hash mismatch)" in line
        for line in lines
    )
    assert _auto_optimization_decision_lines(auto) == [
        "candidate_training=not_started",
        "measured_improvement=none; no candidate was trained or compared",
        "blocked_by=GPU certification report is invalid or stale (report hash mismatch)",
        "next=rerun the same train command; certification is handled automatically",
    ]
    command = _gpu_certification_command(result)
    assert "yolo-agent advanced certify-gpu" in command
    assert "--model yolo26n.pt" in command
    assert "--execute-real-gpu" in command

    _print_optimize_summary(result, "coco_yolo26_auto")
    output = capsys.readouterr().out
    assert "Status:   BLOCKED - optimization safety checks did not pass" in output
    assert "Training: baseline pilot completed; candidate training did not start" in output
    assert "Tried:    0 candidates trained" in output
    assert "Result:   mAP improvement not measured" in output


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
    assert "Status:   FAILED - preflight did not pass" in output
    assert "Result:   no mAP result" in output
    assert "--confirm-full-run" in output
    assert not (tmp_path / "runs" / "cli-coco" / "artifacts" / "experiment_plan.yaml").exists()


def test_optimize_cli_missing_data_does_not_report_dry_run_or_repeat_bad_command(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    missing_data = tmp_path / "missing" / "coco.yaml"
    run_dir = tmp_path / "runs" / "missing-data"

    assert main(
        [
            "optimize",
            "coco",
            "--data",
            str(missing_data),
            "--run-id",
            "missing-data",
            "--run-root",
            str(tmp_path / "runs"),
            "--execute",
        ]
    ) == 1

    output = capsys.readouterr().out
    assert "Status:   FAILED - preflight did not pass" in output
    assert "Training: training did not start" in output
    assert "Next:     Fix --data: point it to an existing dataset YAML" in output
    assert not run_dir.exists()


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
    assert "Status:   FAILED - preflight did not pass" in output
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
    assert "Status:   READY - dry run completed" in output
    assert "Training: not started; execution was not requested" in output
    assert "Result:   training plan created" in output
    assert (tmp_path / "runs" / "cli-coco" / "task.yaml").exists()
    task = yaml.safe_load((tmp_path / "runs" / "cli-coco" / "task.yaml").read_text(encoding="utf-8-sig"))
    assert task["primary_metric"]["name"] == "map50_95"
    plan = yaml.safe_load(
        (tmp_path / "runs" / "cli-coco" / "artifacts" / "experiment_plan.yaml").read_text(encoding="utf-8-sig")
    )
    assert plan["metadata"]["preset"] == "coco_yolo26_auto"


def test_optimize_summary_prints_completed_pilot_metrics(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """A completed pilot should print metrics, batch choice, and trust boundary."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="cli-coco",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )
    store = ExecutionQueueStore(result.run_dir)
    queue = store.load()
    item = queue.items[0]
    results_dir = tmp_path / "ultralytics" / "cli-coco_node_yolo26n_coco_pilot"
    weights_dir = results_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"model")
    (results_dir / "results.csv").write_text(
        "\n".join(
            [
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)",
                "1,0.5,0.4,0.45,0.30",
                "2,0.6,0.5,0.55,0.38",
                "",
            ]
        ),
        encoding="utf-8",
    )
    item.command.expected_artifacts["results_csv"] = results_dir / "results.csv"
    item.command.expected_artifacts["best_pt"] = weights_dir / "best.pt"
    store.save(queue)
    EvidenceStore(result.run_dir.parent).log_candidate_metrics(
        run_id=result.run_id,
        candidate_id=item.candidate_id,
        node_id=item.node_id,
        dataset_version=item.experiment_node.data_version,
        split="runtime",
        source="test",
        metrics={
            "fast_baseline_pilot_passed": True,
            "runtime_avg_it_per_sec": 12.3,
            "execution_duration_seconds": 45.0,
        },
    )
    batch_path = result.run_dir / "artifacts" / f"{item.node_id}_batch_tuning_result.json"
    batch_path.write_text(
        '{"selected_batch": 32, "reason": "Selected batch 32 by highest avg_it_per_sec."}',
        encoding="utf-8",
    )

    _print_optimize_summary(result, preset_name="coco_yolo26_auto")

    output = capsys.readouterr().out
    assert "Result:" in output
    assert "mAP50-95=0.38" in output
    assert "batch=32" in output
    assert "Status:   READY - dry run completed" in output


def test_auto_round_summary_prints_matched_baseline_and_paired_deltas(tmp_path: Path) -> None:
    """Users must see accuracy and resource tradeoffs before promotion decisions."""
    run_dir = tmp_path / "run-r2"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "node_candidate_paired_experiment_result.json").write_text(
        json.dumps(
            {
                "candidate_id": "adamw",
                "baseline_candidate_id": "matched_baseline_control",
                "protocol_match_status": "matched",
                "verified": False,
                "metric_deltas": {
                    "map50_95": {"baseline_value": 0.3938, "candidate_value": 0.1236, "paired_delta": -0.2702},
                    "latency_ms": {"baseline_value": 15.18, "candidate_value": 14.07, "paired_delta": -1.11},
                    "model_size_mb": {"baseline_value": 5.241, "candidate_value": 5.2409, "paired_delta": -0.0001},
                },
                "blockers": ["missing_target_error_fact_pair:person"],
            }
        ),
        encoding="utf-8",
    )
    result = AutoRoundResult(
        round_index=2,
        run_id="run-r2",
        run_dir=run_dir,
        parent_run_id="run",
        stop_reason="asha_evidence_incomplete",
        auto_round_summary_path=artifacts / "auto_round_summary.yaml",
    )

    lines = _auto_round_comparison_lines(result)

    assert "candidate=adamw baseline=matched_baseline_control protocol=matched" in lines[0]
    assert "mAP50-95 candidate=0.123600 baseline=0.393800 paired_delta=-0.270200 (regressed)" in lines
    assert "latency_ms candidate=14.070000 baseline=15.180000 paired_delta=-1.110000 (improved)" in lines
    assert "conclusion=accuracy regressed or did not improve" in lines[-1]


def test_auto_round_summary_uses_verified_asha_history_when_latest_round_is_empty(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-r11"
    run_dir.mkdir()
    asha_path = tmp_path / "asha_state.yaml"
    asha_path.write_text(
        yaml.safe_dump(
            {
                "trials": [
                    {
                        "candidate_id": "next_augmentation_scale_aug_0_7",
                        "observations": [
                            {
                                "stage_id": "pilot_3",
                                "paired_delta": 0.00109199,
                                "paired_result_verified": True,
                            },
                            {
                                "stage_id": "pilot_10",
                                "paired_delta": -0.00395084,
                                "paired_result_verified": True,
                            },
                        ],
                    },
                    {
                        "candidate_id": "next_sampling_small_object",
                        "observations": [
                            {
                                "stage_id": "pilot_3",
                                "paired_delta": -0.002,
                                "paired_result_verified": True,
                            }
                        ],
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    round_result = SimpleNamespace(run_dir=run_dir, stop_reason="method_candidates_exhausted")

    lines = _auto_round_comparison_lines(
        round_result,
        asha_state_path=asha_path,
    )

    assert lines == [
        "history=3 verified paired comparisons across 2 candidates",
        "best_screening=scale_aug_0_7 pilot_3_delta=+0.001092",
        "promotion_check=scale_aug_0_7 pilot_10_delta=-0.003951 (rejected)",
        "status=latest round did not train; comparison history loaded from ASHA",
    ]


def test_auto_round_summary_explains_missing_coco_pair_with_provisional_metrics(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "run-r5" / "artifacts"
    results = artifacts / "execution_results"
    results.mkdir(parents=True)
    (results / "candidate.json").write_text(
        json.dumps({"candidate_id": "paper_recipe", "metrics": {"map50_95": 0.38828}}),
        encoding="utf-8",
    )
    (results / "baseline.json").write_text(
        json.dumps({"candidate_id": "matched_baseline_control", "metrics": {"map50_95": 0.393834}}),
        encoding="utf-8",
    )
    paired = artifacts / "node_candidate_paired_experiment_result.json"
    paired.write_text(
        json.dumps(
            {
                "candidate_id": "paper_recipe",
                "baseline_candidate_id": "unknown",
                "protocol_match_status": "mismatch",
                "verified": False,
                "metric_deltas": {
                    "map50_95": {"baseline_value": None, "candidate_value": None, "paired_delta": None},
                    "latency_ms": {"baseline_value": 17.62, "candidate_value": 21.33, "paired_delta": 3.71},
                },
                "blockers": ["split_mismatch"],
            }
        ),
        encoding="utf-8",
    )
    round_result = SimpleNamespace(run_dir=tmp_path / "run-r5", stop_reason="asha_evidence_incomplete")

    lines = _auto_round_comparison_lines(round_result)

    assert "candidate_training_mAP50-95=0.388280 (provisional)" in lines
    assert "baseline_training_mAP50-95=0.393834 (provisional)" in lines
    assert "candidate/control fixed COCO val2017 metrics are not both available" in lines[-1]


def test_optimize_event_progress_renders_stage_events(capsys) -> None:  # type: ignore[no-untyped-def]
    """Event log lines should render immediately useful progress output."""
    _print_event_progress(
        '{"event_type":"stage_started","stage":"profile_data","status":"running",'
        '"message":"Running profile_data (attempt 1/1)."}'
    )

    output = capsys.readouterr().out
    assert "progress: stage_started stage=profile_data status=running" in output
    assert "Running profile_data" in output


def test_optimize_event_progress_renders_automatic_gpu_certification(capsys) -> None:  # type: ignore[no-untyped-def]
    _print_event_progress(
        '{"event_type":"gpu_certification_started","status":"running",'
        '"message":"Running automatic mini GPU certification before candidate optimization."}'
    )

    output = capsys.readouterr().out
    assert "progress: gpu_certification_started" in output
    assert "automatic mini GPU certification" in output


def test_live_progress_treats_missing_context_as_initialization(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _print_live_status_progress(tmp_path / "new-run")

    output = capsys.readouterr().out
    assert "progress: initializing run context" in output
    assert "status unavailable" not in output


def test_live_progress_reports_non_training_stage(monkeypatch, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_context.yaml").write_text("run_id: run\n", encoding="utf-8")
    status = LoopRunStatus(
        run_id="run",
        run_dir=run_dir,
        current_stage="profile_data",
        current_stage_status="running",
    )
    monkeypatch.setattr("yolo_agent.cli.load_loop_status", lambda _: status)

    _print_live_status_progress(run_dir)

    output = capsys.readouterr().out
    assert "progress: stage profile_data is still running" in output
    assert "training heartbeat" not in output


def test_live_progress_renders_dataset_profile_counts(monkeypatch, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (run_dir / "run_context.yaml").write_text("run_id: run\n", encoding="utf-8")
    status = LoopRunStatus(
        run_id="run",
        run_dir=run_dir,
        current_stage="profile_data",
        current_stage_status="running",
    )
    monkeypatch.setattr("yolo_agent.cli.load_loop_status", lambda _: status)
    (artifacts_dir / "dataset_profile_progress.json").write_text(
        json.dumps(
            {
                "phase": "reading_labels",
                "status": "running",
                "current": 42000,
                "total": 123287,
                "percent": 34.1,
                "pid": 123,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    _print_live_status_progress(run_dir)

    assert "progress: profile_data reading_labels: 42000/123287 (34.1%)" in capsys.readouterr().out


def test_live_progress_reports_stale_dataset_profiler(monkeypatch, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (run_dir / "run_context.yaml").write_text("run_id: run\n", encoding="utf-8")
    status = LoopRunStatus(
        run_id="run",
        run_dir=run_dir,
        current_stage="profile_data",
        current_stage_status="running",
    )
    monkeypatch.setattr("yolo_agent.cli.load_loop_status", lambda _: status)
    monkeypatch.setattr("yolo_agent.cli._profile_process_is_alive", lambda _: False)
    (artifacts_dir / "dataset_profile_progress.json").write_text(
        json.dumps(
            {
                "phase": "reading_labels",
                "status": "running",
                "current": 1000,
                "total": 123287,
                "percent": 0.8,
                "pid": 999999,
                "updated_at": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    _print_live_status_progress(run_dir)

    output = capsys.readouterr().out
    assert "profile_data stale" in output
    assert "is not running" in output


def test_optimize_event_progress_renders_auto_round_strategy(capsys) -> None:  # type: ignore[no-untyped-def]
    """Auto-loop events should show round number and current strategy."""
    _print_event_progress(
        '{"event_type":"auto_round_decision","status":"completed",'
        '"message":"Round 3/30 strategy=hard_negative_mining class=executable.",'
        '"details":{"round_index":3,"total_rounds":30,"strategy":"hard_negative_mining",'
        '"execution_class":"executable","reasons":["train command uses only supported options"]}}'
    )

    output = capsys.readouterr().out
    assert "progress: auto[3/30] strategy=hard_negative_mining class=executable" in output
    assert "reason=train command uses only supported options" in output


def test_optimize_event_progress_hides_ultralytics_noise(capsys) -> None:  # type: ignore[no-untyped-def]
    """Executor log events should hide verbose Ultralytics boilerplate."""
    _print_event_progress(
        '{"event_type":"executor_log","message":"\\u001b[K                 Class     Images  Instances      '
        'Box(P          R      mAP50  mAP50-95): 68% 鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹€鈹€鈹€鈹€ 60/87 2.5it/s 40.3s<10.8s",'
        '"details":{"node_id":"node_yolo26n_coco_debug"}}'
    )
    _print_event_progress(
        '{"event_type":"executor_log","message":"engine\\\\trainer: agnostic_nms=False, amp=True, batch=48, cache=disk",'
        '"details":{"node_id":"node_yolo26n_coco_debug"}}'
    )
    _print_event_progress(
        '{"event_type":"executor_log","message":"0 -1 1 464 ultralytics.nn.modules.conv.Conv [3, 16, 3, 2]",'
        '"details":{"node_id":"node_yolo26n_coco_debug"}}'
    )

    output = capsys.readouterr().out
    assert output == ""


def test_optimize_event_progress_renders_training_progress(capsys) -> None:  # type: ignore[no-untyped-def]
    """Executor log events should still show concise epoch progress."""
    _print_event_progress(
        '{"event_type":"executor_log","message":"5/10 15.1G 0.72 0.51 0.89 640: 42%|####------| 37/87 [00:18<00:24, 2.04it/s]",'
        '"details":{"node_id":"node_yolo26n_coco_debug"}}'
    )

    output = capsys.readouterr().out
    assert "training:" in output
    assert "5/10" in output
    assert "42%" in output
    assert "37/87" in output
    assert "\x1b" not in output
    assert "\ufffd" not in output


def test_optimize_progress_watcher_does_not_replay_existing_child_logs(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    """Rerunning auto-loop should not replay old child training logs from the beginning."""
    base = tmp_path / "runs" / "coco-yolo26n"
    old_child = tmp_path / "runs" / "coco-yolo26n-r1"
    base.mkdir(parents=True)
    old_child.mkdir()
    (base / "events.jsonl").write_text("", encoding="utf-8")
    (old_child / "events.jsonl").write_text(
        '{"event_type":"executor_log","message":"1/10 9.0G old 640: 10% 1/10 3.0it/s",'
        '"details":{"node_id":"old_node"}}\n',
        encoding="utf-8",
    )

    def action() -> str:
        new_child = tmp_path / "runs" / "coco-yolo26n-r2"
        new_child.mkdir()
        (new_child / "events.jsonl").write_text(
            '{"event_type":"executor_log","message":"1/10 9.0G new 640: 10% 1/10 3.0it/s",'
            '"details":{"node_id":"new_node"}}\n',
            encoding="utf-8",
        )
        time.sleep(1.2)
        return "done"

    assert _run_with_event_progress(base, action, enabled=True, include_child_runs=True) == "done"

    output = capsys.readouterr().out
    assert "new 640" in output
    assert "old 640" not in output
