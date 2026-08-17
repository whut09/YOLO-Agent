"""User-facing execution resource failure output tests."""

from pathlib import Path
from types import SimpleNamespace

import yaml

import yolo_agent.cli as cli

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.auto_optimization_loop import AutoOptimizationResult, AutoRoundResult
from yolo_agent.agents.optimize_runner import OptimizeResult
from yolo_agent.agents.orchestrator import TrainingLoopResult
from yolo_agent.cli import (
    _auto_round_comparison_lines,
    _optimize_reason,
    _optimize_state,
    _optimize_training_state,
    _optimize_user_summary_lines,
    _print_optimize_summary,
    _verified_search_summary_lines,
)
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueItem
from yolo_agent.core.execution_failure import ExecutionFailure
from yolo_agent.core.executor import ExecutionResult
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.gpu_runtime import GPURuntimeSnapshot, GPUProcessInfo
from yolo_agent.core.optimization_objective import OptimizationObjectiveStatus


HOST_OOM_OUTPUT = """
SystemError: Caught SystemError in DataLoader worker process 0.
Original numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.17 MiB
SystemError: <built-in function warpAffine> returned a result with an exception set
"""

ADAPTER_DTYPE_OUTPUT = """
Traceback (most recent call last):
RuntimeError: expected scalar type Float but found Half
yolo_agent.adapters.ultralytics.plugin_bridge.PluginExecutionError: plugin hook failed:
yolo_agent.components.adapters.assigners.yolo26_assignment:YOLO26AssignmentRuntimePlugin:compute_loss:
expected scalar type Float but found Half
"""


def test_verified_search_summary_uses_observed_patience_field() -> None:
    auto = AutoOptimizationResult(
        base_run_id="run",
        base_run_dir=Path("runs/run"),
        requested_rounds=12,
        executed=True,
        stopped_reason="no_improvement_patience_reached",
        summary_path=Path("runs/run/summary.md"),
        full_candidate_recommendations_path=Path("runs/run/recommendations.yaml"),
        objective_status=OptimizationObjectiveStatus(
            objective_hash="objective",
            primary_metric="map50_95",
            baseline_value=0.394,
            best_candidate_id="candidate",
            best_value=0.394,
            observed_delta=0.0,
            required_delta=0.02,
            no_improvement_rounds=4,
        ),
    )
    lines = _verified_search_summary_lines(
        auto,
        {
            "candidate_id": "candidate",
            "metric_deltas": {
                "map50_95": {
                    "baseline_value": 0.394,
                    "candidate_value": 0.394,
                    "paired_delta": 0.0,
                }
            },
        },
    )

    assert "Stop: 4 consecutive candidates failed to improve." in lines


def test_cli_explains_legacy_host_memory_failure_to_user(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "resource-failure"
    run_dir.mkdir(parents=True)
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name="pilot",
        batch=48,
        workers=8,
        overrides={"cache": "disk"},
    )
    node = ExperimentNode(
        node_id="node_pilot",
        candidate_config=CandidateConfig(
            candidate_id="pilot",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        command=command.display(),
        command_spec=command,
    )
    item = ExecutionQueueItem.from_node("resource-failure", node)
    item.mark_running()
    item.mark_result(
        ExecutionResult(
            run_id="resource-failure",
            node_id="node_pilot",
            candidate_id="pilot",
            status="failed",
            command=command,
            stdout=HOST_OOM_OUTPUT,
            message="Ultralytics training failed.",
        )
    )
    queue_path = run_dir / "execution_queue.yaml"
    ExecutionQueue(run_id="resource-failure", items=[item]).to_yaml(queue_path)
    result = OptimizeResult(
        kind="coco",
        run_id="resource-failure",
        run_dir=run_dir,
        model="yolo26n.pt",
        data_yaml=Path("coco.yaml"),
        profile="pilot",
        executor="ultralytics-train",
        executed=True,
        task_path=run_dir / "task.yaml",
        experiment_plan_path=run_dir / "plan.yaml",
        queue_path=queue_path,
        queue_counts={"failed": 1},
    )

    lines = _optimize_user_summary_lines(result, [])

    assert _optimize_state(result) == "RECOVERABLE RESOURCE FAILURE"
    assert _optimize_training_state(result) == "stopped; pilot did not complete"
    assert _optimize_reason(result) == "System RAM was exhausted in an Ultralytics DataLoader worker."
    assert lines[0] == "RESOURCE FAILURE - this was not a model-quality result."
    assert any("batch=48 workers=8 cache=disk" in line for line in lines)
    assert any("next attempt will use workers=2" in line for line in lines)
    assert any("same train command" in line for line in lines)


def test_cli_explains_requeued_external_gpu_wait_without_checkpoint_language(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "gpu-wait"
    run_dir.mkdir(parents=True)
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name="debug",
        batch=32,
    )
    item = ExecutionQueueItem.from_node(
        "gpu-wait",
        ExperimentNode(
            node_id="node_debug",
            candidate_config=CandidateConfig(
                candidate_id="debug",
                base_model="yolo26n.pt",
                scale="n",
                framework="ultralytics",
            ),
            data_version="coco2017",
            command=command.display(),
            command_spec=command,
        ),
    )
    item.mark_running()
    item.mark_result(
        ExecutionResult(
            run_id="gpu-wait",
            node_id="node_debug",
            candidate_id="debug",
            status="failed",
            command=command,
            failure=ExecutionFailure(
                kind="gpu_memory_exhausted",
                summary="Training is waiting because another process is using GPU memory.",
                root_cause="An unrelated GPU process is active.",
                recoverable=True,
                waiting_for_external_gpu=True,
                recovery_strategy="wait_for_external_gpu_then_retry_same_batch",
                gpu_snapshot=GPURuntimeSnapshot(
                    used_memory_mb=9954,
                    total_memory_mb=24564,
                ),
            ),
        )
    )
    assert item.status == "needs_resume"
    queue_path = run_dir / "execution_queue.yaml"
    ExecutionQueue(run_id="gpu-wait", items=[item]).to_yaml(queue_path)
    result = OptimizeResult(
        kind="coco",
        run_id="gpu-wait",
        run_dir=run_dir,
        model="yolo26n.pt",
        data_yaml=Path("coco.yaml"),
        profile="debug",
        executor="ultralytics-train",
        executed=True,
        task_path=run_dir / "task.yaml",
        experiment_plan_path=run_dir / "plan.yaml",
        queue_path=queue_path,
        queue_counts={"needs_resume": 1},
    )

    assert _optimize_state(result) == "BLOCKED - GPU is busy"
    assert _optimize_training_state(result) == "paused; waiting for the external GPU workload to finish"
    assert _optimize_reason(result) == "Training is waiting because another process is using GPU memory."
    assert _optimize_user_summary_lines(result, [])[0] == "GPU BUSY - training is paused; this is not a model failure."


def test_optimize_summary_prints_all_paper_dispositions_without_complete_reason(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "runs" / "coverage-summary"
    coverage_path = run_dir / "artifacts" / "paper_candidate_coverage.yaml"
    coverage_path.parent.mkdir(parents=True)
    coverage_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "coverage-summary",
                "records": [
                    {"disposition": name}
                    for name in (
                        "queued",
                        "already_tested",
                        "evidence_recovery",
                        "implementation_request",
                        "incompatible",
                        "blocked_runtime",
                        "deferred_budget",
                    )
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = OptimizeResult(
        kind="coco",
        run_id="coverage-summary",
        run_dir=run_dir,
        model="yolo26n.pt",
        data_yaml=Path("coco.yaml"),
        profile="pilot",
        executor="ultralytics-train",
        executed=True,
        task_path=run_dir / "task.yaml",
        experiment_plan_path=run_dir / "plan.yaml",
        queue_path=run_dir / "execution_queue.yaml",
        queue_counts={"completed": 1},
    )

    _print_optimize_summary(result, "coco_yolo26_auto")
    output = capsys.readouterr().out

    assert "Coverage: queued=1 already_tested=1 evidence_recovery=1" in output
    assert "implementation_request=1 incompatible=1 blocked_runtime=1 deferred_budget=1" in output
    assert "Problem:  none" in output
    assert "Reason:   complete" not in output


def test_cli_catches_handler_exception_and_writes_traceback_artifact(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    def exploding_handler(_args: object) -> int:
        raise RuntimeError("synthetic ASHA registration failure")

    monkeypatch.setattr(
        "yolo_agent.cli.build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda _argv: SimpleNamespace(
                handler=exploding_handler,
                run_root=tmp_path / "runs",
                run_id="exception-summary",
            )
        ),
    )

    result = cli.main(["train"])
    output = capsys.readouterr().out
    log_path = tmp_path / "runs" / "exception-summary" / "artifacts" / "logs" / "cli_exception.log"

    assert result == 1
    assert "Traceback" not in output
    assert "Status:   FAILED - CLI execution error" in output
    assert str(log_path) in output
    assert "Traceback" in log_path.read_text(encoding="utf-8")
    assert "synthetic ASHA registration failure" in log_path.read_text(encoding="utf-8")


def test_live_executor_progress_does_not_echo_traceback_lines() -> None:
    import yolo_agent.cli as cli_module

    assert cli_module._executor_log_prefix("Traceback (most recent call last):") == ""


def test_cli_prints_concise_paired_run_gpu_failure(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "runs" / "improve-map"
    child_dir = tmp_path / "runs" / "improve-map-r10"
    child_dir.mkdir(parents=True)
    candidate_command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name="candidate",
        batch=48,
        workers=8,
    )
    baseline_command = candidate_command.model_copy(
        update={"metadata": {**candidate_command.metadata, "matched_baseline_control": True}}
    )
    candidate = ExecutionQueueItem.from_node(
        "improve-map-r10",
        ExperimentNode(
            node_id="node_candidate",
            candidate_config=CandidateConfig(
                candidate_id="scale_aug_0_7",
                base_model="yolo26n.pt",
                scale="n",
                framework="ultralytics",
            ),
            data_version="coco2017",
            command=candidate_command.display(),
            command_spec=candidate_command,
        ),
    )
    candidate.mark_running()
    candidate.mark_result(
        ExecutionResult(
            run_id="improve-map-r10",
            node_id="node_candidate",
            candidate_id="scale_aug_0_7",
            status="completed",
            command=candidate_command,
            metrics={"map50_95": 0.388003801},
        )
    )
    baseline = ExecutionQueueItem.from_node(
        "improve-map-r10",
        ExperimentNode(
            node_id="node_baseline",
            candidate_config=CandidateConfig(
                candidate_id="matched_baseline_control",
                base_model="yolo26n.pt",
                scale="n",
                framework="ultralytics",
            ),
            data_version="coco2017",
            command=baseline_command.display(),
            command_spec=baseline_command,
        ),
    )
    baseline.mark_running()
    baseline.mark_result(
        ExecutionResult(
            run_id="improve-map-r10",
            node_id="node_baseline",
            candidate_id="matched_baseline_control",
            status="failed",
            command=baseline_command,
            stdout=(
                "Caught SystemError in DataLoader worker process 0. Unable to allocate buffer.\n"
                "torch.AcceleratorError: CUDA error: out of memory"
            ),
        )
    )
    baseline.status = "needs_resume"
    ExecutionQueue(run_id="improve-map-r10", items=[candidate, baseline]).to_yaml(
        child_dir / "execution_queue.yaml"
    )
    training_loop = TrainingLoopResult(
        run_id="improve-map-r10",
        profile="pilot",
        executor="ultralytics-train",
        max_steps=8,
        queue_counts={"completed": 1, "needs_resume": 1},
        stopped_reason="queue_blocked_blocked",
    )
    round_result = AutoRoundResult(
        round_index=10,
        run_id="improve-map-r10",
        run_dir=child_dir,
        parent_run_id="improve-map-r9",
        status="blocked",
        stop_reason="queue_blocked",
        training_loop=training_loop,
        auto_round_summary_path=child_dir / "artifacts" / "auto_round_summary.yaml",
    )
    auto = AutoOptimizationResult(
        base_run_id="improve-map",
        base_run_dir=run_dir,
        requested_rounds=12,
        executed=True,
        rounds=[round_result],
        stopped_reason="queue_blocked",
        summary_path=run_dir / "artifacts" / "summary.md",
        full_candidate_recommendations_path=run_dir / "artifacts" / "recommendations.yaml",
    )
    result = OptimizeResult(
        kind="coco",
        run_id="improve-map",
        run_dir=run_dir,
        model="yolo26n.pt",
        data_yaml=Path("coco.yaml"),
        profile="pilot",
        executor="ultralytics-train",
        executed=True,
        task_path=run_dir / "task.yaml",
        experiment_plan_path=run_dir / "plan.yaml",
        queue_path=run_dir / "execution_queue.yaml",
        auto_optimization=auto,
    )

    comparison = _auto_round_comparison_lines(round_result)
    _print_optimize_summary(result, "coco_yolo26_auto")
    output = capsys.readouterr().out

    assert comparison[0] == "completed=scale_aug_0_7 mAP50-95=0.388004"
    assert "Status:   BLOCKED - candidate/control comparison incomplete" in output
    assert "Training: candidate completed; matched baseline did not complete" in output
    assert "Result:   mAP improvement not measured" in output
    assert "saved work will be recovered automatically" in output
    assert "Auto budget:" not in output
    assert "Plan:" not in output
    assert "Reason:   complete" not in output


def test_cli_explains_paired_candidate_adapter_failure(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "runs" / "improve-map"
    child_dir = tmp_path / "runs" / "improve-map-r6"
    child_dir.mkdir(parents=True)
    objective_dir = run_dir / "artifacts"
    objective_dir.mkdir(parents=True)
    (objective_dir / "optimization_objective.yaml").write_text(
        "goal_expression: +2map\n",
        encoding="utf-8",
    )
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name="candidate",
        batch=48,
        workers=8,
    )
    baseline_command = command.model_copy(
        update={"metadata": {**command.metadata, "matched_baseline_control": True}}
    )
    candidate = ExecutionQueueItem.from_node(
        "improve-map-r6",
        ExperimentNode(
            node_id="node_candidate",
            candidate_config=CandidateConfig(
                candidate_id="paper.assigner.dynamic_smooth_label",
                base_model="yolo26n.pt",
                scale="n",
                framework="ultralytics",
            ),
            data_version="coco2017",
            command=command.display(),
            command_spec=command,
        ),
    )
    candidate.mark_running()
    candidate.mark_result(
        ExecutionResult(
            run_id="improve-map-r6",
            node_id="node_candidate",
            candidate_id="paper.assigner.dynamic_smooth_label",
            status="failed",
            command=command,
            stdout=ADAPTER_DTYPE_OUTPUT,
        )
    )
    baseline = ExecutionQueueItem.from_node(
        "improve-map-r6",
        ExperimentNode(
            node_id="node_baseline",
            candidate_config=CandidateConfig(
                candidate_id="matched_baseline_control",
                base_model="yolo26n.pt",
                scale="n",
                framework="ultralytics",
            ),
            data_version="coco2017",
            command=baseline_command.display(),
            command_spec=baseline_command,
        ),
    )
    baseline.mark_running()
    baseline.mark_result(
        ExecutionResult(
            run_id="improve-map-r6",
            node_id="node_baseline",
            candidate_id="matched_baseline_control",
            status="completed",
            command=baseline_command,
            metrics={"map50_95": 0.394474},
        )
    )
    ExecutionQueue(run_id="improve-map-r6", items=[candidate, baseline]).to_yaml(
        child_dir / "execution_queue.yaml"
    )
    round_result = AutoRoundResult(
        round_index=6,
        run_id="improve-map-r6",
        run_dir=child_dir,
        parent_run_id="improve-map-r5",
        status="completed",
        stop_reason="asha_candidate_failed_isolated",
        training_loop=TrainingLoopResult(
            run_id="improve-map-r6",
            profile="pilot",
            executor="ultralytics-train",
            max_steps=8,
            queue_counts={"completed": 1, "failed": 1},
            stopped_reason="queue_failed",
        ),
        auto_round_summary_path=child_dir / "artifacts" / "auto_round_summary.yaml",
    )
    result = OptimizeResult(
        kind="coco",
        run_id="improve-map",
        run_dir=run_dir,
        model="yolo26n.pt",
        data_yaml=Path("coco.yaml"),
        profile="pilot",
        executor="ultralytics-train",
        executed=True,
        task_path=run_dir / "task.yaml",
        experiment_plan_path=run_dir / "plan.yaml",
        queue_path=run_dir / "execution_queue.yaml",
        auto_optimization=AutoOptimizationResult(
            base_run_id="improve-map",
            base_run_dir=run_dir,
            requested_rounds=12,
            executed=True,
            rounds=[round_result],
            stopped_reason="requested_rounds_completed",
            summary_path=run_dir / "artifacts" / "summary.md",
            full_candidate_recommendations_path=run_dir / "artifacts" / "recommendations.yaml",
        ),
    )

    _print_optimize_summary(result, "coco_yolo26_auto")
    output = capsys.readouterr().out

    assert "Status:   FAILED - paper adapter crashed during candidate training" in output
    assert "Training: matched baseline completed; candidate failed" in output
    assert "Result:   mAP improvement not measured" in output
    assert "fix the adapter, then start a new run-id" in output
    assert "candidate_training=completed" not in output


def test_cli_explains_when_both_paired_jobs_waited_for_gpu(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "runs" / "improve-map"
    child_dir = tmp_path / "runs" / "improve-map-r2"
    child_dir.mkdir(parents=True)
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name="candidate",
        batch=32,
        workers=8,
    )
    snapshot = GPURuntimeSnapshot(
        used_memory_mb=9954,
        total_memory_mb=24564,
        processes=[
            GPUProcessInfo(
                pid=34020,
                process_name="python.exe",
                command_line="python indextts-worker.py",
            )
        ],
    )
    failure = ExecutionFailure(
        kind="gpu_memory_exhausted",
        summary="Training is waiting because another process is using GPU memory.",
        root_cause="An unrelated GPU process is active.",
        recoverable=True,
        waiting_for_external_gpu=True,
        recovery_strategy="wait_for_external_gpu_then_retry_same_batch",
        gpu_snapshot=snapshot,
    )
    items: list[ExecutionQueueItem] = []
    for candidate_id, baseline in (
        ("scale_aug_0_3", False),
        ("matched_baseline_control", True),
    ):
        item_command = command.model_copy(
            update={
                "metadata": {
                    **command.metadata,
                    "matched_baseline_control": baseline,
                }
            }
        )
        item = ExecutionQueueItem.from_node(
            "improve-map-r2",
            ExperimentNode(
                node_id=f"node_{candidate_id}",
                candidate_config=CandidateConfig(
                    candidate_id=candidate_id,
                    base_model="yolo26n.pt",
                    scale="n",
                    framework="ultralytics",
                ),
                data_version="coco2017",
                command=item_command.display(),
                command_spec=item_command,
            ),
        )
        item.mark_running()
        item.mark_result(
            ExecutionResult(
                run_id="improve-map-r2",
                node_id=item.node_id,
                candidate_id=candidate_id,
                status="failed",
                command=item_command,
                duration_seconds=0.0,
                failure=failure,
            )
        )
        items.append(item)
    ExecutionQueue(run_id="improve-map-r2", items=items).to_yaml(
        child_dir / "execution_queue.yaml"
    )
    round_result = AutoRoundResult(
        round_index=2,
        run_id="improve-map-r2",
        run_dir=child_dir,
        parent_run_id="improve-map-r1",
        status="blocked",
        stop_reason="resource_recovery_pending",
        training_loop=TrainingLoopResult(
            run_id="improve-map-r2",
            profile="pilot",
            executor="ultralytics-train",
            max_steps=8,
            queue_counts={"needs_resume": 2},
            stopped_reason="queue_blocked",
        ),
        auto_round_summary_path=child_dir / "artifacts" / "auto_round_summary.yaml",
    )
    result = OptimizeResult(
        kind="coco",
        run_id="improve-map",
        run_dir=run_dir,
        model="yolo26n.pt",
        data_yaml=Path("coco.yaml"),
        profile="pilot",
        executor="ultralytics-train",
        executed=True,
        task_path=run_dir / "task.yaml",
        experiment_plan_path=run_dir / "plan.yaml",
        queue_path=run_dir / "execution_queue.yaml",
        auto_optimization=AutoOptimizationResult(
            base_run_id="improve-map",
            base_run_dir=run_dir,
            requested_rounds=12,
            executed=True,
            rounds=[round_result],
            stopped_reason="resource_recovery_pending",
            summary_path=run_dir / "artifacts" / "summary.md",
            full_candidate_recommendations_path=run_dir / "artifacts" / "recommendations.yaml",
        ),
    )

    _print_optimize_summary(result, "coco_yolo26_auto")
    output = capsys.readouterr().out

    assert "Status:   BLOCKED - GPU busy; this is not a model failure" in output
    assert "Training: candidate and matched baseline did not start" in output
    assert "Tried:    0 candidates; no GPU training budget was consumed" in output
    assert "Result:   mAP improvement not measured" in output
    assert "free the external GPU workload, then rerun the same command" in output
    assert "candidate_training=completed" not in output
    assert "Reason:   complete" not in output
    assert "Auto budget:" not in output
