"""User-facing execution resource failure output tests."""

from pathlib import Path

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
)
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueItem
from yolo_agent.core.executor import ExecutionResult
from yolo_agent.core.experiment_graph import ExperimentNode


HOST_OOM_OUTPUT = """
SystemError: Caught SystemError in DataLoader worker process 0.
Original numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.17 MiB
SystemError: <built-in function warpAffine> returned a result with an exception set
"""


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
    assert "State:    BLOCKED - paired comparison incomplete" in output
    assert "Queue:    candidate=done; matched baseline=automatic retry required" in output
    assert "NO DECISION YET" in output
    assert "GPU memory was exhausted" in output
    assert "retry the original batch=48 after confirming the GPU is free" in output
    assert "Comparison: unavailable" in output
    assert "Auto budget:" not in output
    assert "Plan:" not in output
    assert "Reason:   complete" not in output
