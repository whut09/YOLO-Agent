"""User-facing execution resource failure output tests."""

from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.optimize_runner import OptimizeResult
from yolo_agent.cli import (
    _optimize_reason,
    _optimize_state,
    _optimize_training_state,
    _optimize_user_summary_lines,
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
