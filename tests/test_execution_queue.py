"""Execution queue tests."""

from __future__ import annotations

import pytest
from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueStore
from yolo_agent.core.executor import DryRunExecutor
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan


def _node(candidate_id: str = "baseline") -> ExperimentNode:
    return ExperimentNode(
        node_id=f"node_{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo11n",
            scale="n",
            framework="ultralytics",
        ),
        data_version="dataset-v1",
        command=f"yolo-agent smoke --candidate {candidate_id}",
    )


def test_execution_queue_materializes_experiment_plan(tmp_path: Path) -> None:
    """ExperimentPlan nodes should become queued execution items."""
    plan = ExperimentPlan(plan_id="plan-1", nodes=[_node("baseline"), _node("nwd")])

    queue = ExecutionQueue.from_experiment_plan("run-1", plan)

    assert len(queue.items) == 2
    assert queue.counts()["queued"] == 2
    assert queue.items[0].node_id == "node_baseline"
    assert queue.items[0].candidate_id == "baseline"
    assert queue.items[0].command.metadata["node_id"] == "node_baseline"
    assert queue.next_runnable() == queue.items[0]


def test_execution_queue_respects_max_nodes_under_limit(tmp_path: Path) -> None:
    """Queue creation should succeed when node count is within max_nodes."""
    plan = ExperimentPlan(plan_id="plan-1", nodes=[_node("baseline"), _node("nwd")])

    queue = ExecutionQueue.from_experiment_plan("run-1", plan, max_nodes=5)

    assert len(queue.items) == 2


def test_execution_queue_rejects_exceeding_max_nodes() -> None:
    """Queue creation should raise when node count exceeds max_nodes."""
    plan = ExperimentPlan(plan_id="plan-1", nodes=[_node("baseline"), _node("nwd")])

    with pytest.raises(ValueError) as exc_info:
        ExecutionQueue.from_experiment_plan("run-1", plan, max_nodes=1)

    assert "exceeded max_nodes limit" in str(exc_info.value)
    assert "2 nodes > 1" in str(exc_info.value)


def test_execution_queue_store_enforce_max_nodes(tmp_path: Path) -> None:
    """ExecutionQueueStore should pass max_nodes through to from_experiment_plan."""
    plan = ExperimentPlan(plan_id="plan-1", nodes=[_node("baseline"), _node("nwd")])
    store = ExecutionQueueStore(tmp_path / "runs" / "run-1")

    with pytest.raises(ValueError):
        store.enqueue_from_plan("run-1", plan, max_nodes=1)


def test_execution_queue_store_round_trips_yaml(tmp_path: Path) -> None:
    """Queue store should persist and reload queue state."""
    plan = ExperimentPlan(plan_id="plan-1", nodes=[_node()])
    store = ExecutionQueueStore(tmp_path / "runs" / "run-1")

    queue = store.enqueue_from_plan("run-1", plan)
    loaded = store.load()

    assert store.path == tmp_path / "runs" / "run-1" / "execution_queue.yaml"
    assert loaded.run_id == queue.run_id
    assert loaded.items[0].status == "queued"


def test_execution_queue_item_records_dry_run_result(tmp_path: Path) -> None:
    """Dry-run execution should mark queue items completed without running training."""
    plan = ExperimentPlan(plan_id="plan-1", nodes=[_node()])
    store = ExecutionQueueStore(tmp_path / "run-1")
    queue = store.enqueue_from_plan("run-1", plan)
    item = queue.next_runnable()
    assert item is not None

    item.mark_running()
    store.update_item(item)
    result = DryRunExecutor().execute(item.experiment_node, "run-1", item.command)
    result_path = tmp_path / "run-1" / "artifacts" / "execution_results" / "node_baseline.json"
    item.mark_result(result, result_path)
    updated = store.update_item(item)

    assert updated.counts()["completed"] == 1
    assert updated.items[0].last_result is not None
    assert updated.items[0].last_result.status == "dry_run"
