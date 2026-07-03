"""Persistent execution queue for materialized experiment nodes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from yolo_agent.core.executor import CommandSpec, ExecutionResult
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.core.yaml_io import YAMLModelMixin


QueueStatus = Literal["queued", "running", "completed", "failed", "skipped", "needs_evidence"]


class ExecutionQueueItem(BaseModel):
    """One executable item materialized from an ExperimentNode."""

    queue_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    node_id: str
    candidate_id: str
    command: CommandSpec
    experiment_node: ExperimentNode
    status: QueueStatus = "queued"
    attempts: int = 0
    requires_evidence: list[str] = Field(default_factory=list)
    last_result: ExecutionResult | None = None
    result_artifact: Path | None = None
    message: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_node(
        cls,
        run_id: str,
        node: ExperimentNode,
        requires_evidence: list[str] | None = None,
    ) -> "ExecutionQueueItem":
        """Create a queue item from one experiment node."""
        missing = list(requires_evidence or [])
        return cls(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            command=CommandSpec.from_experiment_node(node),
            experiment_node=node,
            status="needs_evidence" if missing else "queued",
            requires_evidence=missing,
            message="Waiting for required evidence." if missing else "",
        )

    def mark_running(self) -> None:
        """Mark the item as running and increment attempts."""
        self.status = "running"
        self.attempts += 1
        self.updated_at = datetime.now(timezone.utc)

    def mark_result(self, result: ExecutionResult, result_artifact: Path | None = None) -> None:
        """Attach an execution result and map it to queue status."""
        self.last_result = result
        self.result_artifact = result_artifact
        self.status = queue_status_from_execution_status(result.status)
        self.message = result.message
        self.updated_at = datetime.now(timezone.utc)


class ExecutionQueue(BaseModel, YAMLModelMixin):
    """A persisted queue of experiment nodes for one run."""

    run_id: str
    items: list[ExecutionQueueItem] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_experiment_plan(
        cls,
        run_id: str,
        plan: ExperimentPlan,
        requires_evidence_by_node: dict[str, list[str]] | None = None,
    ) -> "ExecutionQueue":
        """Materialize a queue from an experiment plan."""
        requirements = requires_evidence_by_node or {}
        return cls(
            run_id=run_id,
            items=[
                ExecutionQueueItem.from_node(
                    run_id=run_id,
                    node=node,
                    requires_evidence=requirements.get(node.node_id, []),
                )
                for node in plan.nodes
            ],
            metadata={
                "source_plan_id": plan.plan_id,
                "source_node_count": len(plan.nodes),
            },
        )

    def counts(self) -> dict[QueueStatus, int]:
        """Return item counts by queue status."""
        counts: dict[QueueStatus, int] = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "needs_evidence": 0,
        }
        for item in self.items:
            counts[item.status] += 1
        return counts

    def next_runnable(self) -> ExecutionQueueItem | None:
        """Return the next queued item."""
        for item in self.items:
            if item.status == "queued":
                return item
        return None

    def refresh_updated_at(self) -> None:
        """Update queue timestamp."""
        self.updated_at = datetime.now(timezone.utc)


class ExecutionQueueStore:
    """Filesystem store for runs/{run_id}/execution_queue.yaml."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "execution_queue.yaml"

    def save(self, queue: ExecutionQueue) -> Path:
        """Persist the queue."""
        queue.refresh_updated_at()
        return queue.to_yaml(self.path)

    def load(self) -> ExecutionQueue:
        """Load the persisted queue."""
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing execution queue: {self.path}")
        return ExecutionQueue.from_yaml(self.path)

    def enqueue_from_plan(
        self,
        run_id: str,
        plan: ExperimentPlan,
        requires_evidence_by_node: dict[str, list[str]] | None = None,
    ) -> ExecutionQueue:
        """Create and save a queue from an experiment plan."""
        queue = ExecutionQueue.from_experiment_plan(run_id, plan, requires_evidence_by_node)
        self.save(queue)
        return queue

    def next_runnable(self) -> ExecutionQueueItem | None:
        """Load the queue and return its next runnable item."""
        return self.load().next_runnable()

    def update_item(self, updated: ExecutionQueueItem) -> ExecutionQueue:
        """Replace one item and persist the queue."""
        queue = self.load()
        for index, item in enumerate(queue.items):
            if item.queue_id == updated.queue_id:
                queue.items[index] = updated
                self.save(queue)
                return queue
        raise KeyError(f"Queue item not found: {updated.queue_id}")


def queue_status_from_execution_status(status: str) -> QueueStatus:
    """Map executor statuses onto queue statuses."""
    if status in {"completed", "dry_run"}:
        return "completed"
    if status == "failed":
        return "failed"
    if status == "skipped":
        return "skipped"
    return "failed"
