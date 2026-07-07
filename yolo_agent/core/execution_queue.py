"""Persistent execution queue for materialized experiment nodes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.executor import ExecutionResult
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.core.resource_scheduler import ResourceDecision
from yolo_agent.core.yaml_io import YAMLModelMixin


QueueStatus = Literal[
    "queued",
    "running",
    "paused",
    "blocked_by_resource",
    "needs_resume",
    "completed",
    "failed",
    "skipped",
    "needs_evidence",
]


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
    resource_blockers: list[str] = Field(default_factory=list)
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

    def mark_resource_decision(self, decision: ResourceDecision) -> None:
        """Apply a scheduler decision before execution starts."""
        self.resource_blockers = list(decision.reasons)
        self.message = decision.message
        if decision.status == "runnable":
            self.status = "queued"
        elif decision.status == "paused":
            self.status = "paused"
        elif decision.status == "blocked_by_resource":
            self.status = "blocked_by_resource"
        elif decision.status == "needs_resume":
            self.status = "needs_resume"
        self.updated_at = datetime.now(timezone.utc)

    def refresh_evidence(self, missing_evidence: list[str]) -> bool:
        """Refresh evidence blockers and return whether the item changed."""
        if self.status != "needs_evidence":
            return False
        missing = list(dict.fromkeys(missing_evidence))
        if missing:
            changed = self.requires_evidence != missing or self.message != "Waiting for required evidence."
            self.requires_evidence = missing
            self.message = "Waiting for required evidence."
            if changed:
                self.updated_at = datetime.now(timezone.utc)
            return changed
        self.requires_evidence = []
        self.status = "queued"
        self.message = "Evidence requirements satisfied; item is queued."
        self.updated_at = datetime.now(timezone.utc)
        return True

    def refresh_resource_decision(self, decision: ResourceDecision) -> bool:
        """Refresh resource-paused states and return whether the item changed."""
        if self.status not in {"paused", "blocked_by_resource", "needs_resume"}:
            return False
        old_status = self.status
        old_blockers = list(self.resource_blockers)
        self.mark_resource_decision(decision)
        return old_status != self.status or old_blockers != self.resource_blockers

    def recover_stale_running(self, message: str) -> None:
        """Requeue a stale running item as a fresh attempt."""
        self.status = "queued"
        self.attempts = 0
        self.resource_blockers = []
        self.last_result = None
        self.result_artifact = None
        self.message = message
        self.updated_at = datetime.now(timezone.utc)

    def mark_interrupted(self, message: str = "Execution interrupted by user.") -> None:
        """Mark a running item as requiring explicit resume or recovery."""
        self.status = "needs_resume"
        self.resource_blockers = ["interrupted_by_user"]
        self.message = message
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
        max_nodes: int | None = None,
        requires_evidence_by_node: dict[str, list[str]] | None = None,
    ) -> "ExecutionQueue":
        """Materialize a queue from an experiment plan."""
        requirements = requires_evidence_by_node or {}
        node_count = len(plan.nodes)
        if max_nodes is not None and node_count > max_nodes:
            raise ValueError(
                f"ExecutionQueue exceeded max_nodes limit: {node_count} nodes > {max_nodes}."
            )
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
                "queue_source_plan_hash": plan.plan_hash(),
                "source_node_count": node_count,
            },
        )

    def counts(self) -> dict[QueueStatus, int]:
        """Return item counts by queue status."""
        counts: dict[QueueStatus, int] = {
            "queued": 0,
            "running": 0,
            "paused": 0,
            "blocked_by_resource": 0,
            "needs_resume": 0,
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

    def refresh_needs_evidence(self, missing_by_node: dict[str, list[str]]) -> dict[str, int]:
        """Refresh needs_evidence items using current missing evidence by node."""
        refreshed = 0
        unblocked = 0
        still_blocked = 0
        for item in self.items:
            if item.status != "needs_evidence":
                continue
            was_blocked = item.status == "needs_evidence"
            changed = item.refresh_evidence(missing_by_node.get(item.node_id, []))
            if changed:
                refreshed += 1
            if was_blocked and item.status == "queued":
                unblocked += 1
            elif item.status == "needs_evidence":
                still_blocked += 1
        if refreshed:
            self.refresh_updated_at()
        return {
            "refreshed": refreshed,
            "unblocked": unblocked,
            "still_blocked": still_blocked,
        }

    def refresh_resources(self, decisions_by_queue_id: dict[str, ResourceDecision]) -> dict[str, int]:
        """Refresh resource-blocked items using current scheduler decisions."""
        refreshed = 0
        unblocked = 0
        still_blocked = 0
        for item in self.items:
            if item.status not in {"paused", "blocked_by_resource", "needs_resume"}:
                continue
            decision = decisions_by_queue_id.get(item.queue_id)
            if decision is None:
                continue
            was_blocked = item.status != "queued"
            changed = item.refresh_resource_decision(decision)
            if changed:
                refreshed += 1
            if was_blocked and item.status == "queued":
                unblocked += 1
            elif item.status in {"paused", "blocked_by_resource", "needs_resume"}:
                still_blocked += 1
        if refreshed:
            self.refresh_updated_at()
        return {
            "refreshed": refreshed,
            "unblocked": unblocked,
            "still_blocked": still_blocked,
        }

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
        max_nodes: int | None = None,
        requires_evidence_by_node: dict[str, list[str]] | None = None,
    ) -> ExecutionQueue:
        """Create and save a queue from an experiment plan."""
        queue = ExecutionQueue.from_experiment_plan(
            run_id,
            plan,
            max_nodes=max_nodes,
            requires_evidence_by_node=requires_evidence_by_node,
        )
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

    def refresh_needs_evidence(self, missing_by_node: dict[str, list[str]]) -> tuple[ExecutionQueue, dict[str, int]]:
        """Refresh needs_evidence items and persist the queue."""
        queue = self.load()
        summary = queue.refresh_needs_evidence(missing_by_node)
        self.save(queue)
        return queue, summary

    def refresh_resources(self, decisions_by_queue_id: dict[str, ResourceDecision]) -> tuple[ExecutionQueue, dict[str, int]]:
        """Refresh resource-blocked items and persist the queue."""
        queue = self.load()
        summary = queue.refresh_resources(decisions_by_queue_id)
        self.save(queue)
        return queue, summary


def queue_status_from_execution_status(status: str) -> QueueStatus:
    """Map executor statuses onto queue statuses."""
    if status in {"completed", "dry_run"}:
        return "completed"
    if status == "failed":
        return "failed"
    if status == "skipped":
        return "skipped"
    return "failed"
