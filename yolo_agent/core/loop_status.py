"""User-facing status aggregation for loop harness runs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueItem
from yolo_agent.core.experiment_graph import Evidence, MetricValue
from yolo_agent.core.loop_state import LoopState, StageStatus
from yolo_agent.core.run_context import RunContext


KEY_STATUS_METRICS = (
    "map50_95",
    "map50",
    "AP_small",
    "AP_medium",
    "AP_large",
    "precision",
    "recall",
    "latency_ms",
    "model_size_mb",
    "smoke_passed",
    "execution_timed_out",
    "execution_timeout_seconds",
)


class QueueItemStatus(BaseModel):
    """Compact status for one queue item."""

    queue_id: str
    node_id: str
    candidate_id: str
    status: str
    command_type: str
    command: str
    message: str = ""
    requires_evidence: list[str] = Field(default_factory=list)
    resource_blockers: list[str] = Field(default_factory=list)


class EvidenceStatusSummary(BaseModel):
    """Evidence counts and selected trusted metrics for a run."""

    run_metrics: int = 0
    metric_records: int = 0
    verified_metric_records: int = 0
    artifacts: int = 0
    artifact_manifest_entries: int = 0
    key_metrics: dict[str, MetricValue] = Field(default_factory=dict)


class LoopRunStatus(BaseModel):
    """A read-only snapshot for a loop run."""

    run_id: str
    run_dir: Path
    current_stage: str = "unknown"
    current_stage_status: StageStatus = "pending"
    completed: list[str] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    queue_counts: dict[str, int] = Field(default_factory=dict)
    current_training_command: str = ""
    current_queue_item: QueueItemStatus | None = None
    next_queue_item: QueueItemStatus | None = None
    evidence: EvidenceStatusSummary = Field(default_factory=EvidenceStatusSummary)
    blocked_reason: str = ""
    next_command: str = ""


def load_loop_status(run_dir: Path | str) -> LoopRunStatus:
    """Load and aggregate run status without mutating run state."""
    context = RunContext.from_run_dir(run_dir)
    state = LoopState.from_yaml(context.run_dir / "loop_state.yaml")
    evidence = EvidenceStore(context.run_root).load_run(context.run_id)
    queue = _load_queue(context.run_dir)
    current_stage_state = state.stages.get(state.current_stage)
    current_item = _current_item(queue)
    next_item = _next_item(queue)
    queue_counts = {key: int(value) for key, value in queue.counts().items()} if queue is not None else {}
    return LoopRunStatus(
        run_id=context.run_id,
        run_dir=context.run_dir,
        current_stage=state.current_stage,
        current_stage_status=current_stage_state.status if current_stage_state is not None else "pending",
        completed=[str(stage) for stage in state.completed],
        pending=[str(stage) for stage in state.pending],
        blocked=list(state.blocked),
        failed=[str(stage) for stage in state.failed],
        queue_counts=queue_counts,
        current_training_command=_current_training_command(current_item),
        current_queue_item=_queue_item_status(current_item),
        next_queue_item=_queue_item_status(next_item),
        evidence=_evidence_summary(evidence),
        blocked_reason=_blocked_reason(state, queue),
        next_command=_next_command(context, state, queue),
    )


def render_loop_status(status: LoopRunStatus) -> str:
    """Render a compact terminal status panel."""
    lines = [
        f"run_id={status.run_id}",
        f"run_dir={status.run_dir}",
        f"current_stage={status.current_stage} status={status.current_stage_status}",
        f"completed={_csv(status.completed)}",
        f"pending={_csv(status.pending[:5])}",
        f"failed={_csv(status.failed)}",
        f"blocked_reason={status.blocked_reason or 'none'}",
        "queue " + _format_counts(status.queue_counts),
        f"current_training_command={status.current_training_command or 'none'}",
    ]
    if status.current_queue_item is not None:
        lines.append(_format_queue_item("current_queue_item", status.current_queue_item))
    if status.next_queue_item is not None and status.next_queue_item != status.current_queue_item:
        lines.append(_format_queue_item("next_queue_item", status.next_queue_item))
    lines.extend(
        [
            (
                "evidence "
                f"run_metrics={status.evidence.run_metrics} "
                f"metric_records={status.evidence.metric_records} "
                f"verified_metric_records={status.evidence.verified_metric_records} "
                f"artifacts={status.evidence.artifacts} "
                f"artifact_manifest_entries={status.evidence.artifact_manifest_entries}"
            ),
            f"evidence.key_metrics={_format_metrics(status.evidence.key_metrics)}",
            f"next_command={status.next_command or 'none'}",
        ]
    )
    return "\n".join(lines)


def _load_queue(run_dir: Path) -> ExecutionQueue | None:
    queue_path = run_dir / "execution_queue.yaml"
    if not queue_path.is_file():
        return None
    return ExecutionQueue.from_yaml(queue_path)


def _current_item(queue: ExecutionQueue | None) -> ExecutionQueueItem | None:
    if queue is None:
        return None
    for status in ("running", "paused", "blocked_by_resource", "needs_resume"):
        for item in queue.items:
            if item.status == status:
                return item
    return None


def _next_item(queue: ExecutionQueue | None) -> ExecutionQueueItem | None:
    if queue is None:
        return None
    for status in ("queued", "needs_evidence", "failed"):
        for item in queue.items:
            if item.status == status:
                return item
    return None


def _queue_item_status(item: ExecutionQueueItem | None) -> QueueItemStatus | None:
    if item is None:
        return None
    return QueueItemStatus(
        queue_id=item.queue_id,
        node_id=item.node_id,
        candidate_id=item.candidate_id,
        status=item.status,
        command_type=item.command.command_type,
        command=item.command.display(),
        message=item.message,
        requires_evidence=list(item.requires_evidence),
        resource_blockers=list(item.resource_blockers),
    )


def _current_training_command(item: ExecutionQueueItem | None) -> str:
    if item is None or item.status != "running" or item.command.command_type != "train":
        return ""
    return item.command.display()


def _evidence_summary(evidence: Evidence) -> EvidenceStatusSummary:
    index = EvidenceIndex(evidence.metric_records)
    key_metrics: dict[str, MetricValue] = {}
    for metric_name in KEY_STATUS_METRICS:
        value = index.metric_value(metric_name=metric_name, verified=True)
        if value is None:
            value = evidence.metrics.get(metric_name)
        if value is not None:
            key_metrics[metric_name] = value
    return EvidenceStatusSummary(
        run_metrics=len(evidence.metrics),
        metric_records=len(evidence.metric_records),
        verified_metric_records=sum(1 for record in evidence.metric_records if record.verified),
        artifacts=len(evidence.artifacts),
        artifact_manifest_entries=len(evidence.artifact_manifest),
        key_metrics=key_metrics,
    )


def _blocked_reason(state: LoopState, queue: ExecutionQueue | None) -> str:
    reasons = list(state.blocked)
    if queue is not None:
        for item in queue.items:
            if item.status == "needs_evidence":
                missing = ", ".join(item.requires_evidence) if item.requires_evidence else "required evidence"
                reasons.append(f"{item.node_id}: needs_evidence({missing})")
            elif item.status in {"paused", "blocked_by_resource", "needs_resume", "failed"}:
                message = item.message or ", ".join(item.resource_blockers) or item.status
                reasons.append(f"{item.node_id}: {item.status}({message})")
    return "; ".join(dict.fromkeys(reason for reason in reasons if reason))


def _next_command(context: RunContext, state: LoopState, queue: ExecutionQueue | None) -> str:
    run_arg = str(context.run_dir)
    if queue is not None:
        counts = queue.counts()
        next_item = _next_item(queue)
        if counts.get("running", 0):
            return f"yolo-agent loop status --run {run_arg}"
        if counts.get("queued", 0) and next_item is not None:
            executor = "ultralytics-train" if next_item.command.command_type == "train" else "dry-run"
            return f"yolo-agent loop execute --run {run_arg} --executor {executor}"
        if counts.get("needs_evidence", 0):
            return f"yolo-agent loop queue-refresh --run {run_arg}"
        if any(counts.get(status, 0) for status in ("paused", "blocked_by_resource", "needs_resume")):
            return f"yolo-agent loop queue-refresh --run {run_arg}"
        if counts.get("failed", 0):
            return f"yolo-agent loop status --run {run_arg}"

    if (context.artifact_path("experiment_plan.yaml")).is_file() and queue is None:
        return f"yolo-agent loop enqueue --run {run_arg}"
    if state.blocked:
        blocked = " ".join(state.blocked)
        if "missing_metrics" in blocked:
            return f"yolo-agent loop ingest-metrics --run {run_arg} --metrics results.csv"
        if "missing_detection_errors" in blocked:
            return f"yolo-agent loop diagnose --run {run_arg} --errors errors.yaml"
        return f"yolo-agent loop --run {run_arg} --resume"
    next_stage = state.next_pending()
    if next_stage is not None:
        profile = str(context.metadata.get("training_profile", "debug"))
        if next_stage in {"profile_data", "report", "next_round"}:
            return f"yolo-agent loop train --run {run_arg} --profile {profile} --executor dry-run"
        return f"yolo-agent loop run-stage --run {run_arg} --stage {next_stage}"
    return ""


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return " ".join(f"{name}={counts.get(name, 0)}" for name in sorted(counts))


def _format_metrics(metrics: dict[str, MetricValue]) -> str:
    if not metrics:
        return "none"
    return " ".join(f"{name}={value}" for name, value in sorted(metrics.items()))


def _format_queue_item(prefix: str, item: QueueItemStatus) -> str:
    details = [
        f"{prefix}.status={item.status}",
        f"node={item.node_id}",
        f"candidate={item.candidate_id}",
        f"type={item.command_type}",
    ]
    if item.requires_evidence:
        details.append(f"requires_evidence={','.join(item.requires_evidence)}")
    if item.resource_blockers:
        details.append(f"resource_blockers={','.join(item.resource_blockers)}")
    if item.message:
        details.append(f"message={item.message}")
    details.append(f"command={item.command}")
    return " ".join(details)


def _csv(values: list[str]) -> str:
    return ",".join(values) if values else "none"
