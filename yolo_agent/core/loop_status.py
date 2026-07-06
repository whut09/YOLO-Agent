"""User-facing status aggregation for loop harness runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    profile: str = ""
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


class TrainingHeartbeat(BaseModel):
    """Live-ish training progress parsed from stream artifacts."""

    node_id: str = ""
    candidate_id: str = ""
    epoch: int | None = None
    total_epochs: int | None = None
    it_per_sec: float | None = None
    gpu_util_percent: float | None = None
    eta: str = ""
    recent_log_lines: list[str] = Field(default_factory=list)


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
    training_heartbeat: TrainingHeartbeat | None = None
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
        training_heartbeat=_training_heartbeat(context.run_dir, current_item),
        current_queue_item=_queue_item_status(current_item),
        next_queue_item=_queue_item_status(next_item),
        evidence=_evidence_summary(evidence),
        blocked_reason=_blocked_reason(state, queue),
        next_command=_next_command(context, state, queue),
    )


def render_loop_status(status: LoopRunStatus) -> str:
    """Render a compact terminal status panel."""
    lines = [
        *_human_summary(status),
        "",
        "machine_status:",
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
    if status.training_heartbeat is not None:
        lines.append(_format_training_heartbeat(status.training_heartbeat))
        for index, line in enumerate(status.training_heartbeat.recent_log_lines, start=1):
            lines.append(f"training_log.{index}={line}")
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
        profile=str(item.command.metadata.get("training_budget_profile") or item.command.metadata.get("profile") or ""),
        message=item.message,
        requires_evidence=list(item.requires_evidence),
        resource_blockers=list(item.resource_blockers),
    )


def _current_training_command(item: ExecutionQueueItem | None) -> str:
    if item is None or item.status != "running" or item.command.command_type != "train":
        return ""
    return item.command.display()


def _training_heartbeat(run_dir: Path, item: ExecutionQueueItem | None) -> TrainingHeartbeat | None:
    if item is None or item.status != "running" or item.command.command_type != "train":
        return None
    artifacts_dir = run_dir / "artifacts"
    stdout_log = artifacts_dir / f"{item.node_id}_ultralytics_stdout.log"
    runtime_jsonl = artifacts_dir / f"{item.node_id}_runtime_profile.jsonl"
    recent_lines = _tail_text_lines(stdout_log, limit=3)
    runtime_records = _read_runtime_records(runtime_jsonl, limit=200)
    total_epochs = _total_epochs(item)
    epoch = _latest_epoch([*recent_lines, *[str(record.get("line", "")) for record in runtime_records]], total_epochs)
    it_per_sec = _latest_metric(runtime_records, "runtime_stream_it_per_sec")
    gpu_util = _latest_gpu_util(runtime_records)
    eta = _latest_eta(recent_lines)
    if not eta:
        eta = _estimated_eta(item, epoch, total_epochs)
    return TrainingHeartbeat(
        node_id=item.node_id,
        candidate_id=item.candidate_id,
        epoch=epoch,
        total_epochs=total_epochs,
        it_per_sec=it_per_sec,
        gpu_util_percent=gpu_util,
        eta=eta,
        recent_log_lines=recent_lines,
    )


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


def _format_training_heartbeat(heartbeat: TrainingHeartbeat) -> str:
    epoch = "unknown"
    if heartbeat.epoch is not None and heartbeat.total_epochs is not None:
        epoch = f"{heartbeat.epoch}/{heartbeat.total_epochs}"
    elif heartbeat.epoch is not None:
        epoch = str(heartbeat.epoch)
    return (
        "training_heartbeat "
        f"node={heartbeat.node_id} "
        f"candidate={heartbeat.candidate_id} "
        f"epoch={epoch} "
        f"it_per_sec={_value_or_unknown(heartbeat.it_per_sec)} "
        f"gpu_util_percent={_value_or_unknown(heartbeat.gpu_util_percent)} "
        f"eta={heartbeat.eta or 'unknown'}"
    )


def _human_summary(status: LoopRunStatus) -> list[str]:
    return [
        f"当前状态：{_human_current_state(status)}",
        f"进度：{_human_progress(status)}",
        f"当前可信结论：{_human_trust(status)}",
        f"下一步：{_human_next_step(status)}",
    ]


def _human_current_state(status: LoopRunStatus) -> str:
    item = status.current_queue_item or status.next_queue_item
    if item is not None:
        profile = item.profile or item.command_type
        if item.status == "running" and item.command_type == "train":
            return f"{profile} 正在训练"
        if item.status == "queued":
            return f"{profile} 等待执行"
        if item.status == "needs_evidence":
            return f"{profile} 等待补齐 evidence"
        if item.status == "blocked_by_resource":
            return f"{profile} 被资源约束阻塞"
        if item.status == "needs_resume":
            return f"{profile} 需要 resume"
        if item.status == "failed":
            return f"{profile} 执行失败"
    if status.blocked_reason:
        return f"{status.current_stage} 已阻塞"
    if status.failed:
        return f"{status.current_stage} 失败"
    return f"{status.current_stage} {status.current_stage_status}"


def _human_progress(status: LoopRunStatus) -> str:
    heartbeat = status.training_heartbeat
    if heartbeat is not None:
        parts: list[str] = []
        if heartbeat.epoch is not None and heartbeat.total_epochs is not None:
            parts.append(f"epoch {heartbeat.epoch}/{heartbeat.total_epochs}")
        elif heartbeat.epoch is not None:
            parts.append(f"epoch {heartbeat.epoch}")
        if heartbeat.gpu_util_percent is not None:
            parts.append(f"GPU {heartbeat.gpu_util_percent:g}%")
        if heartbeat.it_per_sec is not None:
            parts.append(f"{heartbeat.it_per_sec:g} it/s")
        if heartbeat.eta:
            parts.append(f"预计剩余 {heartbeat.eta}")
        return "，".join(parts) if parts else "训练已启动，等待日志心跳"
    if status.current_queue_item is not None:
        return status.current_queue_item.message or f"queue item {status.current_queue_item.status}"
    if status.queue_counts:
        return "queue " + _format_counts(status.queue_counts)
    if status.pending:
        return f"等待 stage {status.pending[0]}"
    return "暂无运行中的任务"


def _human_trust(status: LoopRunStatus) -> str:
    item = status.current_queue_item or status.next_queue_item
    profile = item.profile if item is not None else ""
    if profile == "debug":
        return "暂无，debug 只验证链路，不能作为效果结论"
    if profile == "pilot":
        return "暂无，pilot 只能作为初步证据，不能作为最终结论"
    if status.evidence.verified_metric_records <= 0:
        return "暂无，需要 verified metric evidence"
    metrics = _format_metrics(status.evidence.key_metrics)
    if metrics == "none":
        return f"已有 {status.evidence.verified_metric_records} 条 verified metric evidence，等待报告汇总"
    return f"已有 verified evidence：{metrics}"


def _human_next_step(status: LoopRunStatus) -> str:
    item = status.current_queue_item
    if item is not None and item.status == "running":
        if item.command_type == "train":
            return "等待训练完成；完成后自动导入 evidence"
        return "等待当前队列项完成"
    if status.blocked_reason:
        return status.next_command or "先处理 blocked reason"
    if status.next_command:
        return status.next_command
    return "暂无下一步，当前 run 可能已经完成"


def _format_queue_item(prefix: str, item: QueueItemStatus) -> str:
    details = [
        f"{prefix}.status={item.status}",
        f"node={item.node_id}",
        f"candidate={item.candidate_id}",
        f"type={item.command_type}",
    ]
    if item.profile:
        details.append(f"profile={item.profile}")
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


def _tail_text_lines(path: Path, limit: int) -> list[str]:
    if not path.is_file():
        return []
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return lines[-limit:]


def _read_runtime_records(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        text = line.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def _total_epochs(item: ExecutionQueueItem) -> int | None:
    raw = item.command.metadata.get("training_budget_epochs")
    if isinstance(raw, (int, float)):
        return int(raw)
    for arg in item.command.argv:
        if arg.startswith("epochs="):
            try:
                return int(float(arg.split("=", 1)[1]))
            except ValueError:
                return None
    return None


def _latest_epoch(lines: list[str], total_epochs: int | None) -> int | None:
    for line in reversed(lines):
        for match in re.finditer(r"(?<!\d)(?P<current>\d+)\s*/\s*(?P<total>\d+)(?!\d)", line):
            current = int(match.group("current"))
            total = int(match.group("total"))
            if total_epochs is None or total == total_epochs:
                return current
    return None


def _latest_metric(records: list[dict[str, Any]], metric_name: str) -> float | None:
    for record in reversed(records):
        metrics = record.get("metrics")
        if isinstance(metrics, dict) and metric_name in metrics:
            return _float_or_none(metrics.get(metric_name))
    return None


def _latest_gpu_util(records: list[dict[str, Any]]) -> float | None:
    for record in reversed(records):
        sample = record.get("sample")
        if isinstance(sample, dict):
            value = _float_or_none(sample.get("gpu_util_percent"))
            if value is not None:
                return value
    return None


def _latest_eta(lines: list[str]) -> str:
    pattern = re.compile(r"\[[^\]<]*<(?P<eta>[^,\]]+)")
    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            return match.group("eta").strip()
    return ""


def _estimated_eta(item: ExecutionQueueItem, epoch: int | None, total_epochs: int | None) -> str:
    if epoch is None or total_epochs is None or epoch <= 0 or epoch >= total_epochs:
        return ""
    elapsed = (datetime.now(timezone.utc) - item.updated_at).total_seconds()
    if elapsed <= 0:
        return ""
    seconds_per_epoch = elapsed / epoch
    return _format_duration(seconds_per_epoch * (total_epochs - epoch))


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value_or_unknown(value: float | None) -> str:
    if value is None:
        return "unknown"
    return str(round(value, 6))
