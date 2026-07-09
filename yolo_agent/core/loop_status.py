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
from yolo_agent.core.process_probe import ProcessProbeResult, probe_command_process
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


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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
    phase: str = ""
    progress_current: int | None = None
    progress_total: int | None = None
    progress_percent: float | None = None
    epoch: int | None = None
    total_epochs: int | None = None
    it_per_sec: float | None = None
    gpu_util_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    process_status: str = "unknown"
    process_detail: str = ""
    last_log_age_seconds: float | None = None
    last_sample_age_seconds: float | None = None
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


def render_loop_status(status: LoopRunStatus, verbose: bool = False) -> str:
    """Render a compact terminal status panel."""
    if not verbose:
        return _render_human_loop_status(status)
    return _render_verbose_loop_status(status)


def _render_human_loop_status(status: LoopRunStatus) -> str:
    """Render the default readable status panel."""
    lines = [
        "YOLO Agent Status",
        "-----------------",
        f"Run:        {status.run_id}",
        f"State:      {_human_current_state(status)}",
        f"Progress:   {_human_progress(status)}",
        f"Trust:      {_human_trust(status)}",
    ]
    if status.current_queue_item is not None:
        lines.extend(
            [
                "",
                "Active item",
                f"  Profile:   {status.current_queue_item.profile or 'unknown'}",
                f"  Candidate: {status.current_queue_item.candidate_id}",
                f"  Node:      {status.current_queue_item.node_id}",
                f"  Queue:     {_format_active_counts(status.queue_counts)}",
            ]
        )
        if status.current_queue_item.status == "running" or status.training_heartbeat is not None:
            lines.append(f"  Process:   {_format_process_status(status.training_heartbeat)}")
    if status.training_heartbeat is not None:
        clean_logs = [_clean_terminal_line(line, limit=120) for line in status.training_heartbeat.recent_log_lines]
        clean_logs = [line for line in clean_logs if line]
        if clean_logs:
            lines.extend(["", "Recent training log"])
            lines.extend(f"  {line}" for line in clean_logs[-3:])
    if status.blocked_reason and not _has_queue_work(status):
        lines.extend(["", f"Blocked:    {_clean_terminal_line(status.blocked_reason, limit=160)}"])
    next_text = _human_next_step(status) if _has_running_queue(status) else status.next_command or _human_next_step(status)
    lines.extend(
        [
            "",
            f"Next:       {next_text}",
            "",
            "Details:    add --verbose for queue, evidence, and full command fields.",
        ]
    )
    return "\n".join(lines)


def _render_verbose_loop_status(status: LoopRunStatus) -> str:
    """Render the full machine-oriented status panel."""
    lines = [
        "YOLO Agent Status (verbose)",
        "---------------------------",
        f"Run:        {status.run_id}",
        f"Run dir:    {status.run_dir}",
        f"State:      {_human_current_state(status)}",
        f"Progress:   {_human_progress(status)}",
        f"Trust:      {_human_trust(status)}",
        f"Next:       {_human_next_step(status) if _has_running_queue(status) else status.next_command or _human_next_step(status)}",
        "",
        "Loop",
        f"  Stage:     {status.current_stage} ({status.current_stage_status})",
        f"  Completed: {_csv(status.completed)}",
        f"  Pending:   {_csv(status.pending[:5])}",
        f"  Failed:    {_csv(status.failed)}",
        f"  Blocked:   {status.blocked_reason or 'none'}",
        "",
        "Queue",
        f"  Counts:    {_format_counts(status.queue_counts)}",
    ]
    if status.training_heartbeat is not None:
        lines.extend(
            [
                "",
                "Training",
                f"  Heartbeat: {_format_training_heartbeat(status.training_heartbeat)}",
                f"  Process:   {_format_process_status(status.training_heartbeat)}",
            ]
        )
        for index, line in enumerate(status.training_heartbeat.recent_log_lines, start=1):
            if index == 1:
                lines.append("  Recent log:")
            lines.append(f"    {index}. {_clean_terminal_line(line, limit=140)}")
    if status.current_queue_item is not None:
        lines.extend(["", "Current item", *_format_queue_item_lines(status.current_queue_item)])
    if status.next_queue_item is not None and status.next_queue_item != status.current_queue_item:
        lines.extend(["", "Next item", *_format_queue_item_lines(status.next_queue_item)])
    lines.extend(
        [
            "",
            "Evidence",
            f"  Run metrics:       {status.evidence.run_metrics}",
            f"  Metric records:    {status.evidence.metric_records}",
            f"  Verified records:  {status.evidence.verified_metric_records}",
            f"  Artifacts:         {status.evidence.artifacts}",
            f"  Manifest entries:  {status.evidence.artifact_manifest_entries}",
            f"  Key metrics:       {_format_metrics(status.evidence.key_metrics)}",
            "",
            f"Next command: {status.next_command or 'none'}",
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
    for status in ("queued", "needs_evidence", "skipped", "failed"):
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
    process_probe = probe_command_process(item.command)
    total_epochs = _total_epochs(item)
    parse_lines = [
        *[_clean_terminal_line(line, limit=500) for line in recent_lines],
        *[_clean_terminal_line(str(record.get("line", "")), limit=500) for record in runtime_records],
    ]
    epoch = _latest_epoch(parse_lines, total_epochs)
    progress = _latest_progress(parse_lines)
    it_per_sec = _latest_metric(runtime_records, "runtime_stream_it_per_sec")
    gpu_util = _latest_gpu_util(runtime_records)
    gpu_memory_used = _latest_sample_value(runtime_records, "gpu_memory_used_mb")
    gpu_memory_total = _latest_sample_value(runtime_records, "gpu_memory_total_mb")
    last_sample_age = _latest_record_age_seconds(runtime_records)
    last_log_age = _file_age_seconds(stdout_log)
    eta = _latest_eta(parse_lines)
    if not eta:
        eta = _estimated_eta(item, epoch, total_epochs)
    return TrainingHeartbeat(
        node_id=item.node_id,
        candidate_id=item.candidate_id,
        phase=progress["phase"],
        progress_current=progress["current"],
        progress_total=progress["total"],
        progress_percent=progress["percent"],
        epoch=epoch,
        total_epochs=total_epochs,
        it_per_sec=it_per_sec,
        gpu_util_percent=gpu_util,
        gpu_memory_used_mb=gpu_memory_used,
        gpu_memory_total_mb=gpu_memory_total,
        process_status=process_probe.status,
        process_detail=process_probe.detail,
        last_log_age_seconds=last_log_age,
        last_sample_age_seconds=last_sample_age,
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
                if item.status == "failed" and item.last_result is not None:
                    detail = _last_error_line(item.last_result.stderr or item.last_result.stdout)
                    if detail:
                        message = f"{message}: {detail}"
                reasons.append(f"{item.node_id}: {item.status}({message})")
    return "; ".join(dict.fromkeys(reason for reason in reasons if reason))


def _next_command(context: RunContext, state: LoopState, queue: ExecutionQueue | None) -> str:
    run_arg = str(context.run_dir)
    if queue is not None:
        counts = queue.counts()
        next_item = _next_item(queue)
        if counts.get("running", 0):
            return f"yolo-agent status --run {run_arg}"
        if counts.get("queued", 0) and next_item is not None:
            if next_item.command.command_type == "train":
                return _optimize_command_for_item(context, next_item)
            return f"yolo-agent loop execute --run {run_arg} --executor dry-run"
        if counts.get("needs_evidence", 0):
            current_item = _current_item(queue)
            if current_item is not None and current_item.command.command_type == "train":
                return _optimize_command_for_item(context, current_item)
            return f"yolo-agent status --run {run_arg}"
        if any(counts.get(status, 0) for status in ("paused", "blocked_by_resource", "needs_resume")):
            current_item = _current_item(queue)
            if (
                current_item is not None
                and current_item.command.command_type == "train"
                and "missing_batch_tuning_result" in current_item.resource_blockers
            ):
                return _optimize_command_for_item(context, current_item)
            return f"yolo-agent status --run {run_arg}"
        if counts.get("skipped", 0):
            skipped_item = _next_item(queue)
            if skipped_item is not None and skipped_item.command.command_type == "train":
                return _optimize_command_for_item(context, skipped_item)
            return _train_command_for_context(context, queue)
        if counts.get("failed", 0):
            failed_item = _next_item(queue)
            if failed_item is not None and failed_item.command.command_type == "train":
                return _optimize_command_for_item(context, failed_item)
            return _train_command_for_context(context, queue)

    if (context.artifact_path("experiment_plan.yaml")).is_file() and queue is None:
        return _train_command_for_context(context, queue)
    if state.blocked:
        blocked = " ".join(state.blocked)
        if "missing_metrics" in blocked or "missing_detection_errors" in blocked:
            return _train_command_for_context(context, queue)
        return _train_command_for_context(context, queue)
    next_stage = state.next_pending()
    if next_stage is not None:
        return _train_command_for_context(context, queue)
    return ""


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return " ".join(f"{name}={counts.get(name, 0)}" for name in sorted(counts))


def _format_active_counts(counts: dict[str, int]) -> str:
    """Return only non-zero queue counts for the human panel."""
    active = {name: value for name, value in sorted(counts.items()) if value}
    if not active:
        return "none"
    return " ".join(f"{name}={value}" for name, value in active.items())


def _format_process_status(heartbeat: TrainingHeartbeat | None) -> str:
    """Return a compact process liveness summary."""
    if heartbeat is None:
        return "unknown"
    if heartbeat.process_status == "not_found" and _heartbeat_is_fresh(heartbeat):
        return f"log heartbeat active ({heartbeat.process_detail})"
    if heartbeat.process_status == "found":
        return f"found ({heartbeat.process_detail})"
    if heartbeat.process_status == "not_found":
        return f"not found ({heartbeat.process_detail})"
    return f"unknown ({heartbeat.process_detail or 'probe unavailable'})"


def _has_running_queue(status: LoopRunStatus) -> bool:
    """Return whether status represents an actively running queue item."""
    return status.queue_counts.get("running", 0) > 0


def _has_queue_work(status: LoopRunStatus) -> bool:
    """Return whether the queue has user-actionable work."""
    return any(
        status.queue_counts.get(name, 0) > 0
        for name in ("queued", "running", "paused", "blocked_by_resource", "needs_resume", "needs_evidence")
    )


def _last_error_line(text: str) -> str:
    """Return the last useful line from a captured process stream."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _optimize_command_for_item(context: RunContext, item: ExecutionQueueItem) -> str:
    """Return the beginner-facing train command for a train queue item."""
    profile = str(
        item.command.metadata.get("training_budget_profile")
        or item.command.metadata.get("profile")
        or context.metadata.get("training_profile", "debug")
    )
    model = item.experiment_node.candidate_config.base_model
    kind = "coco" if "coco" in context.run_id.lower() or context.dataset_version.startswith("coco") else "custom"
    command = (
        f"yolo-agent train --kind {kind} --model {model} --data {context.data_yaml} "
        f"--run-id {context.run_id} --run-root {context.run_dir.parent} --profile {profile}"
    )
    if profile in {"baseline_full", "baseline_confirm", "candidate_full"}:
        command += " --confirm-full-run"
    return command


def _train_command_for_context(context: RunContext, queue: ExecutionQueue | None) -> str:
    """Return a one-command train invocation for continuing a run."""
    profile = str(context.metadata.get("training_profile", "debug"))
    model = "yolo26n.pt"
    if queue is not None:
        for item in queue.items:
            if item.command.command_type != "train":
                continue
            model = item.experiment_node.candidate_config.base_model
            profile = str(
                item.command.metadata.get("training_budget_profile")
                or item.command.metadata.get("profile")
                or profile
            )
            break
    kind = "coco" if "coco" in context.run_id.lower() or context.dataset_version.startswith("coco") else "custom"
    command = (
        f"yolo-agent train --kind {kind} --model {model} --data {context.data_yaml} "
        f"--run-id {context.run_id} --run-root {context.run_dir.parent} --profile {profile}"
    )
    if profile in {"baseline_full", "baseline_confirm", "candidate_full"}:
        command += " --confirm-full-run"
    return command


def _format_metrics(metrics: dict[str, MetricValue]) -> str:
    if not metrics:
        return "none"
    return " ".join(f"{name}={value}" for name, value in sorted(metrics.items()))


def _format_age(seconds: float) -> str:
    """Return a compact age string."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _format_mb(value: float) -> str:
    """Return MB/GB display text."""
    if value >= 1024:
        return f"{value / 1024:.1f}GB"
    return f"{value:.0f}MB"


def _clean_terminal_line(text: str, limit: int = 120) -> str:
    """Return a terminal-safe single-line preview."""
    cleaned = ANSI_ESCAPE_RE.sub("", text)
    cleaned = CONTROL_CHARS_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = "".join(char if _is_terminal_safe_char(char) else " " for char in cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _is_terminal_safe_char(char: str) -> bool:
    """Keep conservative printable characters for Windows terminal status output."""
    if char in {"\t", " "}:
        return True
    codepoint = ord(char)
    return 32 <= codepoint <= 126


def _format_training_heartbeat(heartbeat: TrainingHeartbeat) -> str:
    epoch = "unknown"
    if heartbeat.epoch is not None and heartbeat.total_epochs is not None:
        epoch = f"{heartbeat.epoch}/{heartbeat.total_epochs}"
    elif heartbeat.epoch is not None:
        epoch = str(heartbeat.epoch)
    progress = "unknown"
    if heartbeat.phase and heartbeat.progress_current is not None and heartbeat.progress_total is not None:
        progress = f"{heartbeat.phase}:{heartbeat.progress_current}/{heartbeat.progress_total}"
        if heartbeat.progress_percent is not None:
            progress += f"({heartbeat.progress_percent:g}%)"
    return (
        f"node={heartbeat.node_id} "
        f"candidate={heartbeat.candidate_id} "
        f"progress={progress} "
        f"epoch={epoch} "
        f"it/s={_value_or_unknown(heartbeat.it_per_sec)} "
        f"gpu={_value_or_unknown(heartbeat.gpu_util_percent)}% "
        f"mem={_value_or_unknown(heartbeat.gpu_memory_used_mb)}MB "
        f"eta={heartbeat.eta or 'unknown'}"
    )


def _heartbeat_is_fresh(heartbeat: TrainingHeartbeat) -> bool:
    """Return whether logs or runtime samples show recent activity."""
    ages = [
        age
        for age in (heartbeat.last_log_age_seconds, heartbeat.last_sample_age_seconds)
        if age is not None
    ]
    return bool(ages) and min(ages) <= 60.0


def _human_summary(status: LoopRunStatus) -> list[str]:
    return [
        f"Current status: {_human_current_state(status)}",
        f"Progress: {_human_progress(status)}",
        f"Trusted conclusion: {_human_trust(status)}",
        f"Next step: {_human_next_step(status)}",
    ]


def _human_current_state(status: LoopRunStatus) -> str:
    item = status.current_queue_item or status.next_queue_item
    if item is not None:
        profile = item.profile or item.command_type
        if item.status == "running" and item.command_type == "train":
            if (
                status.training_heartbeat is not None
                and status.training_heartbeat.process_status == "not_found"
                and not _heartbeat_is_fresh(status.training_heartbeat)
            ):
                return f"{profile} stale: no training process detected"
            return f"{profile} training is running"
        if item.status == "queued":
            return f"{profile} is queued"
        if item.status == "needs_evidence":
            return f"{profile} is waiting for evidence"
        if item.status == "blocked_by_resource":
            if "missing_batch_tuning_result" in item.resource_blockers:
                return f"{profile} needs batch tuning"
            return f"{profile} is blocked by resource limits"
        if item.status == "needs_resume":
            return f"{profile} needs resume"
        if item.status == "failed":
            return f"{profile} execution failed"
        if item.status == "skipped":
            return f"{profile} was skipped by a guard"
    if status.blocked_reason:
        return f"{status.current_stage} is blocked"
    if status.failed:
        return f"{status.current_stage} failed"
    return f"{status.current_stage} {status.current_stage_status}"


def _human_progress(status: LoopRunStatus) -> str:
    heartbeat = status.training_heartbeat
    if heartbeat is not None:
        if heartbeat.process_status == "not_found" and not _heartbeat_is_fresh(heartbeat):
            parts = ["no matching training process"]
            if heartbeat.last_log_age_seconds is not None:
                parts.append(f"last log {_format_age(heartbeat.last_log_age_seconds)} ago")
            if heartbeat.gpu_memory_used_mb is not None:
                parts.append(f"GPU memory {_format_mb(heartbeat.gpu_memory_used_mb)}")
            if heartbeat.gpu_util_percent is not None:
                parts.append(f"GPU util {heartbeat.gpu_util_percent:g}%")
            return ", ".join(parts)
        parts: list[str] = []
        batch_tuning = _batch_tuning_label(heartbeat.process_detail)
        if batch_tuning:
            parts.append(f"pre-training batch tuning {batch_tuning}")
        if heartbeat.phase and heartbeat.progress_current is not None and heartbeat.progress_total is not None:
            progress = f"{heartbeat.phase} {heartbeat.progress_current}/{heartbeat.progress_total}"
            if heartbeat.progress_percent is not None:
                progress += f" ({heartbeat.progress_percent:g}%)"
            parts.append(progress)
        if heartbeat.epoch is not None and heartbeat.total_epochs is not None:
            parts.append(f"epoch {heartbeat.epoch}/{heartbeat.total_epochs}")
        elif heartbeat.epoch is not None:
            parts.append(f"epoch {heartbeat.epoch}")
        if heartbeat.gpu_util_percent is not None:
            parts.append(f"GPU {heartbeat.gpu_util_percent:g}%")
        if heartbeat.gpu_memory_used_mb is not None:
            parts.append(f"mem {_format_mb(heartbeat.gpu_memory_used_mb)}")
        if heartbeat.it_per_sec is not None:
            parts.append(f"{heartbeat.it_per_sec:g} it/s")
        if heartbeat.eta:
            parts.append(f"ETA {heartbeat.eta}")
        if heartbeat.last_log_age_seconds is not None:
            parts.append(f"log {_format_age(heartbeat.last_log_age_seconds)} ago")
        return ", ".join(parts) if parts else "process is running; waiting for Ultralytics output"
    if status.current_queue_item is not None:
        if (
            status.current_queue_item.status == "blocked_by_resource"
            and "missing_batch_tuning_result" in status.current_queue_item.resource_blockers
        ):
            return "training is not running; pilot needs BatchTuner to choose a safe batch size first"
        return status.current_queue_item.message or f"queue item {status.current_queue_item.status}"
    if status.next_queue_item is not None:
        if status.next_queue_item.status == "queued":
            return "training is not running; the next command is queued and ready"
        if status.next_queue_item.status == "needs_evidence":
            return status.next_queue_item.message or "waiting for required evidence"
        if status.next_queue_item.status == "failed":
            return status.next_queue_item.message or "last queued command failed"
        if status.next_queue_item.status == "skipped":
            return status.next_queue_item.message or "last queued command was skipped by a guard"
    if status.queue_counts:
        return "queue " + _format_active_counts(status.queue_counts)
    if status.pending:
        return f"waiting for stage {status.pending[0]}"
    return "no active task"


def _human_trust(status: LoopRunStatus) -> str:
    item = status.current_queue_item or status.next_queue_item
    profile = item.profile if item is not None else ""
    if profile == "debug":
        return "none; debug only verifies the pipeline and is not effect evidence"
    if profile == "pilot":
        return "none; pilot is preliminary evidence and not a final conclusion"
    if status.evidence.verified_metric_records <= 0:
        return "none; verified metric evidence is required"
    metrics = _format_metrics(status.evidence.key_metrics)
    if metrics == "none":
        return f"{status.evidence.verified_metric_records} verified metric records exist; report summary pending"
    return f"verified evidence exists: {metrics}"


def _human_next_step(status: LoopRunStatus) -> str:
    item = status.current_queue_item
    if item is not None and item.status == "running":
        if (
            status.training_heartbeat is not None
            and status.training_heartbeat.process_status == "not_found"
            and not _heartbeat_is_fresh(status.training_heartbeat)
        ):
            profile = item.profile or "current"
            return f"rerun the same optimize {profile} command; the stale queue will be requeued automatically"
        if item.command_type == "train":
            return "wait for training to finish; evidence import runs after completion"
        return "wait for the current queue item to finish"
    if (
        item is not None
        and item.status == "blocked_by_resource"
        and "missing_batch_tuning_result" in item.resource_blockers
    ):
        return status.next_command or "rerun optimize with the active pilot profile; BatchTuner will run first"
    if status.blocked_reason:
        return status.next_command or "resolve the blocked reason first"
    if status.next_command:
        return status.next_command
    return "no next step; this run may already be complete"


def _batch_tuning_label(process_detail: str) -> str:
    """Return a concise batch tuning label from process details."""
    match = re.search(r"batch_tuning=b(?P<batch>\d+)", process_detail)
    return f"b{match.group('batch')}" if match else ""


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
    details.append(f"command={_clean_terminal_line(item.command, limit=180)}")
    return " ".join(details)


def _format_queue_item_lines(item: QueueItemStatus) -> list[str]:
    """Render a queue item as readable verbose lines."""
    lines = [
        f"  Status:    {item.status}",
        f"  Node:      {item.node_id}",
        f"  Candidate: {item.candidate_id}",
        f"  Type:      {item.command_type}",
    ]
    if item.profile:
        lines.append(f"  Profile:   {item.profile}")
    if item.message:
        lines.append(f"  Message:   {_clean_terminal_line(item.message, limit=140)}")
    if item.requires_evidence:
        lines.append(f"  Evidence:  {', '.join(item.requires_evidence)}")
    if item.resource_blockers:
        lines.append(f"  Blockers:  {', '.join(item.resource_blockers)}")
    lines.append(f"  Command:   {_clean_terminal_line(item.command, limit=140)}")
    return lines


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


def _latest_progress(lines: list[str]) -> dict[str, str | int | float | None]:
    """Return the latest Ultralytics train/val progress from log lines."""
    empty: dict[str, str | int | float | None] = {
        "phase": "",
        "current": None,
        "total": None,
        "percent": None,
    }
    pattern = re.compile(
        r"(?P<prefix>.*?):\s*(?P<percent>\d+(?:\.\d+)?)%\S*\s+.*?"
        r"(?<!\d)(?P<current>\d+)\s*/\s*(?P<total>\d+)(?!\d)"
    )
    for line in reversed(lines):
        match = pattern.search(line)
        if not match:
            continue
        prefix = match.group("prefix").strip()
        phase = "validation" if "Class" in prefix or "mAP" in prefix else "training"
        return {
            "phase": phase,
            "current": int(match.group("current")),
            "total": int(match.group("total")),
            "percent": float(match.group("percent")),
        }
    return empty


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


def _latest_sample_value(records: list[dict[str, Any]], key: str) -> float | None:
    """Return the newest numeric value from runtime sample records."""
    for record in reversed(records):
        sample = record.get("sample")
        if isinstance(sample, dict):
            value = _float_or_none(sample.get(key))
            if value is not None:
                return value
    return None


def _latest_record_age_seconds(records: list[dict[str, Any]]) -> float | None:
    """Return age in seconds for the newest runtime record."""
    for record in reversed(records):
        timestamp = _parse_datetime(str(record.get("created_at") or ""))
        if timestamp is not None:
            return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())
    return None


def _file_age_seconds(path: Path) -> float | None:
    """Return file modification age in seconds."""
    if not path.is_file():
        return None
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - modified).total_seconds())


def _parse_datetime(value: str) -> datetime | None:
    """Parse ISO datetimes emitted by runtime JSONL."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
