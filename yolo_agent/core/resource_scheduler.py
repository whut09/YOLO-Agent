"""GPU-aware resource scheduler for execution queue items."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.core.command_spec import CommandSpec, ResourceRequirements
from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.experiment_graph import Evidence


ResourceDecisionStatus = Literal["runnable", "paused", "blocked_by_resource", "needs_resume"]


class GPUResource(BaseModel):
    """Current GPU resource snapshot for one device."""

    gpu_id: int
    name: str = ""
    util_percent: float | None = Field(default=None, ge=0.0)
    memory_used_mb: float | None = Field(default=None, ge=0.0)
    memory_total_mb: float | None = Field(default=None, ge=0.0)

    @property
    def free_vram_mb(self) -> float | None:
        """Return available VRAM when memory data is present."""
        if self.memory_total_mb is None or self.memory_used_mb is None:
            return None
        return max(0.0, self.memory_total_mb - self.memory_used_mb)


class ResourceSnapshot(BaseModel):
    """Machine resource snapshot used by the scheduler."""

    gpus: list[GPUResource] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "manual"


class ResourceSchedulerConfig(BaseModel):
    """Scheduling policy for queue execution."""

    gpu_idle_util_threshold: float = Field(default=80.0, ge=0.0, le=100.0)
    default_min_free_vram_mb: int = Field(default=4096, ge=0)
    defer_high_risk: bool = True
    enforce_full_run_windows: bool = True
    require_batch_tuning_result: bool = True


class ResourceDecision(BaseModel):
    """Scheduler decision for one queue item."""

    status: ResourceDecisionStatus
    reasons: list[str] = Field(default_factory=list)
    selected_gpu_id: int | None = None
    message: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResourceScheduler:
    """Decide whether a queued item may run on current local resources."""

    def __init__(
        self,
        config: ResourceSchedulerConfig | None = None,
        snapshot: ResourceSnapshot | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config = config or ResourceSchedulerConfig()
        self.snapshot = snapshot
        self.now = now

    def evaluate(
        self,
        command: CommandSpec,
        evidence: Evidence | None = None,
        attempts: int = 0,
    ) -> ResourceDecision:
        """Return a scheduling decision for one command."""
        requirements = command.resource_requirements
        pause_reasons = _pause_reasons(command, requirements, self.config, self.now or datetime.now())
        if pause_reasons:
            return ResourceDecision(status="paused", reasons=pause_reasons, message="Execution paused by resource policy.")

        resume_reasons = _resume_reasons(command, requirements, attempts)
        if resume_reasons:
            return ResourceDecision(status="needs_resume", reasons=resume_reasons, message="Execution needs a resume checkpoint.")

        evidence_reasons = _evidence_resource_reasons(command, requirements, evidence, self.config)
        if evidence_reasons:
            return ResourceDecision(
                status="blocked_by_resource",
                reasons=evidence_reasons,
                message="Execution blocked by missing resource preparation evidence.",
            )

        if requirements.requires_gpu:
            gpu_decision = self._gpu_decision(requirements)
            if gpu_decision is not None:
                return gpu_decision

        return ResourceDecision(status="runnable", message="Resource requirements satisfied.")

    def _gpu_decision(self, requirements: ResourceRequirements) -> ResourceDecision | None:
        snapshot = self.snapshot or current_resource_snapshot()
        if not snapshot.gpus:
            return ResourceDecision(
                status="blocked_by_resource",
                reasons=["gpu_unavailable"],
                message="Command requires GPU but no GPU is visible.",
            )
        candidates = [
            gpu for gpu in snapshot.gpus
            if requirements.preferred_gpu_id is None or gpu.gpu_id == requirements.preferred_gpu_id
        ]
        if not candidates:
            return ResourceDecision(
                status="blocked_by_resource",
                reasons=[f"preferred_gpu_unavailable:{requirements.preferred_gpu_id}"],
                message="Preferred GPU is not visible.",
            )
        min_free = requirements.min_free_vram_mb or self.config.default_min_free_vram_mb
        blockers: list[str] = []
        for gpu in candidates:
            if gpu.util_percent is not None and gpu.util_percent > self.config.gpu_idle_util_threshold:
                blockers.append(f"gpu_busy:{gpu.gpu_id}:{gpu.util_percent:.1f}%")
                continue
            free = gpu.free_vram_mb
            if free is not None and free < min_free:
                blockers.append(f"insufficient_vram:{gpu.gpu_id}:{free:.0f}<{min_free}")
                continue
            return ResourceDecision(
                status="runnable",
                selected_gpu_id=gpu.gpu_id,
                message=f"GPU {gpu.gpu_id} satisfies resource requirements.",
            )
        return ResourceDecision(
            status="blocked_by_resource",
            reasons=blockers or ["no_gpu_satisfies_requirements"],
            message="No visible GPU satisfies resource requirements.",
        )


def current_resource_snapshot() -> ResourceSnapshot:
    """Collect a best-effort GPU resource snapshot via nvidia-smi."""
    query = "index,name,utilization.gpu,memory.used,memory.total"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return ResourceSnapshot(source="nvidia_smi_unavailable")
    if completed.returncode != 0:
        return ResourceSnapshot(source="nvidia_smi_error")
    gpus: list[GPUResource] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        gpu_id = _optional_int(parts[0])
        if gpu_id is None:
            continue
        gpus.append(
            GPUResource(
                gpu_id=gpu_id,
                name=parts[1],
                util_percent=_optional_float(parts[2]),
                memory_used_mb=_optional_float(parts[3]),
                memory_total_mb=_optional_float(parts[4]),
            )
        )
    return ResourceSnapshot(gpus=gpus, source="nvidia_smi")


def _pause_reasons(
    command: CommandSpec,
    requirements: ResourceRequirements,
    config: ResourceSchedulerConfig,
    now: datetime,
) -> list[str]:
    reasons: list[str] = []
    if requirements.high_risk and config.defer_high_risk and command.metadata.get("high_risk_approved") is not True:
        reasons.append("high_risk_candidate_deferred")
    if (
        requirements.full_run
        and config.enforce_full_run_windows
        and requirements.allowed_start_hours
        and now.hour not in set(requirements.allowed_start_hours)
    ):
        reasons.append(f"outside_full_run_budget_window:{now.hour}")
    return reasons


def _resume_reasons(command: CommandSpec, requirements: ResourceRequirements, attempts: int) -> list[str]:
    if requirements.requires_resume:
        return [] if _can_resume(command) else ["missing_resume_checkpoint"]
    if command.command_type == "train" and attempts > 0 and requirements.allow_resume and not _can_resume(command):
        return ["missing_resume_checkpoint_after_attempt"]
    return []


def _evidence_resource_reasons(
    command: CommandSpec,
    requirements: ResourceRequirements,
    evidence: Evidence | None,
    config: ResourceSchedulerConfig,
) -> list[str]:
    if not requirements.requires_batch_tuning or not config.require_batch_tuning_result:
        return []
    if command.metadata.get("training_executor") == "ultralytics":
        return []
    if command.metadata.get("batch_tuned") is True:
        return []
    if evidence is None:
        return ["missing_batch_tuning_result"]
    candidate_id = str(command.metadata.get("candidate_id", ""))
    node_id = str(command.metadata.get("node_id", ""))
    record = EvidenceIndex(evidence.metric_records).select_one(
        candidate_id=candidate_id or None,
        node_id=node_id or None,
        metric_name="batch_tuning_selected_batch",
        verified=True,
    )
    if record is None and candidate_id:
        record = EvidenceIndex(evidence.metric_records).select_one(
            candidate_id=candidate_id,
            metric_name="batch_tuning_selected_batch",
            verified=True,
        )
    return [] if record is not None and record.value is not None else ["missing_batch_tuning_result"]


def _can_resume(command: CommandSpec) -> bool:
    if any(str(arg).startswith("resume=") and str(arg).split("=", 1)[1] not in {"False", "false", "0", ""} for arg in command.argv):
        return True
    for key in ("last_pt", "resume_checkpoint"):
        path = command.expected_artifacts.get(key)
        if path is not None and Path(path).is_file():
            return True
    return False


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _optional_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None
