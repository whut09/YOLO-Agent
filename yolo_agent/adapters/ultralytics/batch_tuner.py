"""Batch-size tuning for Ultralytics training commands."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.adapters.ultralytics.runtime_profiler import RuntimeProfiler, RuntimeSampler
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode, MetricValue


BatchTrialStatus = Literal["completed", "oom", "failed", "timeout"]


class BatchTuningConfig(BaseModel):
    """Controls short batch-size probes before real training."""

    enabled: bool | None = None
    candidate_batches: list[int] = Field(default_factory=lambda: [32, 48, 64, 96])
    trial_epochs: int = Field(default=1, ge=1)
    trial_fraction: float | None = Field(default=0.01, gt=0.0, le=1.0)
    timeout_seconds: int | None = Field(default=900, ge=1)
    sample_interval_seconds: float = Field(default=2.0, gt=0.0)
    selection_metric: Literal["avg_it_per_sec"] = "avg_it_per_sec"


class BatchTrialResult(BaseModel):
    """One attempted batch-size probe."""

    batch_size: int
    status: BatchTrialStatus
    return_code: int | None = None
    duration_seconds: float | None = None
    avg_it_per_sec: float | None = None
    max_it_per_sec: float | None = None
    avg_gpu_util_percent: float | None = None
    max_gpu_memory_used_mb: float | None = None
    command: str = ""
    run_dir: Path | None = None
    message: str = ""

    @field_serializer("run_dir")
    def serialize_run_dir(self, value: Path | None) -> str | None:
        """Serialize trial run directory portably."""
        return value.as_posix() if value is not None else None


class BatchTuningResult(BaseModel):
    """Result of a batch tuning sweep."""

    selected_batch: int | None = None
    selected_metric: float | None = None
    trials: list[BatchTrialResult] = Field(default_factory=list)
    applied: bool = False
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_metrics(self) -> dict[str, MetricValue]:
        """Return summary and per-batch facts as metric evidence."""
        metrics: dict[str, MetricValue] = {
            "batch_tuning_applied": self.applied,
            "batch_tuning_selected_batch": self.selected_batch,
            "batch_tuning_best_it_per_sec": self.selected_metric,
            "batch_tuning_trial_count": len(self.trials),
            "batch_tuning_oom_trials": sum(1 for trial in self.trials if trial.status == "oom"),
        }
        for trial in self.trials:
            prefix = f"batch_tuning_b{trial.batch_size}"
            metrics[f"{prefix}_oom"] = trial.status == "oom"
            metrics[f"{prefix}_status"] = trial.status
            if trial.avg_it_per_sec is not None:
                metrics[f"{prefix}_avg_it_per_sec"] = trial.avg_it_per_sec
            if trial.avg_gpu_util_percent is not None:
                metrics[f"{prefix}_avg_gpu_util_percent"] = trial.avg_gpu_util_percent
            if trial.max_gpu_memory_used_mb is not None:
                metrics[f"{prefix}_max_gpu_memory_used_mb"] = trial.max_gpu_memory_used_mb
        return metrics


class BatchTuner:
    """Run short batch-size probes and select the highest-throughput safe batch."""

    def __init__(
        self,
        config: BatchTuningConfig | None = None,
        evidence_store: EvidenceStore | None = None,
    ) -> None:
        self.config = config or BatchTuningConfig()
        self.evidence_store = evidence_store

    def tune(self, run_id: str, node: ExperimentNode, command: CommandSpec) -> BatchTuningResult:
        """Try candidate batches and persist tuning evidence."""
        if not should_tune_batch(command, self.config):
            return BatchTuningResult(applied=False, reason="Batch tuning disabled for this command.")

        trials: list[BatchTrialResult] = []
        for batch_size in self.config.candidate_batches:
            trial_command = build_batch_trial_command(command, batch_size, self.config)
            trials.append(self._run_trial(batch_size, trial_command))
        safe_trials = [
            trial
            for trial in trials
            if trial.status == "completed" and trial.avg_it_per_sec is not None
        ]
        if not safe_trials:
            result = BatchTuningResult(
                selected_batch=None,
                trials=trials,
                applied=False,
                reason="No candidate batch completed with measurable throughput; keep original batch.",
            )
            self._persist(run_id, node, result)
            return result

        selected = max(safe_trials, key=lambda trial: trial.avg_it_per_sec or 0.0)
        result = BatchTuningResult(
            selected_batch=selected.batch_size,
            selected_metric=selected.avg_it_per_sec,
            trials=trials,
            applied=True,
            reason=f"Selected batch {selected.batch_size} by highest avg_it_per_sec.",
        )
        self._persist(run_id, node, result)
        return result

    def _run_trial(self, batch_size: int, command: CommandSpec) -> BatchTrialResult:
        started = time.monotonic()
        sampler = RuntimeSampler(interval_seconds=self.config.sample_interval_seconds)
        stdout = ""
        stderr = ""
        return_code: int | None = None
        status: BatchTrialStatus
        message: str
        try:
            with sampler:
                completed = subprocess.run(
                    command.as_subprocess_args(),
                    cwd=command.cwd,
                    env={**os.environ, **command.env} if command.env else None,
                    timeout=command.timeout_seconds,
                    shell=False,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            stdout = completed.stdout
            stderr = completed.stderr
            return_code = completed.returncode
            if _is_oom(stdout, stderr):
                status = "oom"
                message = "Batch trial failed with CUDA out-of-memory."
            elif completed.returncode == 0:
                status = "completed"
                message = "Batch trial completed."
            else:
                status = "failed"
                message = "Batch trial failed."
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            status = "timeout"
            message = f"Batch trial timed out after {command.timeout_seconds} seconds."

        run_dir = _run_dir_from_command(command)
        profile = RuntimeProfiler().profile(
            run_dir or Path("."),
            stdout="\n".join(part for part in (stdout, stderr) if part),
            samples=sampler.samples,
            sample_gpu=False,
        )
        return BatchTrialResult(
            batch_size=batch_size,
            status=status,
            return_code=return_code,
            duration_seconds=round(time.monotonic() - started, 6),
            avg_it_per_sec=profile.avg_it_per_sec,
            max_it_per_sec=profile.max_it_per_sec,
            avg_gpu_util_percent=profile.avg_gpu_util_percent,
            max_gpu_memory_used_mb=profile.max_gpu_memory_used_mb,
            command=command.display(),
            run_dir=run_dir,
            message=message,
        )

    def _persist(self, run_id: str, node: ExperimentNode, result: BatchTuningResult) -> None:
        if self.evidence_store is None:
            return
        artifact_path = (
            self.evidence_store.create_run(run_id)
            / "artifacts"
            / f"{node.node_id}_batch_tuning_result.json"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("w", encoding="utf-8") as file:
            json.dump(result.model_dump(mode="json"), file, indent=2, sort_keys=True)
        self.evidence_store.log_artifact_manifest(
            run_id=run_id,
            name=f"{node.node_id}_batch_tuning_result",
            artifact_path=artifact_path,
            producer_stage="batch_tuner",
        )
        self.evidence_store.log_candidate_metrics(
            run_id=run_id,
            candidate_id=node.candidate_config.candidate_id,
            node_id=node.node_id,
            metrics=result.to_metrics(),
            dataset_version=node.data_version,
            split="runtime",
            source="batch_tuner",
            verified=True,
            validator="ultralytics_batch_tuner",
            source_artifact=artifact_path,
        )


def should_tune_batch(command: CommandSpec, config: BatchTuningConfig | None = None) -> bool:
    """Return whether batch tuning should run for this command."""
    tuning = config or BatchTuningConfig()
    if tuning.enabled is False:
        return False
    auto_policy = str(command.metadata.get("training_batch_policy") or "").strip().lower() == "auto"
    auto_arg = str(_arg_value(command, "batch") or "").strip().lower() in {"auto", "-1"}
    if tuning.enabled is True:
        return auto_policy or auto_arg
    return auto_policy or auto_arg


def apply_selected_batch(command: CommandSpec, batch_size: int) -> CommandSpec:
    """Return a copy of a train command with the selected batch applied."""
    updated = _upsert_args(command, {"batch": batch_size})
    metadata = {
        **command.metadata,
        "batch_tuned": True,
        "batch_tuning_selected_batch": batch_size,
    }
    return updated.model_copy(update={"metadata": metadata})


def build_batch_trial_command(
    command: CommandSpec,
    batch_size: int,
    config: BatchTuningConfig | None = None,
) -> CommandSpec:
    """Build a short trial command that changes only the batch policy."""
    tuning = config or BatchTuningConfig()
    base_name = str(_arg_value(command, "name") or "train")
    trial_name = f"{base_name}_batch_tune_b{batch_size}"
    updates: dict[str, str | int | float | bool] = {
        "name": trial_name,
        "batch": batch_size,
        "epochs": tuning.trial_epochs,
        "val": False,
        "plots": False,
        "save": False,
        "save_json": False,
        "exist_ok": True,
    }
    if tuning.trial_fraction is not None:
        updates["fraction"] = tuning.trial_fraction
    trial = _upsert_args(command, updates)
    run_dir = _run_dir_from_parts(_arg_value(trial, "project"), _arg_value(trial, "name"))
    expected_artifacts = (
        {
            "results_csv": run_dir / "results.csv",
            "args_yaml": run_dir / "args.yaml",
        }
        if run_dir is not None
        else {}
    )
    metadata = {
        **command.metadata,
        "batch_tuning_trial": True,
        "batch_tuning_trial_batch": batch_size,
        "batch_tuning_preserves_imgsz": True,
    }
    return trial.model_copy(
        update={
            "timeout_seconds": tuning.timeout_seconds or command.timeout_seconds,
            "expected_artifacts": expected_artifacts,
            "metadata": metadata,
        }
    )


def _upsert_args(command: CommandSpec, updates: dict[str, str | int | float | bool]) -> CommandSpec:
    argv = list(command.argv or [command.command, *command.args])
    seen: set[str] = set()
    updated_argv: list[str] = []
    for item in argv:
        key = item.split("=", 1)[0] if "=" in item else ""
        if key in updates:
            updated_argv.append(f"{key}={_format_arg(updates[key])}")
            seen.add(key)
        else:
            updated_argv.append(item)
    for key, value in updates.items():
        if key not in seen:
            updated_argv.append(f"{key}={_format_arg(value)}")
    return command.model_copy(
        update={
            "command": updated_argv[0],
            "args": updated_argv[1:],
            "argv": updated_argv,
        }
    )


def _arg_value(command: CommandSpec, key: str) -> str | None:
    for item in command.argv or [command.command, *command.args]:
        if item.startswith(f"{key}="):
            return item.split("=", 1)[1]
    return None


def _run_dir_from_command(command: CommandSpec) -> Path | None:
    return _run_dir_from_parts(_arg_value(command, "project"), _arg_value(command, "name"))


def _run_dir_from_parts(project: str | None, name: str | None) -> Path | None:
    if not project or not name:
        return None
    return Path(project) / name


def _format_arg(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _is_oom(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text
