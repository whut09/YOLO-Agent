"""Batch-size tuning for Ultralytics training commands."""

from __future__ import annotations

import json
import os
import hashlib
import platform
import queue
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.adapters.ultralytics.runtime_profiler import RuntimeProfiler, RuntimeSampler
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.experiment_graph import ExperimentNode, MetricValue


BatchTrialStatus = Literal["completed", "oom", "failed", "timeout"]


class BatchTuningConfig(BaseModel):
    """Controls short batch-size probes before real training."""

    enabled: bool | None = None
    candidate_batches: list[int] = Field(default_factory=lambda: [32, 48, 64, 96])
    auto_expand_candidates: bool = True
    max_candidate_batch: int | None = Field(default=256, ge=1)
    candidate_order: Literal["largest_first", "smallest_first"] = "largest_first"
    trial_epochs: int = Field(default=1, ge=1)
    trial_fraction: float | None = Field(default=0.01, gt=0.0, le=1.0)
    timeout_seconds: int | None = Field(default=900, ge=1)
    no_output_timeout_seconds: float | None = Field(default=45.0, gt=0.0)
    trial_workers: int = Field(default=2, ge=0)
    sample_interval_seconds: float = Field(default=2.0, gt=0.0)
    selection_metric: Literal["avg_it_per_sec"] = "avg_it_per_sec"
    machine_cache_enabled: bool = True
    machine_cache_path: Path | None = None
    machine_cache_max_age_days: int = Field(default=30, ge=1)


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

        candidates = candidate_batches_for_command(command, self.config)
        cache_key = batch_cache_key(command, self.config, candidates)
        capacity_cache_key = batch_capacity_cache_key(command, self.config, candidates)
        cached = load_cached_batch_tuning(cache_key, self.config)
        if cached is None:
            cached = load_cached_batch_tuning(capacity_cache_key, self.config)
        if cached is None:
            cached = load_cached_batch_tuning_by_capacity(command, self.config)
            if cached is not None:
                save_cached_batch_tuning(capacity_cache_key, cached, self.config)
        if cached is not None:
            self._log(
                run_id,
                f"batch tuning cache hit: batch={cached.selected_batch} key={cache_key[:12]}",
            )
            self._persist(run_id, node, cached)
            return cached
        self._log(
            run_id,
            "batch tuning candidates "
            f"({self.config.candidate_order}; not formal training yet): "
            f"{','.join(str(value) for value in candidates)}",
        )
        trials: list[BatchTrialResult] = []
        for batch_size in candidates:
            self._log(run_id, f"batch tuning trial started: batch={batch_size} (not formal training yet)")
            trial_command = build_batch_trial_command(command, batch_size, self.config)
            trial = self._run_trial(run_id, batch_size, trial_command)
            trials.append(trial)
            self._log(run_id, f"batch tuning trial {batch_size}: {trial.status} {trial.message}")
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
            self._log(run_id, result.reason)
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
        save_cached_batch_tuning(cache_key, result, self.config)
        if capacity_cache_key != cache_key:
            save_cached_batch_tuning(capacity_cache_key, result, self.config)
        self._log(run_id, result.reason)
        return result

    def _run_trial(self, run_id: str, batch_size: int, command: CommandSpec) -> BatchTrialResult:
        started = time.monotonic()
        sampler = RuntimeSampler(interval_seconds=self.config.sample_interval_seconds)
        stdout_parts: list[str] = []
        stderr = ""
        return_code: int | None = None
        status: BatchTrialStatus = "failed"
        message = "Batch trial failed."
        line_queue: queue.Queue[str] = queue.Queue()
        process: subprocess.Popen[str] | None = None
        reader: threading.Thread | None = None
        last_output_at = time.monotonic()
        try:
            with sampler:
                process = subprocess.Popen(
                    command.as_subprocess_args(),
                    cwd=command.cwd,
                    env={**os.environ, **command.env} if command.env else None,
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                reader = threading.Thread(
                    target=_read_process_stdout,
                    args=(process, line_queue),
                    name=f"batch-tune-b{batch_size}-reader",
                    daemon=True,
                )
                reader.start()
                while True:
                    elapsed = time.monotonic() - started
                    if command.timeout_seconds is not None and elapsed > command.timeout_seconds:
                        status = "timeout"
                        message = f"Batch trial timed out after {command.timeout_seconds} seconds."
                        _terminate_process_tree(process)
                        break
                    if (
                        self.config.no_output_timeout_seconds is not None
                        and not stdout_parts
                        and time.monotonic() - last_output_at > self.config.no_output_timeout_seconds
                    ):
                        status = "timeout"
                        message = (
                            "Batch trial produced no Ultralytics output for "
                            f"{self.config.no_output_timeout_seconds:g} seconds; killed the trial."
                        )
                        self._log(run_id, f"batch tuning trial {batch_size}: no output watchdog fired")
                        _terminate_process_tree(process)
                        break
                    try:
                        line = line_queue.get(timeout=0.2)
                    except queue.Empty:
                        if process.poll() is not None and (reader is None or not reader.is_alive()) and line_queue.empty():
                            break
                        continue
                    stdout_parts.append(line)
                    last_output_at = time.monotonic()
                    clean = line.strip()
                    if clean:
                        self._log(run_id, f"batch tuning b{batch_size}: {clean[:300]}")
                while not line_queue.empty():
                    line = line_queue.get()
                    stdout_parts.append(line)
                if process.poll() is None:
                    _terminate_process_tree(process)
                return_code = process.wait(timeout=5)
            stdout = "".join(stdout_parts)
            if status != "timeout":
                if _is_oom(stdout, stderr):
                    status = "oom"
                    message = "Batch trial failed with CUDA out-of-memory."
                elif return_code == 0:
                    status = "completed"
                    message = "Batch trial completed."
                else:
                    status = "failed"
                    message = f"Batch trial failed with return code {return_code}."
        except subprocess.TimeoutExpired:
            if process is not None:
                _terminate_process_tree(process)
            stdout = "".join(stdout_parts)
            status = "timeout"
            message = f"Batch trial timed out after {command.timeout_seconds} seconds."
        except OSError as exc:
            stdout = "".join(stdout_parts)
            status = "failed"
            message = f"Batch trial could not start: {exc}"

        run_dir = _run_dir_from_command(command)
        profile = RuntimeProfiler().profile(
            run_dir or Path("."),
            stdout="\n".join(part for part in ("".join(stdout_parts), stderr) if part),
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

    def _log(self, run_id: str, message: str) -> None:
        """Append a user-visible batch tuning progress event when a store is available."""
        if self.evidence_store is None:
            return
        EventLog(self.evidence_store.create_run(run_id) / "events.jsonl").append(
            run_id=run_id,
            event_type="executor_log",
            message=message,
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


def candidate_batches_for_command(command: CommandSpec, config: BatchTuningConfig | None = None) -> list[int]:
    """Return batch candidates expanded by visible GPU memory."""
    tuning = config or BatchTuningConfig()
    candidates = list(tuning.candidate_batches)
    if tuning.auto_expand_candidates:
        candidates.extend(vram_batch_candidates(_visible_gpu_total_mb(), unit="mb"))
    if tuning.max_candidate_batch is not None:
        candidates = [candidate for candidate in candidates if candidate <= tuning.max_candidate_batch]
    unique = sorted({candidate for candidate in candidates if candidate > 0})
    if tuning.candidate_order == "largest_first":
        return list(reversed(unique))
    return unique


def batch_cache_key(
    command: CommandSpec,
    config: BatchTuningConfig | None = None,
    candidates: list[int] | None = None,
) -> str:
    """Return a stable per-machine cache key for equivalent batch tuning sweeps."""
    tuning = config or BatchTuningConfig()
    payload = {
        "schema": "batch_tuning_cache.v1",
        "machine": {
            "node": platform.node(),
            "system": platform.system(),
            "processor": platform.processor(),
            "gpu": _visible_gpu_identity(),
        },
        "command": {
            "model": _arg_value(command, "model"),
            "data": _arg_value(command, "data"),
            "imgsz": _arg_value(command, "imgsz"),
            "device": _arg_value(command, "device"),
            "amp": _arg_value(command, "amp"),
            "workers": _arg_value(command, "workers"),
            "cache": _arg_value(command, "cache"),
        },
        "tuning": {
            "candidate_batches": candidates or candidate_batches_for_command(command, tuning),
            "trial_epochs": tuning.trial_epochs,
            "trial_fraction": tuning.trial_fraction,
            "trial_workers": tuning.trial_workers,
            "selection_metric": tuning.selection_metric,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def batch_capacity_cache_key(
    command: CommandSpec,
    config: BatchTuningConfig | None = None,
    candidates: list[int] | None = None,
) -> str:
    """Return a reusable batch-capacity key across safe candidate variants."""
    tuning = config or BatchTuningConfig()
    payload = {
        "schema": "batch_capacity_cache.v1",
        "machine": {
            "node": platform.node(),
            "system": platform.system(),
            "processor": platform.processor(),
            "gpu": _visible_gpu_identity(),
        },
        "command": {
            "model": _arg_value(command, "model"),
            "data": _arg_value(command, "data"),
            "imgsz": _arg_value(command, "imgsz"),
            "device": _arg_value(command, "device"),
            "amp": _arg_value(command, "amp"),
        },
        "tuning": {
            "trial_epochs": tuning.trial_epochs,
            "trial_fraction": tuning.trial_fraction,
            "selection_metric": tuning.selection_metric,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_batch_tuning(cache_key: str, config: BatchTuningConfig | None = None) -> BatchTuningResult | None:
    """Load a machine-level batch tuning result when it is still fresh."""
    tuning = config or BatchTuningConfig()
    if not tuning.machine_cache_enabled:
        return None
    path = _batch_cache_path(tuning)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get(cache_key)
    if not isinstance(raw, dict):
        return None
    try:
        result = BatchTuningResult.model_validate(raw)
    except ValueError:
        return None
    age_seconds = time.time() - result.created_at.timestamp()
    if age_seconds > tuning.machine_cache_max_age_days * 86400:
        return None
    if result.selected_batch is None:
        return None
    return result.model_copy(
        update={
            "trials": [],
            "applied": True,
            "reason": f"Reused machine batch tuning cache: batch {result.selected_batch}.",
        }
    )


def load_cached_batch_tuning_by_capacity(
    command: CommandSpec,
    config: BatchTuningConfig | None = None,
) -> BatchTuningResult | None:
    """Reuse older exact-cache entries that match the same machine capacity.

    Early versions keyed the cache by candidate-specific details such as run
    name, workers, and cache mode. For auto-loop candidates, those details vary
    while the GPU memory capacity does not, so this scanner migrates compatible
    historical tuning results into the newer capacity cache.
    """
    tuning = config or BatchTuningConfig()
    if not tuning.machine_cache_enabled:
        return None
    path = _batch_cache_path(tuning)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    target = _capacity_signature(command)
    compatible: list[BatchTuningResult] = []
    for raw in data.values():
        if not isinstance(raw, dict):
            continue
        try:
            result = BatchTuningResult.model_validate(raw)
        except ValueError:
            continue
        if result.selected_batch is None or not result.applied:
            continue
        age_seconds = time.time() - result.created_at.timestamp()
        if age_seconds > tuning.machine_cache_max_age_days * 86400:
            continue
        if not _capacity_signatures_compatible(target, _result_capacity_signature(result)):
            continue
        compatible.append(result)
    if not compatible:
        return None
    selected = max(compatible, key=lambda item: item.created_at.timestamp())
    return selected.model_copy(
        update={
            "trials": [],
            "applied": True,
            "reason": f"Reused compatible machine batch tuning cache: batch {selected.selected_batch}.",
        }
    )


def save_cached_batch_tuning(
    cache_key: str,
    result: BatchTuningResult,
    config: BatchTuningConfig | None = None,
) -> Path | None:
    """Persist a successful batch tuning result for future runs on this machine."""
    tuning = config or BatchTuningConfig()
    if not tuning.machine_cache_enabled or result.selected_batch is None or not result.applied:
        return None
    path = _batch_cache_path(tuning)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data[cache_key] = result.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


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
        "cache": False,
        "workers": tuning.trial_workers,
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


def _capacity_signature(command: CommandSpec) -> dict[str, str | None]:
    return {
        "model": _arg_value(command, "model"),
        "data": _arg_value(command, "data"),
        "imgsz": _arg_value(command, "imgsz"),
        "device": _arg_value(command, "device"),
        "amp": _arg_value(command, "amp"),
    }


def _result_capacity_signature(result: BatchTuningResult) -> dict[str, str | None] | None:
    for trial in result.trials:
        if not trial.command:
            continue
        return {
            "model": _arg_value_from_text(trial.command, "model"),
            "data": _arg_value_from_text(trial.command, "data"),
            "imgsz": _arg_value_from_text(trial.command, "imgsz"),
            "device": _arg_value_from_text(trial.command, "device"),
            "amp": _arg_value_from_text(trial.command, "amp"),
        }
    return None


def _capacity_signatures_compatible(
    target: dict[str, str | None],
    cached: dict[str, str | None] | None,
) -> bool:
    if cached is None:
        return False
    for key in ("model", "data", "imgsz"):
        if target.get(key) != cached.get(key):
            return False
    for key in ("device", "amp"):
        if target.get(key) is not None and cached.get(key) is not None and target.get(key) != cached.get(key):
            return False
    return True


def _arg_value_from_text(command: str, key: str) -> str | None:
    prefix = f"{key}="
    for token in command.split():
        if token.startswith(prefix):
            return token.split("=", 1)[1]
    return None


def _batch_cache_path(config: BatchTuningConfig) -> Path:
    if config.machine_cache_path is not None:
        return config.machine_cache_path
    return Path.home() / ".yolo_agent" / "batch_tuning_cache.json"


def _visible_gpu_identity() -> list[dict[str, str]]:
    """Return visible GPU identity fields for cache separation."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    identity: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            identity.append({"name": parts[0], "memory_total_mb": parts[1], "driver": parts[2]})
    return identity


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


def _read_process_stdout(process: subprocess.Popen[str], line_queue: queue.Queue[str]) -> None:
    """Read process output into a queue for watchdog-friendly streaming."""
    stream = process.stdout
    if stream is None:
        return
    while True:
        line = stream.readline()
        if line == "":
            break
        line_queue.put(line)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a batch trial process tree best-effort."""
    if process.poll() is not None:
        return
    pid = getattr(process, "pid", None)
    if os.name == "nt" and pid is not None:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            process.kill()
    else:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _visible_gpu_total_mb() -> float | None:
    """Return the largest visible GPU memory total in MB."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    values: list[float] = []
    for line in completed.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def vram_batch_candidates(total_vram: float | None, unit: Literal["mb", "gb"] = "mb") -> list[int]:
    """Return high-batch probes for the largest visible GPU memory total."""
    if total_vram is None:
        return []
    total_mb = total_vram * 1024 if unit == "gb" else total_vram
    if total_mb >= 32000:
        return [64, 96, 128, 160, 192, 224, 256]
    if total_mb >= 22000:
        return [64, 96, 128, 160, 192, 224, 256]
    if total_mb >= 16000:
        return [48, 64, 96, 128]
    if total_mb >= 10000:
        return [32, 48, 64]
    return [16, 32]
