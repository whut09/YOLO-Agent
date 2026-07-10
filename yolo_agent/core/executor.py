"""Executor abstractions for controlled experiment execution."""

from __future__ import annotations

import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sysconfig
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.experiment_graph import ExperimentNode, MetricEvidence, MetricValue


ExecutionStatus = Literal["planned", "dry_run", "completed", "failed", "skipped"]


class ExecutionResult(BaseModel):
    """Result of executing or planning one command."""

    run_id: str
    node_id: str
    candidate_id: str
    status: ExecutionStatus
    command: CommandSpec
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    artifacts: dict[str, Path] = Field(default_factory=dict)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    message: str = ""

    @field_serializer("artifacts")
    def serialize_artifacts(self, value: dict[str, Path]) -> dict[str, str]:
        """Serialize artifact paths portably."""
        return {key: path.as_posix() for key, path in value.items()}

    def log_to_evidence_store(self, store: EvidenceStore) -> Path:
        """Persist execution result as evidence config and metrics."""
        config_path = store.log_config(
            self.run_id,
            {
                "execution_result": self.model_dump(mode="json"),
            },
        )
        metrics = {
            "execution_duration_seconds": self.duration_seconds,
            "execution_return_code": self.return_code,
            **self.metrics,
        }
        store.log_metrics(self.run_id, metrics)
        for name, artifact in self.artifacts.items():
            if artifact.exists():
                store.log_artifact_manifest(self.run_id, name, artifact, producer_stage="executor")
        return config_path


class ExperimentExecutor(Protocol):
    """Protocol for experiment executors."""

    def execute(self, node: ExperimentNode, run_id: str, command: CommandSpec | None = None) -> ExecutionResult:
        """Execute or plan one experiment node."""


class DryRunExecutor:
    """Executor that records what would run without starting training."""

    def execute(self, node: ExperimentNode, run_id: str, command: CommandSpec | None = None) -> ExecutionResult:
        """Return a dry-run result without executing a command."""
        spec = command or CommandSpec.from_experiment_node(node)
        now = datetime.now(timezone.utc)
        return ExecutionResult(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            status="dry_run",
            command=spec,
            started_at=now,
            ended_at=now,
            duration_seconds=0.0,
            message="Dry run only; command was not executed.",
        )


class ShellExecutor:
    """Explicit shell/subprocess executor for controlled commands."""

    def execute(self, node: ExperimentNode, run_id: str, command: CommandSpec | None = None) -> ExecutionResult:
        """Run a command with subprocess and capture output."""
        spec = command or CommandSpec.from_experiment_node(node)
        started = datetime.now(timezone.utc)
        start_time = time.monotonic()
        try:
            completed = subprocess.run(
                spec.as_subprocess_args(),
                cwd=spec.cwd,
                env={**os.environ, **spec.env} if spec.env else None,
                timeout=spec.timeout_seconds,
                shell=spec.shell,
                check=False,
                capture_output=True,
                text=True,
            )
            status: ExecutionStatus = "completed" if completed.returncode == 0 else "failed"
            return_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            message = "Command completed." if status == "completed" else "Command failed."
        except subprocess.TimeoutExpired as exc:
            status = "failed"
            return_code = None
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            message = f"Command timed out after {spec.timeout_seconds} seconds."
        ended = datetime.now(timezone.utc)
        return ExecutionResult(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            status=status,
            command=spec,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            started_at=started,
            ended_at=ended,
            duration_seconds=time.monotonic() - start_time,
            message=message,
        )


class UltralyticsExecutor:
    """Experimental executor for verified Ultralytics training integration."""

    def __init__(self, adapter: Any | None = None, try_forward: bool = False) -> None:
        self.adapter = adapter
        self.try_forward = try_forward

    def execute(self, node: ExperimentNode, run_id: str, command: CommandSpec | None = None) -> ExecutionResult:
        """Generate artifacts and optionally run training via subprocess.

        @experimental
        Real training execution should be validated through SmokeRunner forward
        checks before use in production.
        """
        from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter

        adapter = self.adapter or UltralyticsAdapter()
        if not isinstance(adapter, UltralyticsAdapter):
            raise TypeError("adapter must be an UltralyticsAdapter instance")

        now = datetime.now(timezone.utc)
        if not adapter.is_available():
            return ExecutionResult(
                run_id=run_id,
                node_id=node.node_id,
                candidate_id=node.candidate_config.candidate_id,
                status="skipped",
                command=command or CommandSpec.from_experiment_node(node),
                started_at=now,
                ended_at=now,
                duration_seconds=0.0,
                message="Ultralytics is not installed or unverified; install 'ultralytics' and set try_forward=True to enable experimental training.",
            )

        try:
            yaml_result = adapter.generate_model_yaml(node.candidate_config)
            model_yaml_path = yaml_result.output_path
            exec_command = adapter.build_train_command(node, model_yaml_path=model_yaml_path)
            spec = command or CommandSpec(command=exec_command, shell=True)
        except Exception as exc:
            return ExecutionResult(
                run_id=run_id,
                node_id=node.node_id,
                candidate_id=node.candidate_config.candidate_id,
                status="failed",
                command=command or CommandSpec.from_experiment_node(node),
                started_at=now,
                ended_at=now,
                duration_seconds=0.0,
                message=f"UltralyticsExecutor failed to prepare experimental artifacts: {exc}",
            )

        if not self.try_forward:
            return ExecutionResult(
                run_id=run_id,
                node_id=node.node_id,
                candidate_id=node.candidate_config.candidate_id,
                status="dry_run",
                command=spec,
                started_at=now,
                ended_at=now,
                duration_seconds=0.0,
                message="Experimental executor prepared training artifacts but did not run them. Set try_forward=True to execute.",
                artifacts={"model_yaml": model_yaml_path},
            )

        start_time = time.monotonic()
        started = datetime.now(timezone.utc)
        try:
            completed = subprocess.run(
                spec.as_subprocess_args(),
                cwd=spec.cwd,
                env={**os.environ, **spec.env} if spec.env else None,
                timeout=spec.timeout_seconds,
                shell=spec.shell,
                check=False,
                capture_output=True,
                text=True,
            )
            status: ExecutionStatus = "completed" if completed.returncode == 0 else "failed"
            message = "Experimental training completed." if status == "completed" else "Experimental training failed."
            return_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            status = "failed"
            return_code = None
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            message = f"Experimental training timed out after {spec.timeout_seconds} seconds."
        ended = datetime.now(timezone.utc)
        return ExecutionResult(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            status=status,
            command=spec,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            started_at=started,
            ended_at=ended,
            duration_seconds=time.monotonic() - start_time,
            message=message,
            artifacts={"model_yaml": model_yaml_path},
        )


class UltralyticsTrainExecutor:
    """Stable typed executor for Ultralytics CLI training runs."""

    def __init__(
        self,
        evidence_store: EvidenceStore | None = None,
        training_config: Any | None = None,
        data_path: Path | str | None = None,
    ) -> None:
        self.evidence_store = evidence_store
        self.training_config = training_config
        self.data_path = Path(data_path) if data_path is not None else None

    def execute(self, node: ExperimentNode, run_id: str, command: CommandSpec | None = None) -> ExecutionResult:
        """Run a typed ``yolo detect train`` command and import produced evidence."""
        from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter
        from yolo_agent.adapters.ultralytics.training import (
            UltralyticsRunImporter,
            UltralyticsTrainingConfig,
            command_from_training_config,
            expected_ultralytics_artifacts,
        )
        from yolo_agent.adapters.ultralytics.batch_tuner import (
            BatchTuner,
            BatchTuningConfig,
            apply_selected_batch,
            should_tune_batch,
        )
        from yolo_agent.adapters.ultralytics.data_cache_policy import (
            DataCachePolicy,
            DataCachePolicyConfig,
        )
        from yolo_agent.adapters.ultralytics.fast_baseline_gate import (
            FastBaselineGate,
            FastBaselineGateConfig,
        )
        from yolo_agent.adapters.ultralytics.runtime_profiler import RuntimeSampler, parse_runtime_line_metrics
        from yolo_agent.adapters.ultralytics.stop_resume import StopResumeConfig, StopResumeGuard

        started = datetime.now(timezone.utc)
        adapter = UltralyticsAdapter()
        spec = command or CommandSpec.from_experiment_node(node)
        if spec.command_type != "train":
            config = self.training_config or UltralyticsTrainingConfig(
                model=node.candidate_config.base_model,
                data=self.data_path or Path("data.yaml"),
            )
            try:
                spec = command_from_training_config(node, config, run_id=run_id, data_path=self.data_path)
            except ValueError as exc:
                now = datetime.now(timezone.utc)
                failed_spec = _with_execution_identity(spec, node, run_id)
                return ExecutionResult(
                    run_id=run_id,
                    node_id=node.node_id,
                    candidate_id=node.candidate_config.candidate_id,
                    status="failed",
                    command=failed_spec,
                    started_at=started,
                    ended_at=now,
                    duration_seconds=0.0,
                    message=str(exc),
                )
        spec = _with_execution_identity(spec, node, run_id)
        if not adapter.is_available():
            now = datetime.now(timezone.utc)
            return ExecutionResult(
                run_id=run_id,
                node_id=node.node_id,
                candidate_id=node.candidate_config.candidate_id,
                status="skipped",
                command=spec,
                started_at=started,
                ended_at=now,
                duration_seconds=0.0,
                message="Ultralytics is not installed; install ultralytics before using UltralyticsTrainExecutor.",
            )
        resolved_command = _resolve_executable(spec.command)
        if resolved_command is None:
            now = datetime.now(timezone.utc)
            return ExecutionResult(
                run_id=run_id,
                node_id=node.node_id,
                candidate_id=node.candidate_config.candidate_id,
                status="failed",
                command=spec,
                started_at=started,
                ended_at=now,
                duration_seconds=0.0,
                message=f"Executable not found on PATH: {spec.command}",
            )
        if resolved_command != spec.command:
            argv = list(spec.argv or [spec.command, *spec.args])
            argv[0] = resolved_command
            spec = spec.model_copy(update={"command": resolved_command, "argv": argv})

        fast_gate_config = _fast_baseline_gate_config_from_training_config(
            self.training_config,
            FastBaselineGateConfig(),
        )
        profile_name = _training_profile_from_spec(spec)
        fast_gate = FastBaselineGate(fast_gate_config)
        fast_gate_applies = bool(profile_name and _fast_baseline_gate_applies(profile_name, node))
        if (
            profile_name
            and fast_gate_config.enabled
            and self.evidence_store is not None
            and fast_gate_applies
        ):
            gate_result = fast_gate.evaluate(
                profile_name,
                evidence=_load_or_create_evidence(self.evidence_store, run_id),
                candidate_id=_fast_gate_candidate_scope(profile_name, node),
            )
            fast_gate.persist_decision(self.evidence_store, run_id, node, gate_result)
            if not gate_result.ok:
                now = datetime.now(timezone.utc)
                return ExecutionResult(
                    run_id=run_id,
                    node_id=node.node_id,
                    candidate_id=node.candidate_config.candidate_id,
                    status="skipped",
                    command=spec,
                    started_at=started,
                    ended_at=now,
                    duration_seconds=0.0,
                    message=gate_result.message,
                )

        data_cache_config = _data_cache_policy_config_from_training_config(
            self.training_config,
            DataCachePolicyConfig(),
        )
        data_yaml = self.data_path or _path_arg_value(spec.argv, "data")
        if data_yaml is not None and data_cache_config.enabled:
            spec, _ = DataCachePolicy(
                config=data_cache_config,
                evidence_store=self.evidence_store,
            ).apply(run_id, node, spec, data_yaml)

        batch_tuning_config = _batch_tuning_config_from_training_config(
            self.training_config,
            BatchTuningConfig(),
        )
        if spec.resource_requirements.requires_batch_tuning and should_tune_batch(spec, batch_tuning_config):
            tuning_result = BatchTuner(
                config=batch_tuning_config,
                evidence_store=self.evidence_store,
            ).tune(run_id, node, spec)
            if tuning_result.selected_batch is not None:
                spec = apply_selected_batch(spec, tuning_result.selected_batch)

        start_time = time.monotonic()
        stream_paths = _stream_artifact_paths(self.evidence_store, run_id, node)
        runtime_lock = threading.Lock()
        stop_decision_queue: queue.Queue[Any] = queue.Queue()
        stop_resume_config = _stop_resume_config_from_training_config(
            self.training_config,
            StopResumeConfig(),
        )
        stop_guard = StopResumeGuard(stop_resume_config)

        def sample_callback(sample: Any) -> None:
            if stream_paths["runtime_jsonl"] is not None:
                _append_runtime_jsonl(
                    stream_paths["runtime_jsonl"],
                    {
                        "record_type": "gpu_sample",
                        "run_id": run_id,
                        "candidate_id": node.candidate_config.candidate_id,
                        "node_id": node.node_id,
                        "dataset_version": node.data_version,
                        "sample": sample.model_dump(mode="json"),
                    },
                    runtime_lock,
                )
            decision = stop_guard.observe_sample(sample)
            if decision is not None:
                _persist_stop_resume_decision(
                    decision=decision,
                    run_id=run_id,
                    node=node,
                    evidence_store=self.evidence_store,
                    event_log=_event_log_for_store(self.evidence_store, run_id),
                    runtime_jsonl_path=stream_paths["runtime_jsonl"],
                    runtime_jsonl_lock=runtime_lock,
                )
                if decision.should_stop:
                    stop_decision_queue.put(decision)

        sampler = RuntimeSampler(sample_callback=sample_callback)
        run_dir = _ultralytics_run_dir(spec)
        stream_result = _run_streaming_process(
            spec=spec,
            run_id=run_id,
            node=node,
            evidence_store=self.evidence_store,
            sampler=sampler,
            stdout_log_path=stream_paths["stdout_log"],
            runtime_jsonl_path=stream_paths["runtime_jsonl"],
            runtime_jsonl_lock=runtime_lock,
            line_metric_parser=parse_runtime_line_metrics,
            stop_resume_guard=stop_guard,
            results_csv_path=run_dir / "results.csv" if run_dir is not None else None,
            stop_decision_queue=stop_decision_queue,
        )
        status = stream_result["status"]
        stdout = stream_result["stdout"]
        stderr = stream_result["stderr"]
        return_code = stream_result["return_code"]
        message = stream_result["message"]
        timed_out = bool(stream_result.get("timed_out", False))
        ended = datetime.now(timezone.utc)
        actual_run_dir = _resolve_completed_ultralytics_run_dir(
            spec=spec,
            expected_run_dir=run_dir,
            stdout=stdout,
            stderr=stderr,
        )
        artifacts = _existing_artifacts(spec.expected_artifacts)
        if actual_run_dir is not None and actual_run_dir != run_dir:
            artifacts.update(_existing_artifacts(expected_ultralytics_artifacts(actual_run_dir)))
        for artifact_name, artifact_path in stream_paths.items():
            if artifact_path is not None and artifact_path.exists():
                artifacts[artifact_name] = artifact_path
                if self.evidence_store is not None:
                    self.evidence_store.log_artifact_manifest(
                        run_id=run_id,
                        name=f"{node.node_id}_{artifact_name}",
                        artifact_path=artifact_path,
                        producer_stage="ultralytics_stream",
                    )
        metrics: dict[str, MetricValue] = {
            "execution_timed_out": timed_out,
            "execution_timeout_seconds": spec.timeout_seconds,
        }
        if timed_out and self.evidence_store is not None:
            self.evidence_store.log_candidate_metrics(
                run_id=run_id,
                candidate_id=node.candidate_config.candidate_id,
                node_id=node.node_id,
                metrics=metrics,
                dataset_version=node.data_version,
                split="runtime",
                source="executor_timeout",
                verified=True,
                validator="ultralytics_train_executor",
            )
        if status == "completed" and self.evidence_store is not None and actual_run_dir is not None:
            imported_metrics = UltralyticsRunImporter(self.evidence_store).import_run(
                run_id,
                node,
                actual_run_dir,
                log_path=stream_paths["stdout_log"],
                stdout="\n".join(part for part in (stdout, stderr) if part),
                runtime_samples=sampler.samples,
                data_path=self.data_path or data_yaml,
            )
            metrics.update(imported_metrics)
            if profile_name and fast_gate_applies:
                stage_metrics = fast_gate.stage_metrics(profile_name, node, success=True)
                if stage_metrics:
                    self.evidence_store.log_candidate_metrics(
                        run_id=run_id,
                        candidate_id=node.candidate_config.candidate_id,
                        node_id=node.node_id,
                        metrics=stage_metrics,
                        dataset_version=node.data_version,
                        split="runtime",
                        source="fast_baseline_gate",
                        verified=True,
                        validator="fast_baseline_gate",
                    )
                    metrics.update(stage_metrics)
            artifacts.update(_existing_artifacts(spec.expected_artifacts))
            artifacts.update(_existing_artifacts(expected_ultralytics_artifacts(actual_run_dir)))
        return ExecutionResult(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            status=status,
            command=spec,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            started_at=started,
            ended_at=ended,
            duration_seconds=time.monotonic() - start_time,
            artifacts=artifacts,
            metrics=metrics,
            message=message,
        )


class BenchmarkImportResult(BaseModel):
    """Result of importing benchmark metrics into EvidenceStore."""

    run_id: str
    metrics_path: Path
    run_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    metric_records: list[MetricEvidence] = Field(default_factory=list)
    metrics_output_path: Path
    metric_records_output_path: Path | None = None

    @field_serializer("metrics_path", "metrics_output_path", "metric_records_output_path")
    def serialize_path(self, value: Path | None) -> str | None:
        """Serialize paths portably."""
        return value.as_posix() if value is not None else None


class BenchmarkImporter:
    """Import external benchmark metrics without running training."""

    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store

    def import_metrics(
        self,
        run_id: str,
        metrics_path: Path | str,
        dataset_version: str = "unversioned",
        default_split: str = "val",
        source: str = "benchmark_import",
    ) -> BenchmarkImportResult:
        """Import run-level and candidate/node-level benchmark metrics."""
        path = Path(metrics_path)
        run_metrics = read_metric_mapping(path, dataset_version, default_split, source, source_artifact=path)
        metric_records = read_metric_records(path, dataset_version, default_split, source, source_artifact=path)
        metrics_output_path = self.evidence_store.log_metrics(run_id, run_metrics)
        records_output_path = (
            self.evidence_store.log_metric_records(run_id, metric_records)
            if metric_records
            else None
        )
        return BenchmarkImportResult(
            run_id=run_id,
            metrics_path=path,
            run_metrics=run_metrics,
            metric_records=metric_records,
            metrics_output_path=metrics_output_path,
            metric_records_output_path=records_output_path,
        )

    def import_ultralytics_run(
        self,
        run_id: str,
        node: ExperimentNode,
        run_dir: Path | str,
    ) -> BenchmarkImportResult:
        """Import an Ultralytics run directory as benchmark evidence."""
        from yolo_agent.adapters.ultralytics.training import UltralyticsRunImporter

        metrics = UltralyticsRunImporter(self.evidence_store).import_run(run_id, node, run_dir, source="benchmark_import")
        metrics_output_path = self.evidence_store.log_metrics(run_id, metrics)
        evidence = self.evidence_store.load_run(run_id)
        return BenchmarkImportResult(
            run_id=run_id,
            metrics_path=Path(run_dir),
            run_metrics=metrics,
            metric_records=evidence.metric_records,
            metrics_output_path=metrics_output_path,
            metric_records_output_path=evidence.metric_records_path,
        )


def read_metric_mapping(
    path: Path | str,
    dataset_version: str = "unversioned",
    default_split: str = "val",
    source: str = "import",
    source_artifact: Path | str | None = None,
) -> dict[str, MetricValue]:
    """Read run-level metrics from CSV, YAML, or JSON."""
    metrics_path = Path(path)
    if metrics_path.suffix.lower() == ".csv":
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        if rows and {"metric_name", "value"}.issubset(rows[0]) and ({"candidate_id", "node_id"} & set(rows[0])):
            return _metric_records_to_mapping(
                _metric_records_from_items(rows, dataset_version, default_split, source, source_artifact or metrics_path)
            )
        return _read_csv_metrics(metrics_path)
    data = _read_yaml(metrics_path) if metrics_path.suffix.lower() in {".yaml", ".yml"} else _read_json(metrics_path)
    if isinstance(data, dict) and isinstance(data.get("metrics"), list):
        return _metric_records_to_mapping(
            read_metric_records(metrics_path, dataset_version, default_split, source, source_artifact)
        )
    if isinstance(data, dict) and isinstance(data.get("metric_records"), list):
        return _metric_records_to_mapping(
            read_metric_records(metrics_path, dataset_version, default_split, source, source_artifact)
        )
    if not isinstance(data, dict):
        raise ValueError("Metrics input must contain a mapping.")
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, (float, int, str, bool)) or value is None
    }


def read_metric_records(
    path: Path | str,
    dataset_version: str = "unversioned",
    default_split: str = "val",
    source: str = "import",
    source_artifact: Path | str | None = None,
) -> list[MetricEvidence]:
    """Read candidate/node metric records from CSV, YAML, or JSON."""
    metrics_path = Path(path)
    if metrics_path.suffix.lower() == ".csv":
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as file:
            return _metric_records_from_items(
                list(csv.DictReader(file)),
                dataset_version,
                default_split,
                source,
                source_artifact,
            )

    data = _read_yaml(metrics_path) if metrics_path.suffix.lower() in {".yaml", ".yml"} else _read_json(metrics_path)
    if isinstance(data, dict):
        for key in ("metric_records", "metrics"):
            values = data.get(key)
            if isinstance(values, list):
                return _metric_records_from_items(values, dataset_version, default_split, source, source_artifact)
    if isinstance(data, list):
        return _metric_records_from_items(data, dataset_version, default_split, source, source_artifact)
    return []


def _read_csv_metrics(path: Path) -> dict[str, MetricValue]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}
    if {"metric_name", "value"}.issubset(rows[0]):
        return {
            str(row["metric_name"]): coerce_metric_value(row["value"])
            for row in rows
            if row.get("metric_name")
        }
    if {"metric", "value"}.issubset(rows[0]):
        return {str(row["metric"]): coerce_metric_value(row["value"]) for row in rows if row.get("metric")}
    return {key: coerce_metric_value(value) for key, value in rows[0].items() if key}


def _metric_records_from_items(
    items: list[Any],
    dataset_version: str,
    default_split: str,
    source: str,
    source_artifact: Path | str | None = None,
) -> list[MetricEvidence]:
    records: list[MetricEvidence] = []
    artifact = Path(source_artifact) if source_artifact is not None else None
    for item in items:
        if not isinstance(item, dict):
            continue
        metric_name = item.get("metric_name", item.get("metric"))
        if metric_name is None:
            continue
        candidate_id = item.get("candidate_id")
        node_id = item.get("node_id")
        if candidate_id is None and node_id is None:
            continue
        records.append(
            MetricEvidence(
                candidate_id=str(candidate_id or node_id),
                node_id=str(node_id or candidate_id),
                dataset_version=str(item.get("dataset_version") or dataset_version),
                split=str(item.get("split", default_split)),
                metric_name=str(metric_name),
                value=coerce_metric_value(item.get("value")),
                source=str(item.get("source", source)),
                verified=coerce_bool(item.get("verified"), default=True),
                validator=str(item.get("validator", "benchmark_import")),
                source_artifact=_optional_path(item.get("source_artifact"), artifact),
                metric_schema_version=str(item.get("metric_schema_version", "1.0")),
                higher_is_better=coerce_optional_bool(item.get("higher_is_better")),
                confidence=coerce_optional_float(item.get("confidence")),
            )
        )
    return records


def _metric_records_to_mapping(records: list[MetricEvidence]) -> dict[str, MetricValue]:
    return EvidenceIndex(records).metric_mapping(verified=True, mode="best_value")


def coerce_metric_value(value: object) -> MetricValue:
    """Coerce metric values from CSV strings."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def coerce_bool(value: object, default: bool = False) -> bool:
    """Coerce optional CSV/YAML boolean values."""
    parsed = coerce_optional_bool(value)
    return default if parsed is None else parsed


def coerce_optional_bool(value: object) -> bool | None:
    """Coerce optional CSV/YAML boolean values."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def coerce_optional_float(value: object) -> float | None:
    """Coerce optional CSV/YAML float values."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _optional_path(value: object, default: Path | None = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return default
    return Path(str(value))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file) or {}


def _existing_artifacts(artifacts: dict[str, Path]) -> dict[str, Path]:
    return {name: path for name, path in artifacts.items() if path.exists()}


def _batch_tuning_config_from_training_config(training_config: object | None, default: Any) -> Any:
    """Return batch tuning config from an optional training config."""
    return getattr(training_config, "batch_tuning", default)


def _data_cache_policy_config_from_training_config(training_config: object | None, default: Any) -> Any:
    """Return data cache policy config from an optional training config."""
    return getattr(training_config, "data_cache_policy", default)


def _fast_baseline_gate_config_from_training_config(training_config: object | None, default: Any) -> Any:
    """Return fast baseline gate config from an optional training config."""
    return getattr(training_config, "fast_baseline_gate", default)


def _stop_resume_config_from_training_config(training_config: object | None, default: Any) -> Any:
    """Return stop/resume config from an optional training config."""
    return getattr(training_config, "stop_resume", default)


def _load_or_create_evidence(store: EvidenceStore, run_id: str) -> Any:
    """Load run evidence, creating the run directory if needed."""
    store.create_run(run_id)
    return store.load_run(run_id)


def _stream_artifact_paths(
    store: EvidenceStore | None,
    run_id: str,
    node: ExperimentNode,
) -> dict[str, Path | None]:
    """Return stream artifact paths for a node when evidence storage is available."""
    if store is None:
        return {"stdout_log": None, "runtime_jsonl": None}
    artifacts_dir = store.create_run(run_id) / "artifacts"
    return {
        "stdout_log": artifacts_dir / f"{node.node_id}_ultralytics_stdout.log",
        "runtime_jsonl": artifacts_dir / f"{node.node_id}_runtime_profile.jsonl",
    }


def _run_streaming_process(
    *,
    spec: CommandSpec,
    run_id: str,
    node: ExperimentNode,
    evidence_store: EvidenceStore | None,
    sampler: Any,
    stdout_log_path: Path | None,
    runtime_jsonl_path: Path | None,
    runtime_jsonl_lock: threading.Lock,
    line_metric_parser: Any,
    stop_resume_guard: Any | None = None,
    results_csv_path: Path | None = None,
    stop_decision_queue: queue.Queue[Any] | None = None,
) -> dict[str, Any]:
    """Run a subprocess while streaming logs into events, metrics, and runtime JSONL."""
    event_log = EventLog(evidence_store.create_run(run_id) / "events.jsonl") if evidence_store is not None else None
    _append_executor_event(
        event_log,
        run_id,
        "executor_started",
        "Ultralytics training process started.",
        node,
        {"command": spec.display(), "timeout_seconds": spec.timeout_seconds},
    )
    if stdout_log_path is not None:
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_log_path.write_text("", encoding="utf-8")
    if runtime_jsonl_path is not None:
        runtime_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_jsonl_path.write_text("", encoding="utf-8")

    lines: list[str] = []
    stderr = ""
    line_queue: queue.Queue[str] = queue.Queue()
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    timed_out = False
    stopped_by_guard_reason: str | None = None
    started = time.monotonic()

    try:
        process = subprocess.Popen(
            spec.as_subprocess_args(),
            cwd=spec.cwd,
            env={**os.environ, **spec.env} if spec.env else None,
            shell=spec.shell,
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
            name="ultralytics-log-reader",
            daemon=False,
        )
        reader.start()
        with sampler:
            while True:
                if spec.timeout_seconds is not None and time.monotonic() - started > spec.timeout_seconds:
                    timed_out = True
                    _terminate_process(process)
                    break
                queued_decision = _pop_stop_decision(stop_decision_queue)
                if queued_decision is not None:
                    stopped_by_guard_reason = queued_decision.reason
                    _terminate_process(process)
                    break
                try:
                    line = line_queue.get(timeout=0.2)
                except queue.Empty:
                    queued_decision = _pop_stop_decision(stop_decision_queue)
                    if queued_decision is not None:
                        stopped_by_guard_reason = queued_decision.reason
                        _terminate_process(process)
                        break
                    decision = _observe_stop_resume_results(
                        stop_resume_guard=stop_resume_guard,
                        results_csv_path=results_csv_path,
                        run_id=run_id,
                        node=node,
                        evidence_store=evidence_store,
                        event_log=event_log,
                        runtime_jsonl_path=runtime_jsonl_path,
                        runtime_jsonl_lock=runtime_jsonl_lock,
                    )
                    if decision is not None and decision.should_stop:
                        stopped_by_guard_reason = decision.reason
                        _terminate_process(process)
                        break
                    if process.poll() is not None and not reader.is_alive() and line_queue.empty():
                        break
                    continue
                lines.append(line)
                _handle_stream_line(
                    line=line,
                    run_id=run_id,
                    node=node,
                    evidence_store=evidence_store,
                    event_log=event_log,
                    stdout_log_path=stdout_log_path,
                    runtime_jsonl_path=runtime_jsonl_path,
                    runtime_jsonl_lock=runtime_jsonl_lock,
                    line_metric_parser=line_metric_parser,
                )
                decision = _observe_stop_resume_results(
                    stop_resume_guard=stop_resume_guard,
                    results_csv_path=results_csv_path,
                    run_id=run_id,
                    node=node,
                    evidence_store=evidence_store,
                    event_log=event_log,
                    runtime_jsonl_path=runtime_jsonl_path,
                    runtime_jsonl_lock=runtime_jsonl_lock,
                )
                if decision is not None and decision.should_stop:
                    stopped_by_guard_reason = decision.reason
                    _terminate_process(process)
                    break
        while not line_queue.empty():
            line = line_queue.get()
            lines.append(line)
            _handle_stream_line(
                line=line,
                run_id=run_id,
                node=node,
                evidence_store=evidence_store,
                event_log=event_log,
                stdout_log_path=stdout_log_path,
                runtime_jsonl_path=runtime_jsonl_path,
                runtime_jsonl_lock=runtime_jsonl_lock,
                line_metric_parser=line_metric_parser,
            )
        return_code = None if timed_out or stopped_by_guard_reason is not None else process.wait(timeout=5)
        if reader is not None:
            reader.join(timeout=5)
    except KeyboardInterrupt:
        if process is not None:
            _terminate_process(process)
        _append_executor_event(
            event_log,
            run_id,
            "executor_failed",
            "Ultralytics training interrupted by user; child process tree was terminated.",
            node,
            {"duration_seconds": round(time.monotonic() - started, 6), "interrupted_by_user": True},
        )
        raise
    except OSError as exc:
        return_code = None
        stderr = str(exc)
    except subprocess.TimeoutExpired:
        timed_out = True
        if process is not None:
            _terminate_process(process)
        return_code = None
    finally:
        if reader is not None:
            reader.join(timeout=5)

    stdout = "".join(lines)
    if stopped_by_guard_reason is not None:
        status: ExecutionStatus = "failed"
        message = f"StopResumeGuard stopped training: {stopped_by_guard_reason}"
        event_type = "executor_failed"
    elif timed_out:
        status: ExecutionStatus = "failed"
        message = f"Ultralytics training timed out after {spec.timeout_seconds} seconds."
        event_type = "executor_timeout"
    else:
        status = "completed" if return_code == 0 else "failed"
        message = "Ultralytics training completed." if status == "completed" else "Ultralytics training failed."
        event_type = "executor_completed" if status == "completed" else "executor_failed"
    _append_executor_event(
        event_log,
        run_id,
        event_type,
        message,
        node,
        {"return_code": return_code, "duration_seconds": round(time.monotonic() - started, 6)},
    )
    return {
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code,
        "message": message,
        "timed_out": timed_out,
        "timeout_seconds": spec.timeout_seconds,
    }


def _read_process_stdout(process: subprocess.Popen[str], line_queue: queue.Queue[str]) -> None:
    """Read process output in a background thread."""
    stream = process.stdout
    if stream is None:
        return
    if hasattr(stream, "readline"):
        while True:
            line = stream.readline()
            if line == "":
                break
            line_queue.put(line)
    else:
        for line in stream:
            line_queue.put(str(line))


def _handle_stream_line(
    *,
    line: str,
    run_id: str,
    node: ExperimentNode,
    evidence_store: EvidenceStore | None,
    event_log: EventLog | None,
    stdout_log_path: Path | None,
    runtime_jsonl_path: Path | None,
    runtime_jsonl_lock: threading.Lock,
    line_metric_parser: Any,
) -> None:
    """Persist one streaming log line and any metrics parsed from it."""
    if stdout_log_path is not None:
        with stdout_log_path.open("a", encoding="utf-8") as file:
            file.write(line)
    clean_line = line.rstrip("\r\n")
    _append_executor_event(
        event_log,
        run_id,
        "executor_log",
        clean_line[:500],
        node,
        {"stream": "stdout"},
    )
    metrics = line_metric_parser(clean_line)
    if not metrics:
        return
    if evidence_store is not None:
        evidence_store.log_candidate_metrics(
            run_id=run_id,
            candidate_id=node.candidate_config.candidate_id,
            node_id=node.node_id,
            metrics=metrics,
            dataset_version=node.data_version,
            split="runtime",
            source="ultralytics_stream",
            verified=True,
            validator="ultralytics_stream_parser",
            source_artifact=stdout_log_path,
        )
    _append_executor_event(
        event_log,
        run_id,
        "executor_metric",
        "Parsed live Ultralytics runtime metrics.",
        node,
        {"metrics": metrics},
    )
    if runtime_jsonl_path is not None:
        _append_runtime_jsonl(
            runtime_jsonl_path,
            {
                "record_type": "log_line",
                "run_id": run_id,
                "candidate_id": node.candidate_config.candidate_id,
                "node_id": node.node_id,
                "dataset_version": node.data_version,
                "metrics": metrics,
                "line": clean_line,
            },
            runtime_jsonl_lock,
        )


def _observe_stop_resume_results(
    *,
    stop_resume_guard: Any | None,
    results_csv_path: Path | None,
    run_id: str,
    node: ExperimentNode,
    evidence_store: EvidenceStore | None,
    event_log: EventLog | None,
    runtime_jsonl_path: Path | None,
    runtime_jsonl_lock: threading.Lock,
) -> Any | None:
    """Observe results.csv and persist a stop/resume decision when one appears."""
    if stop_resume_guard is None or results_csv_path is None:
        return None
    decision = stop_resume_guard.observe_results_csv(results_csv_path)
    if decision is None:
        return None
    _persist_stop_resume_decision(
        decision=decision,
        run_id=run_id,
        node=node,
        evidence_store=evidence_store,
        event_log=event_log,
        runtime_jsonl_path=runtime_jsonl_path,
        runtime_jsonl_lock=runtime_jsonl_lock,
    )
    return decision


def _pop_stop_decision(decisions: queue.Queue[Any] | None) -> Any | None:
    """Return the next stop decision from a callback queue if available."""
    if decisions is None:
        return None
    try:
        return decisions.get_nowait()
    except queue.Empty:
        return None


def _persist_stop_resume_decision(
    *,
    decision: Any,
    run_id: str,
    node: ExperimentNode,
    evidence_store: EvidenceStore | None,
    event_log: EventLog | None,
    runtime_jsonl_path: Path | None,
    runtime_jsonl_lock: threading.Lock,
) -> None:
    """Persist stop/resume guard decisions as evidence and events."""
    metrics = decision.to_metrics()
    if evidence_store is not None:
        evidence_store.log_candidate_metrics(
            run_id=run_id,
            candidate_id=node.candidate_config.candidate_id,
            node_id=node.node_id,
            metrics=metrics,
            dataset_version=node.data_version,
            split="runtime",
            source="stop_resume_guard",
            verified=True,
            validator="stop_resume_guard",
            source_artifact=runtime_jsonl_path,
        )
    _append_executor_event(
        event_log,
        run_id,
        "executor_metric",
        f"Stop/Resume guard flagged {decision.kind}: {decision.reason}",
        node,
        {
            "guard": "stop_resume",
            "kind": decision.kind,
            "severity": decision.severity,
            "recommendations": decision.recommendations,
            "should_stop": decision.should_stop,
            "evidence": decision.evidence,
        },
    )
    if runtime_jsonl_path is not None:
        _append_runtime_jsonl(
            runtime_jsonl_path,
            {
                "record_type": "stop_resume_decision",
                "run_id": run_id,
                "candidate_id": node.candidate_config.candidate_id,
                "node_id": node.node_id,
                "dataset_version": node.data_version,
                "decision": decision.model_dump(mode="json"),
            },
            runtime_jsonl_lock,
        )


def _append_runtime_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    """Append one runtime profile event to a JSONL artifact."""
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _append_executor_event(
    event_log: EventLog | None,
    run_id: str,
    event_type: str,
    message: str,
    node: ExperimentNode,
    details: dict[str, Any] | None = None,
) -> None:
    """Append an executor event when a run event log exists."""
    if event_log is None:
        return
    event_log.append(
        run_id=run_id,
        event_type=event_type,  # type: ignore[arg-type]
        message=message,
        details={
            "candidate_id": node.candidate_config.candidate_id,
            "node_id": node.node_id,
            "dataset_version": node.data_version,
            **(details or {}),
        },
    )


def _event_log_for_store(store: EvidenceStore | None, run_id: str) -> EventLog | None:
    """Return the run event log for an optional evidence store."""
    if store is None:
        return None
    return EventLog(store.create_run(run_id) / "events.jsonl")


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate a process tree best-effort."""
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


def _with_execution_identity(spec: CommandSpec, node: ExperimentNode, run_id: str) -> CommandSpec:
    """Ensure executor commands carry stable run/candidate/node identity."""
    metadata = {
        **spec.metadata,
        "run_id": run_id,
        "candidate_id": node.candidate_config.candidate_id,
        "node_id": node.node_id,
        "dataset_version": node.data_version,
        "seed": node.seed,
    }
    return spec.model_copy(update={"metadata": metadata})


def _ultralytics_run_dir(spec: CommandSpec) -> Path | None:
    results_csv = spec.expected_artifacts.get("results_csv")
    if results_csv is not None:
        return results_csv.parent
    project = _arg_value(spec.argv, "project")
    name = _arg_value(spec.argv, "name")
    if project and name:
        return Path(project) / name
    return None


def _resolve_completed_ultralytics_run_dir(
    *,
    spec: CommandSpec,
    expected_run_dir: Path | None,
    stdout: str,
    stderr: str,
) -> Path | None:
    """Resolve the actual Ultralytics output directory after a run.

    Ultralytics can prepend its default task directory to relative project
    paths on some CLI versions. The executor therefore trusts the observed
    ``Results saved to ...`` line before falling back to the planned path.
    """
    observed = _run_dir_from_results_saved_line("\n".join(part for part in (stdout, stderr) if part))
    if observed is not None and (observed / "results.csv").is_file():
        return observed
    if expected_run_dir is not None and (expected_run_dir / "results.csv").is_file():
        return expected_run_dir
    name = _arg_value(spec.argv, "name")
    if not name:
        return expected_run_dir
    roots: list[Path] = []
    project = _arg_value(spec.argv, "project")
    if project:
        project_path = Path(project)
        roots.extend([project_path, project_path.parent, Path("runs")])
    else:
        roots.append(Path("runs"))
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        matches.extend(path.parent for path in root.rglob("results.csv") if path.parent.name == name)
    if not matches:
        return expected_run_dir
    return max(matches, key=lambda path: (path / "results.csv").stat().st_mtime)


def _run_dir_from_results_saved_line(text: str) -> Path | None:
    """Extract a run directory from Ultralytics' completion message."""
    matches = re.findall(r"Results saved to\s+(.+)", text)
    if not matches:
        return None
    raw = matches[-1].strip().strip("\"'")
    # Ultralytics may style the path with terminal color codes; strip common
    # control sequences and trailing punctuation without touching Windows ':'.
    raw = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw).strip().rstrip(".")
    return Path(raw) if raw else None


def _arg_value(argv: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for arg in argv:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _training_profile_from_spec(spec: CommandSpec) -> str | None:
    value = spec.metadata.get("training_budget_profile")
    return str(value) if value not in {None, "", "custom"} else None


def _fast_gate_candidate_scope(profile_name: str, node: ExperimentNode) -> str | None:
    """Return candidate scope for staged baseline gate checks."""
    if profile_name in {"debug", "pilot", "baseline_full", "baseline_confirm"}:
        return None
    return node.candidate_config.candidate_id


def _fast_baseline_gate_applies(profile_name: str, node: ExperimentNode) -> bool:
    """Return whether the staged baseline gate should guard this node.

    The fast baseline gate enforces the baseline runbook
    debug -> pilot -> full baseline -> confirmation. Auto-loop candidate pilots
    are guarded by policy/evidence/promotion gates instead, so applying the
    baseline gate to action candidates incorrectly skips real pilot experiments.
    """
    if profile_name not in {"debug", "pilot", "baseline_full", "baseline_confirm"}:
        return False
    candidate = node.candidate_config
    if candidate.action_id:
        return False
    if candidate.action_domain not in {"", "baseline", "model"}:
        return False
    return "baseline" in candidate.candidate_id or "_coco_" in candidate.candidate_id or candidate.candidate_id.startswith("yolo")


def _path_arg_value(argv: list[str], key: str) -> Path | None:
    value = _arg_value(argv, key)
    return Path(value) if value else None


def _resolve_executable(command: str) -> str | None:
    found = shutil.which(command)
    if found is not None:
        return found
    scripts_dir = Path(sysconfig.get_path("scripts"))
    suffixes = [".exe", ".cmd", ".bat", ""]
    for suffix in suffixes:
        candidate = scripts_dir / f"{command}{suffix}"
        if candidate.is_file():
            return str(candidate)
    return None
