"""Executor abstractions for controlled experiment execution."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sysconfig
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.evidence_store import EvidenceStore
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
        )
        from yolo_agent.adapters.ultralytics.batch_tuner import (
            BatchTuner,
            BatchTuningConfig,
            apply_selected_batch,
            should_tune_batch,
        )
        from yolo_agent.adapters.ultralytics.runtime_profiler import RuntimeSampler

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

        batch_tuning_config = _batch_tuning_config_from_training_config(
            self.training_config,
            BatchTuningConfig(),
        )
        if should_tune_batch(spec, batch_tuning_config):
            tuning_result = BatchTuner(
                config=batch_tuning_config,
                evidence_store=self.evidence_store,
            ).tune(run_id, node, spec)
            if tuning_result.selected_batch is not None:
                spec = apply_selected_batch(spec, tuning_result.selected_batch)

        start_time = time.monotonic()
        sampler = RuntimeSampler()
        try:
            with sampler:
                completed = subprocess.run(
                    spec.as_subprocess_args(),
                    cwd=spec.cwd,
                    env={**os.environ, **spec.env} if spec.env else None,
                    timeout=spec.timeout_seconds,
                    shell=False,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            status: ExecutionStatus = "completed" if completed.returncode == 0 else "failed"
            stdout = completed.stdout
            stderr = completed.stderr
            return_code = completed.returncode
            message = "Ultralytics training completed." if status == "completed" else "Ultralytics training failed."
        except subprocess.TimeoutExpired as exc:
            status = "failed"
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return_code = None
            message = f"Ultralytics training timed out after {spec.timeout_seconds} seconds."
        ended = datetime.now(timezone.utc)
        artifacts = _existing_artifacts(spec.expected_artifacts)
        metrics: dict[str, MetricValue] = {}
        run_dir = _ultralytics_run_dir(spec)
        if status == "completed" and self.evidence_store is not None and run_dir is not None:
            metrics = UltralyticsRunImporter(self.evidence_store).import_run(
                run_id,
                node,
                run_dir,
                stdout="\n".join(part for part in (stdout, stderr) if part),
                runtime_samples=sampler.samples,
            )
            artifacts.update(_existing_artifacts(spec.expected_artifacts))
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


def _arg_value(argv: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for arg in argv:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


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
