"""Executor abstractions for controlled experiment execution."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode, MetricEvidence, MetricValue


ExecutionStatus = Literal["planned", "dry_run", "completed", "failed", "skipped"]


class CommandSpec(BaseModel):
    """A command prepared from an experiment node."""

    command: str
    args: list[str] = Field(default_factory=list)
    cwd: Path | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int | None = None
    shell: bool = False
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_serializer("cwd")
    def serialize_cwd(self, value: Path | None) -> str | None:
        """Serialize paths portably."""
        return value.as_posix() if value is not None else None

    @classmethod
    def from_experiment_node(cls, node: ExperimentNode) -> "CommandSpec":
        """Build a shell-safe command spec from an experiment node command string."""
        return cls(
            command=node.command,
            shell=True,
            metadata={
                "node_id": node.node_id,
                "candidate_id": node.candidate_config.candidate_id,
                "dataset_version": node.data_version,
                "seed": node.seed,
            },
        )

    def as_subprocess_args(self) -> str | list[str]:
        """Return subprocess command representation."""
        if self.shell:
            return " ".join([self.command, *self.args]).strip()
        return [self.command, *self.args]


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
    """Placeholder executor for future verified Ultralytics integration."""

    def execute(self, node: ExperimentNode, run_id: str, command: CommandSpec | None = None) -> ExecutionResult:
        """Skip execution until training integration is explicitly verified."""
        spec = command or CommandSpec.from_experiment_node(node)
        now = datetime.now(timezone.utc)
        return ExecutionResult(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            status="skipped",
            command=spec,
            started_at=now,
            ended_at=now,
            duration_seconds=0.0,
            message="UltralyticsExecutor requires verified implementation before training.",
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
        run_metrics = read_metric_mapping(path)
        metric_records = read_metric_records(path, dataset_version, default_split, source)
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


def read_metric_mapping(path: Path | str) -> dict[str, MetricValue]:
    """Read run-level metrics from CSV, YAML, or JSON."""
    metrics_path = Path(path)
    if metrics_path.suffix.lower() == ".csv":
        return _read_csv_metrics(metrics_path)
    data = _read_yaml(metrics_path) if metrics_path.suffix.lower() in {".yaml", ".yml"} else _read_json(metrics_path)
    if isinstance(data, dict) and isinstance(data.get("metrics"), list):
        return _metric_records_to_mapping(read_metric_records(metrics_path))
    if isinstance(data, dict) and isinstance(data.get("metric_records"), list):
        return _metric_records_to_mapping(read_metric_records(metrics_path))
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
) -> list[MetricEvidence]:
    """Read candidate/node metric records from CSV, YAML, or JSON."""
    metrics_path = Path(path)
    if metrics_path.suffix.lower() == ".csv":
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as file:
            return _metric_records_from_items(list(csv.DictReader(file)), dataset_version, default_split, source)

    data = _read_yaml(metrics_path) if metrics_path.suffix.lower() in {".yaml", ".yml"} else _read_json(metrics_path)
    if isinstance(data, dict):
        for key in ("metric_records", "metrics"):
            values = data.get(key)
            if isinstance(values, list):
                return _metric_records_from_items(values, dataset_version, default_split, source)
    if isinstance(data, list):
        return _metric_records_from_items(data, dataset_version, default_split, source)
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
) -> list[MetricEvidence]:
    records: list[MetricEvidence] = []
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
            )
        )
    return records


def _metric_records_to_mapping(records: list[MetricEvidence]) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {}
    for record in records:
        metrics.setdefault(record.metric_name, record.value)
    return metrics


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


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file) or {}
