"""Stop/resume guardrails for long Ultralytics training runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.adapters.ultralytics.runtime_profiler import RuntimeSample
from yolo_agent.core.experiment_graph import MetricValue


StopResumeDecisionKind = Literal["runtime_bottleneck", "training_failure"]


class StopResumeConfig(BaseModel):
    """Runtime guard configuration for stop/resume recommendations."""

    enabled: bool = True
    stop_on_trigger: bool = False
    low_gpu_util_threshold: float = Field(default=40.0, ge=0.0, le=100.0)
    low_gpu_duration_seconds: float = Field(default=600.0, ge=0.0)
    low_gpu_min_samples: int = Field(default=3, ge=1)
    early_map_metric: str = "metrics/mAP50-95(B)"
    early_map_max_epoch: int = Field(default=10, ge=1)
    early_map_drop_threshold: float = Field(default=0.05, ge=0.0)


class StopResumeDecision(BaseModel):
    """One runtime stop/resume diagnosis."""

    kind: StopResumeDecisionKind
    severity: Literal["low", "medium", "high"] = "medium"
    reason: str
    evidence: dict[str, MetricValue] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    should_stop: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_metrics(self) -> dict[str, MetricValue]:
        """Return node-level metrics for this decision."""
        prefix = self.kind
        metrics: dict[str, MetricValue] = {
            prefix: True,
            f"{prefix}_severity": self.severity,
            f"{prefix}_should_stop": self.should_stop,
            f"{prefix}_reason": self.reason,
            f"{prefix}_recommendations": ", ".join(self.recommendations),
        }
        metrics.update({f"{prefix}_{key}": value for key, value in self.evidence.items()})
        return metrics


class StopResumeGuard:
    """Detect runtime bottlenecks and early training failures without destroying evidence."""

    def __init__(self, config: StopResumeConfig | None = None) -> None:
        self.config = config or StopResumeConfig()
        self._low_gpu_started_at: datetime | None = None
        self._low_gpu_sample_count = 0
        self._emitted: set[StopResumeDecisionKind] = set()
        self._best_map: float | None = None
        self._last_results_signature: tuple[int, int] | None = None

    def observe_sample(self, sample: RuntimeSample) -> StopResumeDecision | None:
        """Inspect one GPU sample for sustained low utilization."""
        if not self.config.enabled or sample.gpu_util_percent is None:
            return None
        if sample.gpu_util_percent >= self.config.low_gpu_util_threshold:
            self._low_gpu_started_at = None
            self._low_gpu_sample_count = 0
            return None
        if self._low_gpu_started_at is None:
            self._low_gpu_started_at = sample.created_at
            self._low_gpu_sample_count = 1
            return None
        self._low_gpu_sample_count += 1
        elapsed = max(0.0, (sample.created_at - self._low_gpu_started_at).total_seconds())
        if (
            "runtime_bottleneck" in self._emitted
            or elapsed < self.config.low_gpu_duration_seconds
            or self._low_gpu_sample_count < self.config.low_gpu_min_samples
        ):
            return None
        self._emitted.add("runtime_bottleneck")
        return StopResumeDecision(
            kind="runtime_bottleneck",
            severity="medium",
            reason=(
                f"GPU util stayed below {self.config.low_gpu_util_threshold:.1f}% "
                f"for {elapsed:.1f}s across {self._low_gpu_sample_count} samples."
            ),
            evidence={
                "low_gpu_util_threshold": self.config.low_gpu_util_threshold,
                "low_gpu_duration_seconds": round(elapsed, 3),
                "low_gpu_sample_count": self._low_gpu_sample_count,
                "latest_gpu_util_percent": sample.gpu_util_percent,
            },
            recommendations=[
                "inspect_dataloader_wait",
                "increase_workers_or_enable_cache_disk",
                "rerun_batch_tuner",
                "resume_from_last_checkpoint_after_config_change",
            ],
            should_stop=self.config.stop_on_trigger,
        )

    def observe_results_csv(self, path: Path | str | None) -> StopResumeDecision | None:
        """Inspect results.csv for early mAP collapse."""
        if not self.config.enabled or path is None:
            return None
        results_path = Path(path)
        if not results_path.is_file():
            return None
        stat = results_path.stat()
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
        if signature == self._last_results_signature:
            return None
        self._last_results_signature = signature
        rows = _read_rows(results_path)
        if not rows:
            return None
        latest = rows[-1]
        epoch = _epoch(latest)
        if epoch is None or epoch > self.config.early_map_max_epoch:
            return None
        value = _metric(latest, self.config.early_map_metric)
        if value is None:
            return None
        if self._best_map is None or value > self._best_map:
            self._best_map = value
            return None
        drop = self._best_map - value
        if "training_failure" in self._emitted or drop < self.config.early_map_drop_threshold:
            return None
        self._emitted.add("training_failure")
        return StopResumeDecision(
            kind="training_failure",
            severity="high",
            reason=(
                f"Early {self.config.early_map_metric} dropped by {drop:.4f} "
                f"at epoch {epoch}."
            ),
            evidence={
                "early_epoch": epoch,
                "early_map_best": round(self._best_map, 6),
                "early_map_latest": round(value, 6),
                "early_map_drop": round(drop, 6),
            },
            recommendations=[
                "check_learning_rate_and_warmup",
                "inspect_bad_labels_or_corrupt_images",
                "resume_from_last_checkpoint_with_safer_lr",
                "keep_current_evidence_before_restarting",
            ],
            should_stop=self.config.stop_on_trigger,
        )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _epoch(row: dict[str, str]) -> int | None:
    value = row.get("epoch")
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _metric(row: dict[str, str], name: str) -> float | None:
    value = row.get(name)
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
