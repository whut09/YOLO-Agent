"""Runtime profiling helpers for Ultralytics runs."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.experiment_graph import MetricValue


IT_PER_SECOND_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*it/s", re.IGNORECASE)
SECONDS_PER_IT_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*s/it", re.IGNORECASE)
GPU_MEM_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>G|M)(?:i?B)?", re.IGNORECASE)
AUTOBATCH_RE = re.compile(r"AutoBatch:\s+Using batch-size\s+(?P<value>\d+)", re.IGNORECASE)
SLOW_DATA_RE = re.compile(r"slow image access|dataloader.*wait|workers are bottleneck", re.IGNORECASE)


class RuntimeSample(BaseModel):
    """One point-in-time runtime sample."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gpu_util_percent: float | None = Field(default=None, ge=0.0)
    gpu_memory_used_mb: float | None = Field(default=None, ge=0.0)
    gpu_memory_total_mb: float | None = Field(default=None, ge=0.0)
    gpu_memory_util_percent: float | None = Field(default=None, ge=0.0)
    power_w: float | None = Field(default=None, ge=0.0)
    source: str = "nvidia_smi"


class RuntimeProfile(BaseModel):
    """Runtime facts extracted from a training run."""

    run_dir: Path
    batch_size: int | str | None = None
    cache_mode: str | None = None
    dataloader_workers: int | None = None
    avg_it_per_sec: float | None = None
    max_it_per_sec: float | None = None
    min_it_per_sec: float | None = None
    epoch_time_seconds: float | None = None
    avg_gpu_util_percent: float | None = None
    max_gpu_memory_used_mb: float | None = None
    dataloader_wait_warning: bool = False
    warnings: list[str] = Field(default_factory=list)
    samples: list[RuntimeSample] = Field(default_factory=list)

    @field_serializer("run_dir")
    def serialize_run_dir(self, value: Path) -> str:
        """Serialize run directory portably."""
        return value.as_posix()

    def to_metrics(self) -> dict[str, MetricValue]:
        """Return profile facts as node-level metric evidence."""
        metrics: dict[str, MetricValue] = {
            "runtime_dataloader_wait_warning": self.dataloader_wait_warning,
        }
        optional_values: dict[str, MetricValue] = {
            "runtime_batch_size": self.batch_size,
            "runtime_cache_mode": self.cache_mode,
            "runtime_dataloader_workers": self.dataloader_workers,
            "runtime_avg_it_per_sec": self.avg_it_per_sec,
            "runtime_max_it_per_sec": self.max_it_per_sec,
            "runtime_min_it_per_sec": self.min_it_per_sec,
            "runtime_epoch_time_seconds": self.epoch_time_seconds,
            "runtime_avg_gpu_util_percent": self.avg_gpu_util_percent,
            "runtime_max_gpu_memory_used_mb": self.max_gpu_memory_used_mb,
        }
        metrics.update({key: value for key, value in optional_values.items() if value is not None})
        return metrics


class RuntimeProfiler:
    """Build runtime profiles from Ultralytics artifacts and optional GPU samples."""

    def profile(
        self,
        run_dir: Path | str,
        log_path: Path | str | None = None,
        stdout: str | None = None,
        samples: list[RuntimeSample] | None = None,
        sample_gpu: bool = True,
    ) -> RuntimeProfile:
        """Profile one Ultralytics run directory without requiring a GPU."""
        directory = Path(run_dir)
        args = _read_yaml_mapping(directory / "args.yaml")
        log_text = _read_log_text(directory, log_path, stdout)
        it_per_sec = _parse_it_per_sec(log_text)
        runtime_samples = list(samples or [])
        runtime_samples.extend(_parse_gpu_memory_samples(log_text))
        warnings = _warnings_from_log(log_text)
        if sample_gpu:
            sample = sample_nvidia_smi()
            if sample is not None:
                runtime_samples.append(sample)

        batch_size = _batch_size(args, log_text)
        cache_mode = _cache_mode(args)
        workers = _optional_int(args.get("workers"))
        epoch_time = _epoch_time_seconds(directory / "results.csv")
        gpu_util_values = [
            sample.gpu_util_percent
            for sample in runtime_samples
            if sample.gpu_util_percent is not None
        ]
        memory_values = [
            sample.gpu_memory_used_mb
            for sample in runtime_samples
            if sample.gpu_memory_used_mb is not None
        ]
        dataloader_wait_warning = bool(SLOW_DATA_RE.search(log_text))
        if dataloader_wait_warning:
            warnings.append("Slow image access or dataloader wait was reported by the training log.")
        if cache_mode in {None, "False", "false", "0", "none"}:
            warnings.append("Dataset cache is disabled; slow storage can bottleneck training throughput.")

        return RuntimeProfile(
            run_dir=directory,
            batch_size=batch_size,
            cache_mode=cache_mode,
            dataloader_workers=workers,
            avg_it_per_sec=round(mean(it_per_sec), 6) if it_per_sec else None,
            max_it_per_sec=round(max(it_per_sec), 6) if it_per_sec else None,
            min_it_per_sec=round(min(it_per_sec), 6) if it_per_sec else None,
            epoch_time_seconds=epoch_time,
            avg_gpu_util_percent=round(mean(gpu_util_values), 6) if gpu_util_values else None,
            max_gpu_memory_used_mb=round(max(memory_values), 4) if memory_values else None,
            dataloader_wait_warning=dataloader_wait_warning,
            warnings=_dedupe(warnings),
            samples=runtime_samples,
        )


class RuntimeSampler:
    """Background nvidia-smi sampler for a running subprocess."""

    def __init__(
        self,
        interval_seconds: float = 10.0,
        enabled: bool = True,
        sample_callback: Callable[[RuntimeSample], None] | None = None,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self.sample_callback = sample_callback
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._samples: list[RuntimeSample] = []

    def __enter__(self) -> "RuntimeSampler":
        """Start the sampler thread."""
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="runtime-profiler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Stop the sampler thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, min(self.interval_seconds, 5.0)))

    @property
    def samples(self) -> list[RuntimeSample]:
        """Return a stable copy of collected samples."""
        with self._lock:
            return list(self._samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = sample_nvidia_smi()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
                if self.sample_callback is not None:
                    self.sample_callback(sample)
            self._stop.wait(self.interval_seconds)


def write_runtime_profile(profile: RuntimeProfile, path: Path | str) -> Path:
    """Write a runtime profile JSON artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(profile.model_dump(mode="json"), file, indent=2, sort_keys=True)
    return output


def parse_runtime_line_metrics(line: str) -> dict[str, MetricValue]:
    """Parse live runtime facts from one Ultralytics log line."""
    metrics: dict[str, MetricValue] = {}
    it_match = IT_PER_SECOND_RE.search(line)
    if it_match:
        metrics["runtime_stream_it_per_sec"] = float(it_match.group("value"))
    seconds_match = SECONDS_PER_IT_RE.search(line)
    if seconds_match:
        seconds = float(seconds_match.group("value"))
        if seconds > 0:
            metrics["runtime_stream_it_per_sec"] = round(1.0 / seconds, 6)
    if "GPU_mem" in line or "GPU memory" in line or "it/s" in line or "s/it" in line:
        memory_match = GPU_MEM_RE.search(line)
        if memory_match:
            value = float(memory_match.group("value"))
            unit = memory_match.group("unit").lower()
            metrics["runtime_stream_gpu_memory_used_mb"] = value * 1024 if unit == "g" else value
    if SLOW_DATA_RE.search(line):
        metrics["runtime_dataloader_wait_warning"] = True
    return metrics


def sample_nvidia_smi() -> RuntimeSample | None:
    """Return one current GPU sample from nvidia-smi, or None when unavailable."""
    query = (
        "utilization.gpu,memory.used,memory.total,utilization.memory,power.draw"
    )
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
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if completed.returncode != 0:
        return None
    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return None
    values = [part.strip() for part in first_line.split(",")]
    if len(values) < 5:
        return None
    return RuntimeSample(
        gpu_util_percent=_optional_float(values[0]),
        gpu_memory_used_mb=_optional_float(values[1]),
        gpu_memory_total_mb=_optional_float(values[2]),
        gpu_memory_util_percent=_optional_float(values[3]),
        power_w=_optional_float(values[4]),
        source="nvidia_smi",
    )


def _read_log_text(directory: Path, log_path: Path | str | None, stdout: str | None) -> str:
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    candidates = [Path(log_path)] if log_path is not None else []
    candidates.extend(
        [
            directory / "stdout.log",
            directory / "train.log",
            directory.parent / f"{directory.name}_launch" / "stdout.log",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            parts.append(candidate.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _parse_it_per_sec(text: str) -> list[float]:
    values = [float(match.group("value")) for match in IT_PER_SECOND_RE.finditer(text)]
    for match in SECONDS_PER_IT_RE.finditer(text):
        seconds = float(match.group("value"))
        if seconds > 0:
            values.append(1.0 / seconds)
    return values


def _parse_gpu_memory_samples(text: str) -> list[RuntimeSample]:
    samples: list[RuntimeSample] = []
    for line in text.splitlines():
        if "GPU_mem" not in line and "GPU memory" not in line:
            continue
        matches = list(GPU_MEM_RE.finditer(line))
        if not matches:
            continue
        value = float(matches[0].group("value"))
        unit = matches[0].group("unit").lower()
        samples.append(
            RuntimeSample(
                gpu_memory_used_mb=value * 1024 if unit == "g" else value,
                source="ultralytics_log",
            )
        )
    return samples


def _warnings_from_log(text: str) -> list[str]:
    warnings: list[str] = []
    for line in text.splitlines():
        if "warning" in line.lower() or "slow image access" in line.lower():
            warnings.append(line.strip())
    return warnings


def _batch_size(args: dict[str, Any], log_text: str) -> int | str | None:
    match = AUTOBATCH_RE.search(log_text)
    if match:
        return int(match.group("value"))
    value = args.get("batch")
    if value is None:
        return None
    parsed = _optional_int(value)
    return parsed if parsed is not None else str(value)


def _cache_mode(args: dict[str, Any]) -> str | None:
    value = args.get("cache")
    if value is None:
        return None
    return str(value)


def _epoch_time_seconds(results_csv: Path) -> float | None:
    if not results_csv.is_file():
        return None
    with results_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    times = [_optional_float(row.get("time")) for row in rows if row.get("time") not in {None, ""}]
    clean_times = [value for value in times if value is not None]
    if not clean_times:
        return None
    if len(clean_times) == 1:
        return round(clean_times[0], 6)
    deltas = [
        current - previous
        for previous, current in zip(clean_times, clean_times[1:])
        if current >= previous
    ]
    if deltas:
        return round(mean(deltas), 6)
    return round(mean(clean_times), 6)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a", "[not supported]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in seen:
            result.append(clean)
            seen.add(clean)
    return result
