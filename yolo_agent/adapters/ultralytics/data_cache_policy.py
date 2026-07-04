"""Data cache policy for Ultralytics training commands."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode, MetricValue


CacheMode = Literal["ram", "disk", "False"]
StorageKind = Literal["nvme", "ssd", "hdd", "unknown"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DataCachePolicyConfig(BaseModel):
    """Controls automatic dataset cache selection."""

    enabled: bool = True
    candidate_cache_modes: list[CacheMode] = Field(default_factory=lambda: ["ram", "disk", "False"])
    decoded_ram_multiplier: float = Field(default=3.0, gt=0.0)
    min_free_ram_after_cache_gb: float = Field(default=8.0, ge=0.0)
    ram_cache_max_available_fraction: float = Field(default=0.75, gt=0.0, le=1.0)
    disk_cache_requires_nvme: bool = True
    respect_explicit_cache: bool = False
    worker_increment_on_no_cache: int = Field(default=4, ge=0)
    max_workers: int = Field(default=16, ge=0)


class MemorySnapshot(BaseModel):
    """System memory snapshot used for cache decisions."""

    total_bytes: int | None = None
    available_bytes: int | None = None
    source: str = "unknown"


class DataCacheDecision(BaseModel):
    """One cache-mode decision for a training command."""

    selected_cache: CacheMode
    selected_workers: int | None = None
    dataset_size_bytes: int = 0
    estimated_ram_cache_bytes: int = 0
    memory: MemorySnapshot = Field(default_factory=MemorySnapshot)
    storage_kind: StorageKind = "unknown"
    applied: bool = False
    preheat_recommended: bool = False
    reason: str = ""
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_metrics(self) -> dict[str, MetricValue]:
        """Return cache policy facts as metric evidence."""
        return {
            "data_cache_policy_applied": self.applied,
            "data_cache_selected_cache": self.selected_cache,
            "data_cache_selected_workers": self.selected_workers,
            "data_cache_dataset_size_mb": round(self.dataset_size_bytes / (1024 * 1024), 4),
            "data_cache_estimated_ram_cache_mb": round(self.estimated_ram_cache_bytes / (1024 * 1024), 4),
            "data_cache_available_ram_mb": (
                round(self.memory.available_bytes / (1024 * 1024), 4)
                if self.memory.available_bytes is not None
                else None
            ),
            "data_cache_total_ram_mb": (
                round(self.memory.total_bytes / (1024 * 1024), 4)
                if self.memory.total_bytes is not None
                else None
            ),
            "data_cache_storage_kind": self.storage_kind,
            "data_cache_preheat_recommended": self.preheat_recommended,
        }


class DataCachePolicy:
    """Choose Ultralytics cache mode from dataset size and local resources."""

    def __init__(
        self,
        config: DataCachePolicyConfig | None = None,
        evidence_store: EvidenceStore | None = None,
    ) -> None:
        self.config = config or DataCachePolicyConfig()
        self.evidence_store = evidence_store

    def decide(
        self,
        data_yaml: Path | str,
        command: CommandSpec,
        memory: MemorySnapshot | None = None,
        storage_kind: StorageKind | None = None,
    ) -> DataCacheDecision:
        """Return a cache decision without mutating the command."""
        if not self.config.enabled:
            return DataCacheDecision(
                selected_cache=_existing_cache(command) or "False",
                selected_workers=_existing_workers(command),
                applied=False,
                reason="Data cache policy disabled.",
            )
        explicit_cache = _existing_cache(command)
        if explicit_cache in {"ram", "disk"} and self.config.respect_explicit_cache:
            return DataCacheDecision(
                selected_cache=explicit_cache,
                selected_workers=_existing_workers(command),
                applied=False,
                reason=f"Respecting explicit cache={explicit_cache}.",
            )

        dataset_root, image_bytes = estimate_yolo_image_bytes(data_yaml)
        estimated_ram = int(image_bytes * self.config.decoded_ram_multiplier)
        snapshot = memory or current_memory_snapshot()
        kind = storage_kind or detect_storage_kind(dataset_root)
        warnings: list[str] = []
        current_workers = _existing_workers(command)
        selected_workers = current_workers

        if _can_use_ram_cache(estimated_ram, snapshot, self.config):
            selected_cache: CacheMode = "ram"
            reason = "Available memory is sufficient for RAM cache with safety margin."
        elif _can_use_disk_cache(kind, self.config):
            selected_cache = "disk"
            reason = f"RAM cache is not safe; storage kind={kind} supports disk cache."
        else:
            selected_cache = "False"
            selected_workers = _raised_workers(current_workers, self.config)
            reason = "RAM cache is not safe and fast disk cache is not confirmed."
            warnings.append("Preheat the dataset or move it to NVMe; cache remains disabled.")
        if snapshot.available_bytes is None:
            warnings.append("Available RAM could not be detected; avoided RAM cache.")

        return DataCacheDecision(
            selected_cache=selected_cache,
            selected_workers=selected_workers,
            dataset_size_bytes=image_bytes,
            estimated_ram_cache_bytes=estimated_ram,
            memory=snapshot,
            storage_kind=kind,
            applied=True,
            preheat_recommended=selected_cache == "False",
            reason=reason,
            warnings=warnings,
        )

    def apply(
        self,
        run_id: str,
        node: ExperimentNode,
        command: CommandSpec,
        data_yaml: Path | str,
        memory: MemorySnapshot | None = None,
        storage_kind: StorageKind | None = None,
    ) -> tuple[CommandSpec, DataCacheDecision]:
        """Apply the selected cache mode to a command and persist evidence."""
        decision = self.decide(data_yaml, command, memory=memory, storage_kind=storage_kind)
        if decision.applied:
            command = apply_cache_decision(command, decision)
        self._persist(run_id, node, decision)
        return command, decision

    def _persist(self, run_id: str, node: ExperimentNode, decision: DataCacheDecision) -> None:
        if self.evidence_store is None:
            return
        artifact_path = (
            self.evidence_store.create_run(run_id)
            / "artifacts"
            / f"{node.node_id}_data_cache_policy.json"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("w", encoding="utf-8") as file:
            json.dump(decision.model_dump(mode="json"), file, indent=2, sort_keys=True)
        self.evidence_store.log_artifact_manifest(
            run_id=run_id,
            name=f"{node.node_id}_data_cache_policy",
            artifact_path=artifact_path,
            producer_stage="data_cache_policy",
        )
        self.evidence_store.log_candidate_metrics(
            run_id=run_id,
            candidate_id=node.candidate_config.candidate_id,
            node_id=node.node_id,
            metrics=decision.to_metrics(),
            dataset_version=node.data_version,
            split="runtime",
            source="data_cache_policy",
            verified=True,
            validator="ultralytics_data_cache_policy",
            source_artifact=artifact_path,
        )


def apply_cache_decision(command: CommandSpec, decision: DataCacheDecision) -> CommandSpec:
    """Return a command copy with cache/workers policy applied."""
    updates: dict[str, str | int | bool] = {"cache": decision.selected_cache}
    if decision.selected_workers is not None:
        updates["workers"] = decision.selected_workers
    updated = _upsert_args(command, updates)
    metadata = {
        **command.metadata,
        "data_cache_policy_applied": decision.applied,
        "data_cache_selected_cache": decision.selected_cache,
    }
    if decision.selected_workers is not None:
        metadata["data_cache_selected_workers"] = decision.selected_workers
    return updated.model_copy(update={"metadata": metadata})


def estimate_yolo_image_bytes(data_yaml: Path | str) -> tuple[Path, int]:
    """Estimate dataset image bytes from YOLO data.yaml split entries."""
    path = Path(data_yaml)
    data = _read_yaml_mapping(path)
    root = _dataset_root(path, data)
    images = _collect_images(path, data, root)
    total = sum(image.stat().st_size for image in images if image.is_file())
    return root, total


def current_memory_snapshot() -> MemorySnapshot:
    """Return current total/available system memory without extra dependencies."""
    if os.name == "nt":
        snapshot = _windows_memory_snapshot()
        if snapshot is not None:
            return snapshot
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return MemorySnapshot(
                total_bytes=page_size * phys_pages,
                available_bytes=page_size * avail_pages,
                source="sysconf",
            )
        except (OSError, ValueError):
            pass
    return MemorySnapshot(source="unknown")


def detect_storage_kind(path: Path | str) -> StorageKind:
    """Best-effort storage kind detection; returns unknown when unavailable."""
    root = Path(path).resolve().anchor
    if os.name == "nt":
        detected = _windows_storage_kind(root)
        if detected != "unknown":
            return detected
    return "unknown"


def _can_use_ram_cache(
    estimated_ram_bytes: int,
    memory: MemorySnapshot,
    config: DataCachePolicyConfig,
) -> bool:
    if "ram" not in config.candidate_cache_modes:
        return False
    if estimated_ram_bytes <= 0:
        return False
    if memory.available_bytes is None:
        return False
    min_free = int(config.min_free_ram_after_cache_gb * 1024 * 1024 * 1024)
    if memory.available_bytes - estimated_ram_bytes < min_free:
        return False
    return estimated_ram_bytes <= int(memory.available_bytes * config.ram_cache_max_available_fraction)


def _can_use_disk_cache(storage_kind: StorageKind, config: DataCachePolicyConfig) -> bool:
    if "disk" not in config.candidate_cache_modes:
        return False
    if config.disk_cache_requires_nvme:
        return storage_kind == "nvme"
    return storage_kind in {"nvme", "ssd"}


def _raised_workers(current_workers: int | None, config: DataCachePolicyConfig) -> int | None:
    if current_workers is None:
        return None
    return min(config.max_workers, current_workers + config.worker_increment_on_no_cache)


def _existing_cache(command: CommandSpec) -> CacheMode | None:
    value = _arg_value(command, "cache")
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"ram", "true"}:
        return "ram"
    if lowered == "disk":
        return "disk"
    return "False"


def _existing_workers(command: CommandSpec) -> int | None:
    value = _arg_value(command, "workers")
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return data


def _dataset_root(data_path: Path, data: dict[str, Any]) -> Path:
    raw_root = data.get("path")
    if raw_root is None:
        return data_path.parent
    root = Path(str(raw_root))
    return root if root.is_absolute() else data_path.parent / root


def _collect_images(data_path: Path, data: dict[str, Any], dataset_root: Path) -> list[Path]:
    images: list[Path] = []
    for split in ("train", "val", "test"):
        for item in _as_list(data.get(split)):
            images.extend(_images_from_split_item(data_path, dataset_root, item))
    return sorted(dict.fromkeys(path.resolve() for path in images))


def _images_from_split_item(data_path: Path, dataset_root: Path, item: object) -> list[Path]:
    split_path = Path(str(item))
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    if split_path.is_dir():
        return [path for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        return _images_from_list_file(split_path, data_path, dataset_root)
    if split_path.is_file() and split_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [split_path]
    return []


def _images_from_list_file(list_path: Path, data_path: Path, dataset_root: Path) -> list[Path]:
    images: list[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        image_path = Path(text)
        if not image_path.is_absolute():
            image_path = (list_path.parent / image_path) if text.startswith(".") else (dataset_root / image_path)
        images.append(image_path)
    return images


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _upsert_args(command: CommandSpec, updates: dict[str, str | int | bool]) -> CommandSpec:
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


def _format_arg(value: str | int | bool) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_memory_snapshot() -> MemorySnapshot | None:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
    except AttributeError:
        return None
    if not ok:
        return None
    return MemorySnapshot(
        total_bytes=int(status.ullTotalPhys),
        available_bytes=int(status.ullAvailPhys),
        source="GlobalMemoryStatusEx",
    )


def _windows_storage_kind(root: str) -> StorageKind:
    if not shutil.which("powershell"):
        return "unknown"
    drive = root.rstrip("\\/").replace(":", "")
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "$drive = Get-Partition -DriveLetter "
            f"{drive} -ErrorAction SilentlyContinue | Get-Disk -ErrorAction SilentlyContinue; "
            "if ($drive) { $drive | Select-Object -First 1 -ExpandProperty BusType }"
        ),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = completed.stdout.strip().lower()
    if "nvme" in text:
        return "nvme"
    if "ssd" in text or "scm" in text:
        return "ssd"
    if "ata" in text or "sata" in text or "raid" in text or "sas" in text:
        return "ssd"
    return "unknown"
