"""Ultralytics dataset and dataloader runtime helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from yolo_agent.components.adapters.data_pipeline.contracts import DataSampleRecord


def records_from_yolo_dataset(dataset: Any) -> list[DataSampleRecord]:
    labels = getattr(dataset, "labels", None)
    if labels is None and callable(getattr(dataset, "get_labels", None)):
        labels = dataset.get_labels()
    if not isinstance(labels, list) or len(labels) != len(dataset):
        raise ValueError("Ultralytics train dataset must expose one labels entry per image")
    image_files = list(getattr(dataset, "im_files", []))
    hard_indices = {int(value) for value in getattr(dataset, "hard_negative_indices", [])}
    fn_scores = {
        int(key): float(value)
        for key, value in dict(getattr(dataset, "false_negative_scores", {})).items()
    }
    records: list[DataSampleRecord] = []
    for index, label in enumerate(labels):
        if not isinstance(label, dict):
            raise ValueError(f"dataset label {index} is not a mapping")
        if label.get("normalized") is False:
            raise ValueError("data adapters require normalized YOLO bboxes")
        if str(label.get("bbox_format", "xywh")).lower() != "xywh":
            raise ValueError("data adapters require xywh bboxes")
        boxes = _nested_values(label.get("bboxes", []))
        classes = [int(value) for value in _flat_values(label.get("cls", []))]
        areas = [float(box[2]) * float(box[3]) for box in boxes if len(box) >= 4]
        image_path = str(
            label.get("im_file")
            or (image_files[index] if index < len(image_files) else f"image-{index}")
        )
        records.append(DataSampleRecord(
            image_path=image_path,
            normalized_areas=areas,
            class_ids=classes,
            is_hard_negative=(
                index in hard_indices or bool(label.get("is_hard_negative", False))
            ),
            false_negative_score=float(
                label.get("false_negative_score", fn_scores.get(index, 0.0))
            ),
        ))
    return records


def dataset_manifest_hash(dataset: Any, records: list[DataSampleRecord]) -> str:
    declared = getattr(dataset, "manifest_hash", None) or getattr(
        dataset, "dataset_manifest", None
    )
    if declared:
        return str(declared)
    payload = [item.model_dump(mode="json") for item in records]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rebuild_dataloader(dataloader: Any, sampler: Any) -> Any:
    loader_type = type(dataloader)
    workers = int(getattr(dataloader, "num_workers", 0))
    kwargs: dict[str, Any] = {
        "dataset": dataloader.dataset,
        "batch_size": dataloader.batch_size,
        "shuffle": False,
        "sampler": sampler,
        "num_workers": workers,
        "collate_fn": dataloader.collate_fn,
        "pin_memory": bool(getattr(dataloader, "pin_memory", False)),
        "drop_last": bool(getattr(dataloader, "drop_last", False)),
        "timeout": float(getattr(dataloader, "timeout", 0)),
        "worker_init_fn": getattr(dataloader, "worker_init_fn", None),
        "generator": getattr(dataloader, "generator", None),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(
            getattr(dataloader, "persistent_workers", False)
        )
        prefetch_factor = getattr(dataloader, "prefetch_factor", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    close = getattr(dataloader, "close", None)
    if callable(close):
        close()
    try:
        return loader_type(**kwargs)
    except TypeError:
        return DataLoader(**kwargs)


def world_size(rank: int) -> int:
    if rank < 0:
        return 1
    return max(int(os.environ.get("WORLD_SIZE", "1")), rank + 1)


def write_json_atomic(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"runtime state must contain a mapping: {path}")
    return value


def _flat_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    return [float(item[0] if isinstance(item, list) else item) for item in value]


def _nested_values(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    return [[float(item) for item in row] for row in value if isinstance(row, list)]


__all__ = [
    "dataset_manifest_hash",
    "read_json",
    "rebuild_dataloader",
    "records_from_yolo_dataset",
    "world_size",
    "write_json_atomic",
]
