"""Model-bound backend for isolated Ultralytics inference policies."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from yolo_agent.components.adapters.inference.policy import InferencePolicyProtocol
from yolo_agent.components.adapters.inference.postprocess import (
    apply_class_thresholds,
    calibrate_confidence,
    merge_predictions,
)


class InferenceImage(BaseModel):
    image_id: int
    path: Path


class BackendResult(BaseModel):
    predictions: list[dict[str, Any]]
    latency_ms: float
    throughput: float
    peak_vram_mb: float = 0.0
    merge_statistics: dict[str, int | float | str | bool]


class UltralyticsInferenceBackend:
    """Execute policies through an injected or installed Ultralytics model."""

    def __init__(self, model_factory: Callable[[str], Any] | None = None) -> None:
        self._model_factory = model_factory

    def run(
        self,
        images: list[InferenceImage],
        protocol: InferencePolicyProtocol,
        *,
        category_ids: list[int],
    ) -> BackendResult:
        config = protocol.config
        if not config.model_path:
            raise ValueError("inference policy execution requires model_path")
        model = self._load_model(config.model_path)
        _reset_peak_vram(config.device)
        started = time.perf_counter()
        predictions: list[dict[str, Any]] = []
        for image in images:
            if config.kind == "tiled_multi_scale":
                predictions.extend(
                    self._predict_tiles(model, image, protocol, category_ids)
                )
            elif config.kind in {"test_time_augmentation", "merge_policy"}:
                predictions.extend(
                    self._predict_tta(model, image, protocol, category_ids)
                )
            else:
                predictions.extend(
                    self._predict_view(
                        model,
                        source=str(image.path),
                        image_id=image.image_id,
                        imgsz=640,
                        device=config.device,
                        confidence_threshold=config.confidence_threshold,
                        category_ids=category_ids,
                    )
                )
        if config.kind == "confidence_calibration":
            predictions = calibrate_confidence(predictions, config.temperature)
        if config.kind == "class_aware_thresholding":
            predictions = apply_class_thresholds(
                predictions,
                config.class_thresholds,
                default_threshold=config.confidence_threshold,
            )
        predictions, merge_statistics = merge_predictions(
            predictions,
            policy=config.merge_policy,
            iou_threshold=config.merge_iou_threshold,
            max_detections=config.max_detections,
        )
        elapsed = time.perf_counter() - started
        image_count = max(len(images), 1)
        return BackendResult(
            predictions=predictions,
            latency_ms=elapsed * 1000.0 / image_count,
            throughput=len(images) / elapsed if elapsed > 0 else 0.0,
            peak_vram_mb=_peak_vram(config.device),
            merge_statistics=merge_statistics,
        )

    def _predict_tta(
        self,
        model: Any,
        image: InferenceImage,
        protocol: InferencePolicyProtocol,
        category_ids: list[int],
    ) -> list[dict[str, Any]]:
        config = protocol.config
        output: list[dict[str, Any]] = []
        for scale in config.scales:
            output.extend(
                self._predict_view(
                    model,
                    source=str(image.path),
                    image_id=image.image_id,
                    imgsz=max(32, int(round(640 * scale / 32.0)) * 32),
                    device=config.device,
                    confidence_threshold=config.confidence_threshold,
                    category_ids=category_ids,
                )
            )
        if config.horizontal_flip:
            from PIL import Image, ImageOps

            with Image.open(image.path) as source_image:
                width = source_image.width
                flipped = ImageOps.mirror(source_image.convert("RGB"))
                records = self._predict_view(
                    model,
                    source=flipped,
                    image_id=image.image_id,
                    imgsz=640,
                    device=config.device,
                    confidence_threshold=config.confidence_threshold,
                    category_ids=category_ids,
                )
            for item in records:
                x, y, box_width, height = item["bbox"]
                item["bbox"] = [width - x - box_width, y, box_width, height]
            output.extend(records)
        return output

    def _predict_tiles(
        self,
        model: Any,
        image: InferenceImage,
        protocol: InferencePolicyProtocol,
        category_ids: list[int],
    ) -> list[dict[str, Any]]:
        from PIL import Image

        config = protocol.config
        output: list[dict[str, Any]] = []
        with Image.open(image.path) as source_image:
            rgb = source_image.convert("RGB")
            for tile_size in config.tile_sizes:
                for left, top, right, bottom in _tile_windows(
                    rgb.width, rgb.height, tile_size, config.overlap_ratio
                ):
                    tile = rgb.crop((left, top, right, bottom))
                    records = self._predict_view(
                        model,
                        source=tile,
                        image_id=image.image_id,
                        imgsz=640,
                        device=config.device,
                        confidence_threshold=config.confidence_threshold,
                        category_ids=category_ids,
                    )
                    for item in records:
                        item["bbox"][0] += left
                        item["bbox"][1] += top
                    output.extend(records)
        return output

    @staticmethod
    def _predict_view(
        model: Any,
        *,
        source: Any,
        image_id: int,
        imgsz: int,
        device: str,
        confidence_threshold: float,
        category_ids: list[int],
    ) -> list[dict[str, Any]]:
        results = model.predict(
            source=source,
            imgsz=imgsz,
            device=device,
            conf=confidence_threshold,
            verbose=False,
        )
        output: list[dict[str, Any]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = _tolist(getattr(boxes, "xyxy", []))
            scores = _tolist(getattr(boxes, "conf", []))
            classes = _tolist(getattr(boxes, "cls", []))
            for coordinates, score, class_index in zip(xyxy, scores, classes, strict=True):
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                index = int(class_index)
                category_id = category_ids[index] if 0 <= index < len(category_ids) else index + 1
                output.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )
        return output

    def _load_model(self, model_path: str) -> Any:
        if self._model_factory is not None:
            return self._model_factory(model_path)
        from ultralytics import YOLO

        return YOLO(model_path)


def _tile_windows(
    width: int, height: int, tile_size: int, overlap_ratio: float
) -> list[tuple[int, int, int, int]]:
    stride = max(1, int(round(tile_size * (1.0 - overlap_ratio))))
    xs = _axis_starts(width, tile_size, stride)
    ys = _axis_starts(height, tile_size, stride)
    return [
        (left, top, min(left + tile_size, width), min(top + tile_size, height))
        for top in ys
        for left in xs
    ]


def _axis_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    values = list(range(0, max(length - size + 1, 1), stride))
    last = length - size
    if values[-1] != last:
        values.append(last)
    return values


def _tolist(value: Any) -> list[Any]:
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    converted = cpu.tolist() if callable(getattr(cpu, "tolist", None)) else cpu
    return list(converted)


def _reset_peak_vram(device: str) -> None:
    try:
        import torch

        if str(device).lower() != "cpu" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        return


def _peak_vram(device: str) -> float:
    try:
        import torch

        if str(device).lower() != "cpu" and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except (ImportError, RuntimeError):
        pass
    return 0.0


__all__ = [
    "BackendResult",
    "InferenceImage",
    "UltralyticsInferenceBackend",
]
