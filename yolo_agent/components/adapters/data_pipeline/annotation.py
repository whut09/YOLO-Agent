"""Train-side annotation validation and filtering primitives.

These helpers intentionally operate on detection tensors only.  They are
usable by a paper route after that route has supplied explicit evidence for
the filtering rule; the helper itself never infers a paper method from a
title or a component alias.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import Dataset


class AnnotationFilterConfig(BaseModel):
    """Bounds used by an explicit annotation-filter route."""

    model_config = ConfigDict(extra="forbid")

    min_width: float = Field(default=0.002, ge=0.0, le=1.0)
    min_height: float = Field(default=0.002, ge=0.0, le=1.0)
    min_area: float = Field(default=0.000004, ge=0.0, le=1.0)
    max_area: float = Field(default=0.95, gt=0.0, le=1.0)
    max_aspect_ratio: float = Field(default=12.0, gt=0.0)
    allowed_class_ids: list[int] = Field(default_factory=list)
    clip_boxes: bool = True
    imgsz: int = 640

    def model_post_init(self, __context: object) -> None:
        if self.imgsz != 640:
            raise ValueError("annotation adapters require fixed imgsz=640")
        if self.min_area > self.max_area:
            raise ValueError("annotation min_area cannot exceed max_area")


class AnnotationFilterResult(BaseModel):
    """Auditable result of filtering one detection sample."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample: dict[str, Any]
    kept_indices: list[int] = Field(default_factory=list)
    removed_indices: list[int] = Field(default_factory=list)
    removal_reasons: dict[str, list[int]] = Field(default_factory=dict)
    quality_scores: list[float] = Field(default_factory=list)


def filter_detection_sample(
    sample: dict[str, Any],
    config: AnnotationFilterConfig,
) -> AnnotationFilterResult:
    """Filter invalid or explicitly out-of-policy YOLO detection boxes.

    Box geometry is clipped before quality checks when ``clip_boxes`` is true.
    A box and its class are always kept or removed together, and an empty
    result retains the canonical ``(0, 4)``/``(0, 1)`` tensor shapes.
    """

    image, boxes, classes = _detection_tensors(sample)
    del image
    clipped = _clip_xywh(boxes, enabled=config.clip_boxes)

    kept: list[int] = []
    removed: list[int] = []
    reasons: dict[str, list[int]] = {}
    scores: list[float] = []
    output_boxes: list[torch.Tensor] = []
    output_classes: list[torch.Tensor] = []
    allowed = set(config.allowed_class_ids)

    for index, (box, class_value) in enumerate(zip(clipped, classes, strict=True)):
        reason = _rejection_reason(box, int(class_value.item()), config, allowed)
        score = annotation_quality_score(box, config=config)
        scores.append(score)
        if reason is not None:
            removed.append(index)
            reasons.setdefault(reason, []).append(index)
            continue
        kept.append(index)
        output_boxes.append(box)
        output_classes.append(class_value.reshape(1))

    filtered = dict(sample)
    filtered["bboxes"] = (
        torch.stack(output_boxes)
        if output_boxes
        else clipped.new_empty((0, 4))
    )
    filtered["cls"] = (
        torch.stack(output_classes).reshape(-1, 1)
        if output_classes
        else classes.new_empty((0, 1))
    )
    batch_idx = sample.get("batch_idx")
    if isinstance(batch_idx, torch.Tensor) and len(batch_idx) == len(boxes):
        filtered["batch_idx"] = batch_idx[kept].clone()
    else:
        filtered["batch_idx"] = torch.zeros(
            len(kept), dtype=torch.int64, device=clipped.device
        )
    return AnnotationFilterResult(
        sample=filtered,
        kept_indices=kept,
        removed_indices=removed,
        removal_reasons=reasons,
        quality_scores=scores,
    )


class AnnotationFilterDataset(Dataset[Any]):
    """Apply one explicit annotation policy to train samples only."""

    def __init__(self, dataset: Any, config: AnnotationFilterConfig) -> None:
        self.dataset = dataset
        self.config = config

    def __getattr__(self, name: str) -> Any:
        if name in {"dataset", "config"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        if not isinstance(sample, dict):
            raise ValueError("annotation filter dataset requires mapping samples")
        return filter_detection_sample(sample, self.config).sample


def _clip_xywh(boxes: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    """Clip normalized ``xywh`` boxes by corners, preserving valid geometry."""

    output = boxes.detach().clone().to(dtype=torch.float32)
    if not enabled or not len(output):
        return output
    x1 = output[:, 0] - output[:, 2] / 2.0
    y1 = output[:, 1] - output[:, 3] / 2.0
    x2 = output[:, 0] + output[:, 2] / 2.0
    y2 = output[:, 1] + output[:, 3] / 2.0
    x1, y1, x2, y2 = (
        value.clamp(0.0, 1.0) for value in (x1, y1, x2, y2)
    )
    return torch.stack(
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1),
        dim=1,
    )


def annotation_quality_score(
    box: torch.Tensor,
    *,
    config: AnnotationFilterConfig | None = None,
) -> float:
    """Return a bounded geometry quality score for one normalized box."""

    active = config or AnnotationFilterConfig()
    values = box.detach().to(dtype=torch.float32).reshape(-1)
    if values.numel() != 4 or not bool(torch.isfinite(values).all()):
        return 0.0
    width = float(values[2].clamp(0.0, 1.0))
    height = float(values[3].clamp(0.0, 1.0))
    area = width * height
    if width <= 0.0 or height <= 0.0:
        return 0.0
    size_score = min(
        1.0,
        width / max(active.min_width, 1e-12),
        height / max(active.min_height, 1e-12),
    )
    area_score = min(
        1.0,
        area / max(active.min_area, 1e-12),
        active.max_area / max(area, 1e-12),
    )
    aspect = max(width / max(height, 1e-12), height / max(width, 1e-12))
    aspect_score = min(1.0, active.max_aspect_ratio / max(aspect, 1e-12))
    return float(max(0.0, min(1.0, size_score * area_score * aspect_score)))


def _rejection_reason(
    box: torch.Tensor,
    class_id: int,
    config: AnnotationFilterConfig,
    allowed: set[int],
) -> str | None:
    if not bool(torch.isfinite(box).all()):
        return "non_finite_geometry"
    if allowed and class_id not in allowed:
        return "class_not_allowed"
    x_center, y_center, width, height = [float(value) for value in box]
    if not config.clip_boxes:
        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
        if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
            return "outside_normalized_range"
    if width <= config.min_width or height <= config.min_height:
        return "box_too_small"
    area = width * height
    if area < config.min_area:
        return "area_too_small"
    if area > config.max_area:
        return "area_too_large"
    aspect = max(width / max(height, 1e-12), height / max(width, 1e-12))
    if aspect > config.max_aspect_ratio:
        return "aspect_ratio_too_large"
    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
        return "center_outside_normalized_range"
    return None


def _detection_tensors(
    sample: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unsupported = {
        key
        for key in ("masks", "segments", "obb", "keypoints")
        if sample.get(key) is not None
    }
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(
            "data-side annotation routes currently support detect boxes only; "
            f"unsupported annotation fields: {names}"
        )
    image = sample.get("img")
    boxes = sample.get("bboxes")
    classes = sample.get("cls")
    if not all(isinstance(item, torch.Tensor) for item in (image, boxes, classes)):
        raise ValueError("annotation filtering requires tensor img/bboxes/cls")
    if image.ndim != 3 or boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("annotation filtering requires image (C,H,W) and boxes (N,4)")
    flat_classes = classes.reshape(-1)
    if len(flat_classes) != len(boxes):
        raise ValueError("annotation classes must align one-to-one with boxes")
    return image, boxes, flat_classes


__all__ = [
    "AnnotationFilterConfig",
    "AnnotationFilterResult",
    "AnnotationFilterDataset",
    "annotation_quality_score",
    "filter_detection_sample",
]
