"""Tensor geometry for independent train-only data pipeline mechanisms."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as functional
from pydantic import BaseModel, ConfigDict, Field


TransformMechanism = Literal[
    "copy_paste_rare_classes",
    "scale_aware_crop",
    "object_centric_crop",
    "multi_image_sampling_schedule",
]


class DataTransformConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mechanism: TransformMechanism
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int = Field(default=0, ge=0)
    rare_class_ids: list[int] = Field(default_factory=list)
    crop_scale: float = Field(default=0.75, gt=0.0, le=1.0)
    small_area_threshold: float = Field(default=0.01, gt=0.0, lt=1.0)
    multi_image_count: int = Field(default=2, ge=2, le=4)
    active_epoch_start: int = Field(default=0, ge=0)
    active_epoch_end: int | None = Field(default=None, ge=0)
    imgsz: int = 640

    def model_post_init(self, __context: object) -> None:
        if self.imgsz != 640:
            raise ValueError("data transform adapters require fixed imgsz=640")
        if (
            self.active_epoch_end is not None
            and self.active_epoch_end < self.active_epoch_start
        ):
            raise ValueError("active epoch end precedes start")


def zero_effect_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Clone tensors so exact native equivalence can be asserted safely."""
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }


def crop_sample(
    sample: dict[str, Any],
    *,
    center_x: float,
    center_y: float,
    scale: float,
) -> dict[str, Any]:
    image, boxes, classes = _sample_tensors(sample)
    height, width = image.shape[-2:]
    crop_width = max(1, round(width * scale))
    crop_height = max(1, round(height * scale))
    left = min(max(round(center_x * width - crop_width / 2), 0), width - crop_width)
    top = min(max(round(center_y * height - crop_height / 2), 0), height - crop_height)
    cropped = image[..., top : top + crop_height, left : left + crop_width]
    resized = functional.interpolate(
        cropped.unsqueeze(0).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).to(image.dtype)
    centers_x = boxes[:, 0] * width
    centers_y = boxes[:, 1] * height
    keep = (
        (centers_x >= left)
        & (centers_x <= left + crop_width)
        & (centers_y >= top)
        & (centers_y <= top + crop_height)
    )
    adjusted = boxes[keep].clone()
    if adjusted.numel():
        adjusted[:, 0] = (adjusted[:, 0] * width - left) / crop_width
        adjusted[:, 1] = (adjusted[:, 1] * height - top) / crop_height
        adjusted[:, 2] = (adjusted[:, 2] / scale).clamp(max=1.0)
        adjusted[:, 3] = (adjusted[:, 3] / scale).clamp(max=1.0)
    return _replace_sample(sample, resized, adjusted, classes[keep])


def copy_paste_sample(
    target: dict[str, Any],
    donor: dict[str, Any],
    *,
    rare_class_ids: set[int],
) -> dict[str, Any]:
    target_image, target_boxes, target_classes = _sample_tensors(target)
    donor_image, donor_boxes, donor_classes = _sample_tensors(donor)
    if target_image.shape != donor_image.shape:
        raise ValueError("copy-paste requires matching transformed image shapes")
    donor_ids = donor_classes.reshape(-1).to(torch.int64)
    selected = torch.tensor(
        [int(value) in rare_class_ids for value in donor_ids.tolist()],
        dtype=torch.bool,
        device=donor_boxes.device,
    )
    if not bool(selected.any()):
        raise ValueError("copy-paste donor contains no requested rare class")
    pasted = target_image.clone()
    height, width = pasted.shape[-2:]
    for box in donor_boxes[selected]:
        cx, cy, bw, bh = box.tolist()
        left = max(0, int((cx - bw / 2) * width))
        right = min(width, max(left + 1, int((cx + bw / 2) * width)))
        top = max(0, int((cy - bh / 2) * height))
        bottom = min(height, max(top + 1, int((cy + bh / 2) * height)))
        pasted[..., top:bottom, left:right] = donor_image[
            ..., top:bottom, left:right
        ]
    return _replace_sample(
        target,
        pasted,
        torch.cat([target_boxes, donor_boxes[selected]], dim=0),
        torch.cat([target_classes, donor_classes[selected]], dim=0),
    )


def blend_multi_image_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 2:
        raise ValueError("multi-image sampling requires at least two samples")
    unpacked = [_sample_tensors(item) for item in samples]
    images = [item[0] for item in unpacked]
    if len({tuple(image.shape) for image in images}) != 1:
        raise ValueError("multi-image sampling requires matching image shapes")
    blended = torch.stack([image.float() for image in images]).mean(dim=0)
    blended = blended.to(images[0].dtype)
    return _replace_sample(
        samples[0],
        blended,
        torch.cat([item[1] for item in unpacked], dim=0),
        torch.cat([item[2] for item in unpacked], dim=0),
    )


def _sample_tensors(
    sample: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = sample.get("img")
    boxes = sample.get("bboxes")
    classes = sample.get("cls")
    if not all(isinstance(item, torch.Tensor) for item in (image, boxes, classes)):
        raise ValueError("data transform sample requires tensor img/bboxes/cls")
    if image.ndim != 3 or boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("data transform sample has unsupported tensor shapes")
    return image, boxes, classes


def _replace_sample(
    sample: dict[str, Any],
    image: torch.Tensor,
    boxes: torch.Tensor,
    classes: torch.Tensor,
) -> dict[str, Any]:
    output = dict(sample)
    output["img"] = image
    output["bboxes"] = boxes
    output["cls"] = classes.reshape(-1, 1)
    output["batch_idx"] = torch.zeros(
        len(boxes),
        dtype=torch.int64,
        device=boxes.device,
    )
    return output


__all__ = [
    "DataTransformConfig",
    "TransformMechanism",
    "blend_multi_image_samples",
    "copy_paste_sample",
    "crop_sample",
    "zero_effect_sample",
]
