"""Explicit, deterministic image preprocessing for data-side routes."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import Dataset


class PreprocessingConfig(BaseModel):
    """Configuration for one paper-authorized image preprocessing operation."""

    model_config = ConfigDict(extra="forbid")

    normalize: bool = True
    mean: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    std: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    imgsz: int = 640

    def model_post_init(self, __context: object) -> None:
        if self.imgsz != 640:
            raise ValueError("preprocessing adapters require fixed imgsz=640")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("preprocessing mean and std must have three channels")
        if any(value <= 0.0 for value in self.std):
            raise ValueError("preprocessing standard deviations must be positive")


def preprocess_image(image: torch.Tensor, config: PreprocessingConfig) -> torch.Tensor:
    """Return a float image with explicit normalization and no geometry change."""

    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise ValueError("image preprocessing requires a tensor with shape (C,H,W)")
    if image.shape[0] != 3:
        raise ValueError("image preprocessing currently supports three channels")
    output = image.to(dtype=torch.float32)
    if not config.normalize:
        return output
    if image.dtype == torch.uint8 or float(output.max()) > 1.0:
        output = output / 255.0
    mean = torch.tensor(config.mean, dtype=output.dtype, device=output.device)
    std = torch.tensor(config.std, dtype=output.dtype, device=output.device)
    return (output - mean[:, None, None]) / std[:, None, None]


def preprocess_sample(
    sample: dict[str, Any],
    config: PreprocessingConfig,
) -> dict[str, Any]:
    """Apply preprocessing to ``img`` while preserving labels unchanged."""

    image = sample.get("img")
    if not isinstance(image, torch.Tensor):
        raise ValueError("image preprocessing requires sample['img'] tensor")
    output = dict(sample)
    output["img"] = preprocess_image(image, config)
    return output


class PreprocessingDataset(Dataset[Any]):
    """Apply deterministic image preprocessing while preserving annotations."""

    def __init__(self, dataset: Any, config: PreprocessingConfig) -> None:
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
            raise ValueError("preprocessing dataset requires mapping samples")
        return preprocess_sample(sample, self.config)


__all__ = [
    "PreprocessingConfig",
    "PreprocessingDataset",
    "preprocess_image",
    "preprocess_sample",
]
