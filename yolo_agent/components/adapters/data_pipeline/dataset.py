"""Spawn-safe dataset wrapper for train-only data transformations."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import Dataset

from yolo_agent.components.adapters.data_pipeline.transforms import (
    DataTransformConfig,
    blend_multi_image_samples,
    copy_paste_sample,
    crop_sample,
    zero_effect_sample,
)


class DataPipelineDataset(Dataset[Any]):
    """Apply exactly one configured mechanism with deterministic epoch state."""

    def __init__(self, dataset: Any, config: DataTransformConfig) -> None:
        self.dataset = dataset
        self.config = config
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.transform_count = 0

    def __getattr__(self, name: str) -> Any:
        """Preserve the Ultralytics dataset surface required by its loader."""
        if name in {"dataset", "config", "_epoch", "transform_count"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        native = self.dataset[index]
        if not isinstance(native, dict):
            raise ValueError("data pipeline dataset requires mapping samples")
        if not self._active(index):
            return zero_effect_sample(native)
        mechanism = self.config.mechanism
        generator = self._random(index)
        if mechanism == "copy_paste_rare_classes":
            donor_index = self._donor_index(index, generator)
            output = copy_paste_sample(
                native,
                self.dataset[donor_index],
                rare_class_ids=set(self.config.rare_class_ids),
            )
        elif mechanism in {"scale_aware_crop", "object_centric_crop"}:
            center = self._crop_center(native, generator)
            if center is None:
                return zero_effect_sample(native)
            center_x, center_y = center
            output = crop_sample(
                native,
                center_x=center_x,
                center_y=center_y,
                scale=self.config.crop_scale,
            )
        elif mechanism == "multi_image_sampling_schedule":
            indices = [index]
            while len(indices) < self.config.multi_image_count:
                candidate = generator.randrange(len(self.dataset))
                if candidate not in indices or len(self.dataset) == 1:
                    indices.append(candidate)
            output = blend_multi_image_samples([self.dataset[item] for item in indices])
        else:  # pragma: no cover - validated literal
            raise AssertionError(f"unsupported transform mechanism: {mechanism}")
        self.transform_count += 1
        return output

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "data_pipeline_dataset_state.v1",
            "mechanism_id": self.config.mechanism,
            "seed": self.config.seed,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key, expected in (
            ("mechanism_id", self.config.mechanism),
            ("seed", self.config.seed),
        ):
            if state.get(key) != expected:
                raise ValueError(
                    f"data pipeline resume mismatch for {key}: "
                    f"expected={expected!r} actual={state.get(key)!r}"
                )
        self.set_epoch(int(state.get("epoch", 0)))

    def _active(self, index: int) -> bool:
        if self.epoch < self.config.active_epoch_start:
            return False
        if (
            self.config.active_epoch_end is not None
            and self.epoch > self.config.active_epoch_end
        ):
            return False
        return self._random(index).random() < self.config.probability

    def _random(self, index: int) -> random.Random:
        return random.Random(self.config.seed + self.epoch * 1_000_003 + index)

    def _donor_index(self, index: int, generator: random.Random) -> int:
        candidates = [item for item in range(len(self.dataset)) if item != index]
        if not candidates:
            raise ValueError("copy-paste requires a separate donor sample")
        generator.shuffle(candidates)
        for candidate in candidates:
            donor = self.dataset[candidate]
            classes = donor.get("cls") if isinstance(donor, dict) else None
            if isinstance(classes, torch.Tensor) and set(
                int(value) for value in classes.reshape(-1).tolist()
            ).intersection(self.config.rare_class_ids):
                return candidate
        raise ValueError("copy-paste dataset has no rare-class donor")

    def _crop_center(
        self,
        sample: dict[str, Any],
        generator: random.Random,
    ) -> tuple[float, float] | None:
        boxes = sample.get("bboxes")
        if not isinstance(boxes, torch.Tensor) or not len(boxes):
            raise ValueError("object-centric crop requires at least one object")
        areas = boxes[:, 2] * boxes[:, 3]
        small = torch.where(areas <= self.config.small_area_threshold)[0]
        if self.config.mechanism == "scale_aware_crop" and not len(small):
            return None
        choices = small.tolist() or list(range(len(boxes)))
        selected = boxes[generator.choice(choices)]
        return float(selected[0]), float(selected[1])


__all__ = ["DataPipelineDataset"]
