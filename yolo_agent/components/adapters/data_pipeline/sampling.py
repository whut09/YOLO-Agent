"""Deterministic exposure sampling shared by independent data mechanisms."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Sampler


def bound_exposure(
    values: list[float],
    *,
    max_weight: float,
    max_ratio: float,
) -> tuple[list[float], dict[str, int | float]]:
    """Clip positive exposure values while preserving their relative signal."""
    if not values or any(value <= 0 for value in values):
        raise ValueError("exposure values must be non-empty and positive")
    minimum = min(values)
    cap = min(max_weight, minimum * max_ratio)
    final = [min(value, cap) for value in values]
    clipped = sum(raw != bounded for raw, bounded in zip(values, final))
    return final, {
        "clipped_count": clipped,
        "clipped_fraction": clipped / len(values),
        "raw_min": minimum,
        "raw_max": max(values),
        "final_min": min(final),
        "final_max": max(final),
        "max_weight": max_weight,
        "max_ratio": max_ratio,
    }


class DistributedExposureSampler(Sampler[int]):
    """Generate one global weighted stream and shard positions across DDP ranks."""

    def __init__(
        self,
        exposure: list[float],
        *,
        sample_count: int,
        seed: int,
        rank: int,
        world_size: int,
        dataset_manifest: str,
        adapter_hash: str,
        mechanism_id: str,
    ) -> None:
        if not exposure or any(value <= 0 for value in exposure):
            raise ValueError("exposure values must be non-empty and positive")
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if not 0 <= rank < world_size:
            raise ValueError(f"invalid distributed rank {rank}/{world_size}")
        self.exposure = torch.as_tensor(exposure, dtype=torch.double)
        self.sample_count = int(sample_count)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.dataset_manifest = dataset_manifest
        self.adapter_hash = adapter_hash
        self.mechanism_id = mechanism_id
        self.epoch = 0
        self.total_size = math.ceil(sample_count / world_size) * world_size
        self.num_samples = self.total_size // world_size

    def __iter__(self) -> Iterator[int]:
        stream = self.global_indices()
        return iter(stream[self.rank : self.total_size : self.world_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def global_indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.multinomial(
            self.exposure,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "data_exposure_sampler_state.v1",
            "mechanism_id": self.mechanism_id,
            "epoch": self.epoch,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "rank": self.rank,
            "world_size": self.world_size,
            "dataset_manifest": self.dataset_manifest,
            "adapter_hash": self.adapter_hash,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "mechanism_id": self.mechanism_id,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "dataset_manifest": self.dataset_manifest,
            "adapter_hash": self.adapter_hash,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"data sampler resume mismatch for {key}: "
                    f"expected={value!r} actual={state.get(key)!r}"
                )
        self.epoch = int(state.get("epoch", 0))


__all__ = ["DistributedExposureSampler", "bound_exposure"]
