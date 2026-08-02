from __future__ import annotations

import pytest

from yolo_agent.components.adapters.data_pipeline import (
    DistributedExposureSampler,
    bound_exposure,
)


def _sampler(rank: int) -> DistributedExposureSampler:
    return DistributedExposureSampler(
        [1.0, 2.0, 3.0, 4.0],
        sample_count=7,
        seed=23,
        rank=rank,
        world_size=2,
        dataset_manifest="dataset-v1",
        adapter_hash="adapter-v1",
        mechanism_id="class_balanced_sampling",
    )


def test_exposure_clipping_is_bounded() -> None:
    final, statistics = bound_exposure(
        [1.0, 2.0, 8.0],
        max_weight=4.0,
        max_ratio=3.0,
    )

    assert final == [1.0, 2.0, 3.0]
    assert statistics["clipped_count"] == 1


def test_global_stream_is_deterministic_and_ddp_shards_positions() -> None:
    rank0 = _sampler(0)
    rank1 = _sampler(1)
    stream = rank0.global_indices()

    assert rank1.global_indices() == stream
    assert list(rank0) == stream[0::2]
    assert list(rank1) == stream[1::2]
    rank0.set_epoch(2)
    rank1.set_epoch(2)
    assert rank0.global_indices() == rank1.global_indices()
    assert rank0.global_indices() != stream


def test_resume_rejects_cross_mechanism_state() -> None:
    sampler = _sampler(0)
    state = sampler.state_dict()
    state["mechanism_id"] = "repeat_factor_sampling"

    with pytest.raises(ValueError, match="mechanism_id"):
        sampler.load_state_dict(state)
