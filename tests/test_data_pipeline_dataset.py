from __future__ import annotations

import pickle

import pytest
import torch
from torch.utils.data import Dataset

from yolo_agent.components.adapters.data_pipeline.dataset import DataPipelineDataset
from yolo_agent.components.adapters.data_pipeline.transforms import DataTransformConfig


class TransformDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.samples = [
            _sample(0, 1, [0.2, 0.2, 0.2, 0.2]),
            _sample(100, 3, [0.7, 0.7, 0.05, 0.05]),
            _sample(200, 2, [0.5, 0.5, 0.3, 0.3]),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.samples[index].items()}


def _sample(value: int, class_id: int, box: list[float]) -> dict[str, torch.Tensor]:
    return {
        "img": torch.full((3, 16, 16), value, dtype=torch.uint8),
        "bboxes": torch.tensor([box], dtype=torch.float32),
        "cls": torch.tensor([[class_id]], dtype=torch.float32),
        "batch_idx": torch.tensor([0]),
    }


@pytest.mark.parametrize(
    ("mechanism", "options"),
    [
        ("copy_paste_rare_classes", {"rare_class_ids": [3]}),
        ("scale_aware_crop", {"crop_scale": 0.75}),
        ("object_centric_crop", {"crop_scale": 0.75}),
        ("multi_image_sampling_schedule", {"multi_image_count": 2}),
    ],
)
def test_dataset_mechanisms_act_on_training_samples(
    mechanism: str,
    options: dict[str, object],
) -> None:
    wrapped = DataPipelineDataset(
        TransformDataset(),
        DataTransformConfig(mechanism=mechanism, **options),  # type: ignore[arg-type]
    )

    output = wrapped[0]

    assert wrapped.transform_count == 1
    assert output["img"].shape == torch.Size([3, 16, 16])


def test_zero_probability_is_exact_native_equivalence() -> None:
    native = TransformDataset()
    wrapped = DataPipelineDataset(
        native,
        DataTransformConfig(mechanism="scale_aware_crop", probability=0),
    )

    output = wrapped[0]

    assert all(torch.equal(output[key], native[0][key]) for key in output)
    assert wrapped.transform_count == 0


def test_dataset_is_spawn_picklable_and_resume_reproducible() -> None:
    config = DataTransformConfig(
        mechanism="multi_image_sampling_schedule",
        seed=17,
        multi_image_count=2,
    )
    wrapped = DataPipelineDataset(TransformDataset(), config)
    wrapped.set_epoch(3)
    restored = pickle.loads(pickle.dumps(wrapped))

    assert torch.equal(wrapped[0]["img"], restored[0]["img"])
    resumed = DataPipelineDataset(TransformDataset(), config)
    resumed.load_state_dict(wrapped.state_dict())
    assert torch.equal(wrapped[1]["img"], resumed[1]["img"])
