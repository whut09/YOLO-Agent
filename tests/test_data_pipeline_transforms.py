from __future__ import annotations

import torch

from yolo_agent.components.adapters.data_pipeline.transforms import (
    blend_multi_image_samples,
    copy_paste_sample,
    crop_sample,
    zero_effect_sample,
)


def _sample(value: int, class_id: int, box: list[float]) -> dict[str, torch.Tensor]:
    return {
        "img": torch.full((3, 16, 16), value, dtype=torch.uint8),
        "bboxes": torch.tensor([box], dtype=torch.float32),
        "cls": torch.tensor([[class_id]], dtype=torch.float32),
        "batch_idx": torch.tensor([0]),
    }


def test_zero_effect_is_tensor_equivalent_to_native_sample() -> None:
    native = _sample(20, 1, [0.5, 0.5, 0.2, 0.2])
    output = zero_effect_sample(native)

    assert all(torch.equal(native[key], output[key]) for key in native)
    assert output["img"] is not native["img"]


def test_crop_updates_image_and_normalized_boxes_together() -> None:
    output = crop_sample(
        _sample(20, 1, [0.5, 0.5, 0.2, 0.2]),
        center_x=0.5,
        center_y=0.5,
        scale=0.5,
    )

    assert output["img"].shape == torch.Size([3, 16, 16])
    assert torch.allclose(output["bboxes"], torch.tensor([[0.5, 0.5, 0.4, 0.4]]))


def test_copy_paste_adds_only_requested_rare_classes() -> None:
    target = _sample(0, 1, [0.2, 0.2, 0.2, 0.2])
    donor = _sample(255, 3, [0.7, 0.7, 0.2, 0.2])

    output = copy_paste_sample(target, donor, rare_class_ids={3})

    assert output["bboxes"].shape[0] == 2
    assert output["cls"].reshape(-1).tolist() == [1.0, 3.0]
    assert output["img"].max() == 255


def test_multi_image_blend_combines_exposure_and_targets() -> None:
    output = blend_multi_image_samples([
        _sample(0, 1, [0.2, 0.2, 0.2, 0.2]),
        _sample(100, 2, [0.8, 0.8, 0.1, 0.1]),
    ])

    assert output["img"].float().mean() == 50
    assert output["bboxes"].shape[0] == 2
