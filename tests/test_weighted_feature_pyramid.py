from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yolo_agent.components.adapters.neck.weighted_fpn import (  # noqa: E402
    WeightedFeaturePyramidNeck,
)


def _features() -> list[torch.Tensor]:
    return [
        torch.randn(2, 16, 8, 8, requires_grad=True),
        torch.randn(2, 24, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]


def test_weighted_fpn_preserves_boundary_and_backpropagates() -> None:
    neck = WeightedFeaturePyramidNeck([16, 24, 32], fusion_channels=12)
    features = _features()

    outputs = neck(features)
    sum(value.mean() for value in outputs).backward()

    assert [value.shape for value in outputs] == [value.shape for value in features]
    assert all(float(weight.detach().sum()) == 2.0 for weight in neck.fusion_weights)
    assert any(parameter.grad is not None for parameter in neck.parameters())


def test_weighted_fpn_starts_as_identity_and_supports_cpu_amp() -> None:
    neck = WeightedFeaturePyramidNeck([16, 24, 32], fusion_channels=12)
    features = [value.detach() for value in _features()]

    outputs = neck(features)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_outputs = neck(features)

    assert all(torch.equal(output, source) for output, source in zip(outputs, features))
    assert [value.shape for value in amp_outputs] == [value.shape for value in features]
