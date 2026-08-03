from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yolo_agent.components.adapters.neck.spatial_attention import (  # noqa: E402
    SpatialAttentionNeck,
)


def test_spatial_attention_is_identity_safe_and_trainable() -> None:
    neck = SpatialAttentionNeck([16, 24, 32], kernel_size=7)
    features = [
        torch.randn(2, 16, 8, 8, requires_grad=True),
        torch.randn(2, 24, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]

    outputs = neck(features)
    assert all(torch.equal(output, source) for output, source in zip(outputs, features))
    sum(value.mean() for value in outputs).backward()

    assert any(parameter.grad is not None for parameter in neck.parameters())
    assert neck.output_contract.strides == [8, 16, 32]


def test_spatial_attention_rejects_even_kernel_and_supports_amp() -> None:
    with pytest.raises(ValueError, match="must be odd"):
        SpatialAttentionNeck([16, 24, 32], kernel_size=4)

    neck = SpatialAttentionNeck([16, 24, 32], kernel_size=5)
    features = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 24, 4, 4),
        torch.randn(1, 32, 2, 2),
    ]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = neck(features)
    assert [value.shape for value in outputs] == [value.shape for value in features]
