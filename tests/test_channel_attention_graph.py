from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yolo_agent.components.adapters.neck.channel_attention import (  # noqa: E402
    ChannelAttentionNeck,
)


def test_channel_attention_is_identity_safe_and_trainable() -> None:
    neck = ChannelAttentionNeck([16, 24, 32], reduction=8)
    features = [
        torch.randn(2, 16, 8, 8, requires_grad=True),
        torch.randn(2, 24, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]

    outputs = neck(features)
    assert all(torch.equal(output, source) for output, source in zip(outputs, features))
    sum(value.mean() for value in outputs).backward()

    assert any(parameter.grad is not None for parameter in neck.parameters())
    assert neck.input_contract.channels == [16, 24, 32]


def test_channel_attention_supports_cpu_amp() -> None:
    neck = ChannelAttentionNeck([16, 24, 32], reduction=4)
    features = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 24, 4, 4),
        torch.randn(1, 32, 2, 2),
    ]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = neck(features)

    assert [value.shape for value in outputs] == [value.shape for value in features]
