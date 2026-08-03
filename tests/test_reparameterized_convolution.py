from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yolo_agent.components.adapters.neck.repconv import (  # noqa: E402
    ReparameterizableConvBlock,
    ReparameterizedConvolutionNeck,
)


def test_repconv_training_and_deploy_paths_are_equivalent() -> None:
    torch.manual_seed(41)
    block = ReparameterizableConvBlock(8).eval()
    value = torch.randn(2, 8, 9, 9)
    expected = block(value)

    block.switch_to_deploy()
    actual = block(value)

    assert block.deploy_conv is not None
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_repconv_neck_preserves_shapes_and_backpropagates() -> None:
    neck = ReparameterizedConvolutionNeck([8, 12, 16])
    features = [
        torch.randn(2, 8, 8, 8, requires_grad=True),
        torch.randn(2, 12, 4, 4, requires_grad=True),
        torch.randn(2, 16, 2, 2, requires_grad=True),
    ]

    outputs = neck(features)
    sum(value.mean() for value in outputs).backward()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_outputs = neck([value.detach() for value in features])

    assert [value.shape for value in outputs] == [value.shape for value in features]
    assert [value.shape for value in amp_outputs] == [value.shape for value in features]
    assert any(parameter.grad is not None for parameter in neck.parameters())
