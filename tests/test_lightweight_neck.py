from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yolo_agent.components.adapters.neck.lightweight import LightweightNeck  # noqa: E402


def test_lightweight_neck_preserves_shapes_and_supports_backward_amp() -> None:
    neck = LightweightNeck([16, 24, 32], expansion=0.5)
    features = [
        torch.randn(2, 16, 8, 8, requires_grad=True),
        torch.randn(2, 24, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]

    outputs = neck(features)
    sum(value.mean() for value in outputs).backward()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_outputs = neck([value.detach() for value in features])

    assert [value.shape for value in outputs] == [value.shape for value in features]
    assert [value.shape for value in amp_outputs] == [value.shape for value in features]
    assert any(parameter.grad is not None for parameter in neck.parameters())


def test_lightweight_neck_identity_initialization_and_activation_estimate() -> None:
    neck = LightweightNeck([16, 24, 32], expansion=0.5)
    features = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 24, 4, 4),
        torch.randn(1, 32, 2, 2),
    ]

    outputs = neck(features)

    assert all(torch.equal(output, source) for output, source in zip(outputs, features))
    assert neck.estimated_intermediate_elements(imgsz=640) > 0
