from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision.ops")

from yolo_agent.components.adapters.neck.deformable_aggregation import (  # noqa: E402
    DeformableFeatureAggregationNeck,
    load_deformable_operator,
)


def test_missing_deformable_operator_fails_closed() -> None:
    with pytest.raises(ImportError, match="module is unavailable"):
        load_deformable_operator("missing_yolo_agent_deformable_ops", "DeformConv2d")
    with pytest.raises(ImportError, match="class is unavailable"):
        load_deformable_operator("torchvision.ops", "MissingDeformConv")


def test_torchvision_deformable_operator_runs_forward_backward_and_amp() -> None:
    neck = DeformableFeatureAggregationNeck(
        [8, 16, 32],
        operator_module="torchvision.ops",
        operator_class="DeformConv2d",
    )
    features = [
        torch.randn(1, 8, 8, 8, requires_grad=True),
        torch.randn(1, 16, 4, 4, requires_grad=True),
        torch.randn(1, 32, 2, 2, requires_grad=True),
    ]

    outputs = neck(features)
    sum(value.mean() for value in outputs).backward()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_outputs = neck([value.detach() for value in features])

    assert neck.operator_calls == 6
    assert [value.shape for value in outputs] == [value.shape for value in features]
    assert [value.shape for value in amp_outputs] == [value.shape for value in features]
    assert any(parameter.grad is not None for parameter in neck.parameters())
