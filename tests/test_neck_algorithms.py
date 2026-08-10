"""Mock graph tests for each isolated multi-scale neck algorithm."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yolo_agent.components.adapters.neck.gold_gd import GoldGatherDistributeNeck  # noqa: E402
from yolo_agent.components.adapters.neck.multi_scale_fusion import (  # noqa: E402
    MultiScaleFusionNeck,
)
from yolo_agent.components.adapters.neck.rtmdet_large_kernel import (  # noqa: E402
    RTMDetLargeKernelNeck,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MultiScaleFusionNeck([8, 16, 32], fusion_channels=8),
        lambda: GoldGatherDistributeNeck([8, 16, 32], context_channels=8),
        lambda: RTMDetLargeKernelNeck([8, 16, 32], kernel_size=5),
    ],
)
def test_neck_plugins_preserve_shapes_and_support_backward_amp_export(factory) -> None:
    neck = factory()
    features = [
        torch.randn(2, 8, 8, 8, requires_grad=True),
        torch.randn(2, 16, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]
    outputs = neck(features)
    assert [item.shape for item in outputs] == [item.shape for item in features]
    sum(item.mean() for item in outputs).backward()
    assert any(parameter.grad is not None for parameter in neck.parameters())

    neck.zero_grad(set_to_none=True)
    detached = [item.detach() for item in features]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_outputs = neck(detached)
    assert [item.shape for item in amp_outputs] == [item.shape for item in features]
    exported = torch.export.export(neck.eval(), (tuple(detached),))
    assert [item.shape for item in exported.module()(tuple(detached))] == [
        item.shape for item in features
    ]


def test_paper_adaptations_are_explicitly_not_exact_reproductions() -> None:
    gold = GoldGatherDistributeNeck([8, 16, 32], context_channels=8)
    rtmdet = RTMDetLargeKernelNeck([8, 16, 32], kernel_size=5)

    assert gold.paper_ids == ("arxiv:2309.11331",)
    assert rtmdet.paper_ids == ("arxiv:2212.07784",)
    assert gold.exact_paper_reproduction is False
    assert rtmdet.exact_paper_reproduction is False
    assert gold.estimated_intermediate_elements(imgsz=640) > 0
    assert rtmdet.estimated_intermediate_elements(imgsz=640) > 0


@pytest.mark.parametrize(
    "neck",
    [
        MultiScaleFusionNeck([8, 16, 32], fusion_channels=8),
        GoldGatherDistributeNeck([8, 16, 32], context_channels=8),
        RTMDetLargeKernelNeck([8, 16, 32], kernel_size=5),
    ],
)
def test_neck_plugins_start_as_identity_residuals(neck) -> None:
    features = [
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 16, 4, 4),
        torch.randn(2, 32, 2, 2),
    ]

    outputs = neck(features)

    assert all(
        torch.allclose(output, feature, atol=1e-6)
        for output, feature in zip(outputs, features, strict=True)
    )


def test_multi_scale_fusion_eval_accepts_rectangular_feature_pyramid() -> None:
    neck = MultiScaleFusionNeck([8, 16, 32], fusion_channels=8).eval()
    features = [
        torch.randn(1, 8, 56, 80),
        torch.randn(1, 16, 28, 40),
        torch.randn(1, 32, 14, 20),
    ]

    with torch.no_grad():
        outputs = neck(features)

    assert [item.shape for item in outputs] == [item.shape for item in features]
