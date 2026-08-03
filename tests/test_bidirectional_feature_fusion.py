from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yolo_agent.components.adapters.neck.bidirectional_fusion import (  # noqa: E402
    BidirectionalFeatureFusionNeck,
)
from yolo_agent.components.adapters.neck.multi_scale_fusion import (  # noqa: E402
    MultiScaleFusionNeck,
)


def test_bidirectional_fusion_reuses_algorithm_with_independent_identity() -> None:
    neck = BidirectionalFeatureFusionNeck([16, 24, 32], fusion_channels=12)
    features = [
        torch.randn(1, 16, 8, 8, requires_grad=True),
        torch.randn(1, 24, 4, 4, requires_grad=True),
        torch.randn(1, 32, 2, 2, requires_grad=True),
    ]

    outputs = neck(features)
    sum(value.mean() for value in outputs).backward()

    assert isinstance(neck, MultiScaleFusionNeck)
    assert neck.plugin_id == "neck.bidirectional_feature_fusion"
    assert [value.shape for value in outputs] == [value.shape for value in features]
    assert any(parameter.grad is not None for parameter in neck.parameters())
