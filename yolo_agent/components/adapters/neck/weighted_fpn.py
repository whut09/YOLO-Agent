"""Weighted feature-pyramid fusion isolated from any complete detector."""

from __future__ import annotations

from typing import Any

from yolo_agent.components.adapters.neck.common import build_feature_contract
from yolo_agent.components.model_graph import FeaturePyramidContract, ModelGraphPlugin

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc, assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


if nn is not None:

    class WeightedFeaturePyramidNeck(nn.Module, ModelGraphPlugin):
        """Top-down feature pyramid with explicit normalized learnable weights."""

        plugin_id = "neck.weighted_feature_pyramid"
        plugin_version = "weighted_feature_pyramid.v1"
        paper_ids: tuple[str, ...] = ()
        exact_paper_reproduction = False

        def __init__(
            self,
            channels: list[int],
            *,
            fusion_channels: int = 64,
            epsilon: float = 1e-4,
        ) -> None:
            super().__init__()
            self.channels = list(channels)
            self.fusion_channels = int(fusion_channels)
            self.epsilon = float(epsilon)
            self._contract = build_feature_contract(self.channels)
            self.lateral = nn.ModuleList(
                [nn.Conv2d(value, self.fusion_channels, 1) for value in channels]
            )
            self.output = nn.ModuleList(
                [nn.Conv2d(self.fusion_channels, value, 1) for value in channels]
            )
            self.fusion_weights = nn.ParameterList(
                [nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2))]
            )
            for projection in self.output:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

        @property
        def input_contract(self) -> FeaturePyramidContract:
            return self._contract

        @property
        def output_contract(self) -> FeaturePyramidContract:
            return self._contract

        def _weights(self, index: int) -> Tensor:
            values = F.relu(self.fusion_weights[index])
            return values / (values.sum() + self.epsilon)

        def forward(self, features: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor]:
            input_hw = self.input_contract.input_hw_from_finest_feature(features)
            self.input_contract.validate_features(features, input_hw)
            lateral = [
                projection(feature)
                for projection, feature in zip(self.lateral, features, strict=True)
            ]
            p5 = lateral[2]
            p4_weights = self._weights(0)
            p4 = (
                p4_weights[0] * lateral[1]
                + p4_weights[1]
                * F.interpolate(p5, size=lateral[1].shape[-2:], mode="nearest")
            )
            p3_weights = self._weights(1)
            p3 = (
                p3_weights[0] * lateral[0]
                + p3_weights[1]
                * F.interpolate(p4, size=lateral[0].shape[-2:], mode="nearest")
            )
            fused = [p3, p4, p5]
            outputs = [
                source + projection(value)
                for source, projection, value in zip(
                    features,
                    self.output,
                    fused,
                    strict=True,
                )
            ]
            self.output_contract.validate_features(outputs, input_hw)
            return outputs

        def estimated_intermediate_elements(self, *, imgsz: int) -> int:
            spatial = sum((imgsz // stride) ** 2 for stride in self._contract.strides)
            return int(spatial * self.fusion_channels * 2)

else:

    class WeightedFeaturePyramidNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("WeightedFeaturePyramidNeck requires torch")


__all__ = ["WeightedFeaturePyramidNeck"]
