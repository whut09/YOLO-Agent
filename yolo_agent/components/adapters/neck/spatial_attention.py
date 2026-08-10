"""Spatial-attention graph block with explicit per-scale contracts."""

from __future__ import annotations

from typing import Any

from yolo_agent.components.adapters.neck.common import build_feature_contract
from yolo_agent.components.model_graph import FeaturePyramidContract, ModelGraphPlugin

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc, assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class SpatialAttentionBlock(nn.Module):
        def __init__(self, *, kernel_size: int = 7) -> None:
            super().__init__()
            if kernel_size % 2 == 0:
                raise ValueError("spatial attention kernel must be odd")
            self.mask = nn.Conv2d(
                2,
                1,
                kernel_size,
                padding=kernel_size // 2,
            )
            nn.init.zeros_(self.mask.weight)
            nn.init.zeros_(self.mask.bias)

        def forward(self, value: Tensor) -> Tensor:
            descriptor = torch.cat(
                [value.mean(dim=1, keepdim=True), value.amax(dim=1, keepdim=True)],
                dim=1,
            )
            return value * (2.0 * self.mask(descriptor).sigmoid())


    class SpatialAttentionNeck(nn.Module, ModelGraphPlugin):
        plugin_id = "attention.spatial"
        plugin_version = "spatial_attention.v1"
        paper_ids: tuple[str, ...] = ()
        exact_paper_reproduction = False

        def __init__(self, channels: list[int], *, kernel_size: int = 7) -> None:
            super().__init__()
            self.channels = list(channels)
            self.kernel_size = int(kernel_size)
            self._contract = build_feature_contract(self.channels)
            self.blocks = nn.ModuleList(
                [SpatialAttentionBlock(kernel_size=kernel_size) for _ in channels]
            )

        @property
        def input_contract(self) -> FeaturePyramidContract:
            return self._contract

        @property
        def output_contract(self) -> FeaturePyramidContract:
            return self._contract

        def forward(self, features: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor]:
            input_hw = self.input_contract.input_hw_from_finest_feature(features)
            self.input_contract.validate_features(features, input_hw)
            outputs = [
                block(feature)
                for block, feature in zip(self.blocks, features, strict=True)
            ]
            self.output_contract.validate_features(outputs, input_hw)
            return outputs

        def estimated_intermediate_elements(self, *, imgsz: int) -> int:
            return int(
                sum(3 * (imgsz // stride) ** 2 for stride in self._contract.strides)
            )

else:

    class SpatialAttentionBlock:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("SpatialAttentionBlock requires torch")

    class SpatialAttentionNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("SpatialAttentionNeck requires torch")


__all__ = ["SpatialAttentionBlock", "SpatialAttentionNeck"]
