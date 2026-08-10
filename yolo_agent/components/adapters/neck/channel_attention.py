"""Channel-attention graph block with identity-safe initialization."""

from __future__ import annotations

from typing import Any

from yolo_agent.components.adapters.neck.common import build_feature_contract
from yolo_agent.components.model_graph import FeaturePyramidContract, ModelGraphPlugin

try:
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional dependency
    Tensor = Any  # type: ignore[misc, assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class ChannelAttentionBlock(nn.Module):
        def __init__(self, channels: int, *, reduction: int = 8) -> None:
            super().__init__()
            hidden = max(4, channels // reduction)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.reduce = nn.Conv2d(channels, hidden, 1)
            self.activation = nn.SiLU(inplace=True)
            self.expand = nn.Conv2d(hidden, channels, 1)
            nn.init.zeros_(self.expand.weight)
            nn.init.zeros_(self.expand.bias)

        def forward(self, value: Tensor) -> Tensor:
            logits = self.expand(self.activation(self.reduce(self.pool(value))))
            return value * (2.0 * logits.sigmoid())


    class ChannelAttentionNeck(nn.Module, ModelGraphPlugin):
        plugin_id = "attention.channel"
        plugin_version = "channel_attention.v1"
        paper_ids: tuple[str, ...] = ()
        exact_paper_reproduction = False

        def __init__(self, channels: list[int], *, reduction: int = 8) -> None:
            super().__init__()
            self.channels = list(channels)
            self.reduction = int(reduction)
            self._contract = build_feature_contract(self.channels)
            self.blocks = nn.ModuleList(
                [ChannelAttentionBlock(value, reduction=reduction) for value in channels]
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
            del imgsz
            return int(
                sum(channels + max(4, channels // self.reduction) for channels in self.channels)
            )

else:

    class ChannelAttentionBlock:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("ChannelAttentionBlock requires torch")

    class ChannelAttentionNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("ChannelAttentionNeck requires torch")


__all__ = ["ChannelAttentionBlock", "ChannelAttentionNeck"]
