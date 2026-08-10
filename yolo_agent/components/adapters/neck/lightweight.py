"""Shape-preserving lightweight depthwise neck blocks for YOLO26."""

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

    class LightweightDepthwiseBlock(nn.Module):
        def __init__(self, channels: int, *, expansion: float = 1.0) -> None:
            super().__init__()
            hidden = max(8, int(round(channels * expansion)))
            self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
            self.expand_norm = nn.BatchNorm2d(hidden)
            self.depthwise = nn.Conv2d(
                hidden,
                hidden,
                3,
                padding=1,
                groups=hidden,
                bias=False,
            )
            self.depthwise_norm = nn.BatchNorm2d(hidden)
            self.project = nn.Conv2d(hidden, channels, 1, bias=False)
            self.project_norm = nn.BatchNorm2d(channels)
            self.activation = nn.SiLU(inplace=True)
            nn.init.zeros_(self.project_norm.weight)
            nn.init.zeros_(self.project_norm.bias)

        def forward(self, value: Tensor) -> Tensor:
            update = self.activation(self.expand_norm(self.expand(value)))
            update = self.activation(self.depthwise_norm(self.depthwise(update)))
            return value + self.project_norm(self.project(update))


    class LightweightNeck(nn.Module, ModelGraphPlugin):
        plugin_id = "neck.lightweight"
        plugin_version = "lightweight_neck.v1"
        paper_ids: tuple[str, ...] = ()
        exact_paper_reproduction = False

        def __init__(self, channels: list[int], *, expansion: float = 1.0) -> None:
            super().__init__()
            self.channels = list(channels)
            self.expansion = float(expansion)
            self._contract = build_feature_contract(self.channels)
            self.blocks = nn.ModuleList(
                [
                    LightweightDepthwiseBlock(value, expansion=self.expansion)
                    for value in channels
                ]
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
                sum(
                    2
                    * max(8, int(round(channels * self.expansion)))
                    * (imgsz // stride) ** 2
                    for stride, channels in zip(
                        self._contract.strides,
                        self.channels,
                        strict=True,
                    )
                )
            )

else:

    class LightweightDepthwiseBlock:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("LightweightDepthwiseBlock requires torch")

    class LightweightNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("LightweightNeck requires torch")


__all__ = ["LightweightDepthwiseBlock", "LightweightNeck"]
