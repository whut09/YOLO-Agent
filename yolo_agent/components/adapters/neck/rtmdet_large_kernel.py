"""Isolated RTMDet-style large-kernel depthwise neck blocks for YOLO26."""

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

    class LargeKernelDepthwiseBlock(nn.Module):
        """Shape-preserving 5x5 depthwise and pointwise residual block."""

        def __init__(self, channels: int, *, kernel_size: int = 5) -> None:
            super().__init__()
            if kernel_size % 2 == 0:
                raise ValueError("large-kernel depthwise convolution requires an odd kernel")
            self.depthwise = nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            )
            self.depthwise_norm = nn.BatchNorm2d(channels)
            self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
            self.pointwise_norm = nn.BatchNorm2d(channels)
            self.activation = nn.SiLU(inplace=True)
            nn.init.zeros_(self.pointwise_norm.weight)
            nn.init.zeros_(self.pointwise_norm.bias)

        def forward(self, value: Tensor) -> Tensor:
            update = self.activation(self.depthwise_norm(self.depthwise(value)))
            update = self.pointwise_norm(self.pointwise(update))
            return value + update


    class RTMDetLargeKernelNeck(nn.Module, ModelGraphPlugin):
        """Apply independent RTMDet-style 5x5 depthwise blocks at P3/P4/P5."""

        plugin_id = "neck.rtmdet_large_kernel"
        plugin_version = "rtmdet_large_kernel.v1"
        paper_ids = ("arxiv:2212.07784",)
        exact_paper_reproduction = False

        def __init__(self, channels: list[int], *, kernel_size: int = 5) -> None:
            super().__init__()
            self.channels = list(channels)
            self.kernel_size = int(kernel_size)
            self._contract = build_feature_contract(self.channels)
            self.blocks = nn.ModuleList(
                [
                    LargeKernelDepthwiseBlock(value, kernel_size=self.kernel_size)
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
            imgsz = int(features[0].shape[-1]) * self._contract.strides[0]
            self.input_contract.validate_features(features, imgsz)
            outputs = [block(feature) for block, feature in zip(self.blocks, features, strict=True)]
            self.output_contract.validate_features(outputs, imgsz)
            return outputs

        def estimated_intermediate_elements(self, *, imgsz: int) -> int:
            return int(
                sum(
                    channels * (imgsz // stride) ** 2
                    for stride, channels in zip(
                        self._contract.strides, self.channels, strict=True
                    )
                )
            )

else:

    class LargeKernelDepthwiseBlock:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("LargeKernelDepthwiseBlock requires torch")

    class RTMDetLargeKernelNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("RTMDetLargeKernelNeck requires torch")


__all__ = ["LargeKernelDepthwiseBlock", "RTMDetLargeKernelNeck"]
