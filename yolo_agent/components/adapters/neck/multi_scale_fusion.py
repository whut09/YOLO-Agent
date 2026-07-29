"""Generic bidirectional multi-scale fusion for the YOLO26 P3/P4/P5 boundary."""

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

    class MultiScaleFusionNeck(nn.Module, ModelGraphPlugin):
        """Fuse all scales in top-down and bottom-up directions with residual outputs."""

        plugin_id = "neck.multi_scale_fusion"
        plugin_version = "multi_scale_fusion.v1"
        exact_paper_reproduction = False

        def __init__(self, channels: list[int], *, fusion_channels: int = 64) -> None:
            super().__init__()
            self.channels = list(channels)
            self.fusion_channels = int(fusion_channels)
            self._contract = build_feature_contract(self.channels)
            self.lateral = nn.ModuleList(
                [nn.Conv2d(value, self.fusion_channels, 1) for value in channels]
            )
            self.output = nn.ModuleList(
                [nn.Conv2d(self.fusion_channels, value, 1) for value in channels]
            )
            for projection in self.output:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
            self.refine = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            self.fusion_channels,
                            self.fusion_channels,
                            3,
                            padding=1,
                            groups=self.fusion_channels,
                        ),
                        nn.BatchNorm2d(self.fusion_channels),
                        nn.SiLU(inplace=True),
                    )
                    for _ in channels
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
            lateral = [projection(feature) for projection, feature in zip(self.lateral, features, strict=True)]
            top_down = [lateral[0], lateral[1], lateral[2]]
            top_down[1] = top_down[1] + F.interpolate(top_down[2], size=top_down[1].shape[-2:], mode="nearest")
            top_down[0] = top_down[0] + F.interpolate(top_down[1], size=top_down[0].shape[-2:], mode="nearest")
            bottom_up = [top_down[0], top_down[1], top_down[2]]
            bottom_up[1] = bottom_up[1] + F.adaptive_avg_pool2d(bottom_up[0], bottom_up[1].shape[-2:])
            bottom_up[2] = bottom_up[2] + F.adaptive_avg_pool2d(bottom_up[1], bottom_up[2].shape[-2:])
            outputs = [
                feature + projection(refine(fused))
                for feature, projection, refine, fused in zip(
                    features, self.output, self.refine, bottom_up, strict=True
                )
            ]
            self.output_contract.validate_features(outputs, imgsz)
            return outputs

        def estimated_intermediate_elements(self, *, imgsz: int) -> int:
            spatial = sum((imgsz // stride) ** 2 for stride in self._contract.strides)
            return int(spatial * self.fusion_channels * 3)

else:

    class MultiScaleFusionNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("MultiScaleFusionNeck requires torch")


__all__ = ["MultiScaleFusionNeck"]
