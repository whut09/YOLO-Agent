"""Independent multi-scale feature-pyramid graph for YOLO26."""

from __future__ import annotations

from typing import Any

from yolo_agent.components.adapters.neck.common import build_feature_contract
from yolo_agent.components.model_graph import FeaturePyramidContract, ModelGraphPlugin

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc, assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


if nn is not None:

    class MultiScaleFeaturePyramidNeck(nn.Module, ModelGraphPlugin):
        """Use one scale-aware context gate while preserving P3/P4/P5 tensors."""

        plugin_id = "feature_pyramid.multi_scale"
        plugin_version = "feature_pyramid_multi_scale.v1"
        paper_ids = ("arxiv:2309.11331",)
        exact_paper_reproduction = False

        def __init__(self, channels: list[int], *, fusion_channels: int = 64) -> None:
            super().__init__()
            self.channels = list(channels)
            self.fusion_channels = int(fusion_channels)
            self._contract = build_feature_contract(self.channels)
            self.projections = nn.ModuleList(
                [nn.Conv2d(value, self.fusion_channels, 1) for value in channels]
            )
            self.gates = nn.ModuleList(
                [nn.Conv2d(self.fusion_channels, value, 1) for value in channels]
            )
            self.refinements = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(value, value, 3, padding=1, groups=value),
                        nn.BatchNorm2d(value),
                        nn.SiLU(inplace=True),
                    )
                    for value in channels
                ]
            )
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)

        @property
        def input_contract(self) -> FeaturePyramidContract:
            return self._contract

        @property
        def output_contract(self) -> FeaturePyramidContract:
            return self._contract

        def forward(self, features: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor]:
            input_hw = self.input_contract.input_hw_from_finest_feature(features)
            self.input_contract.validate_features(features, input_hw)
            projected = [
                projection(feature)
                for projection, feature in zip(self.projections, features, strict=True)
            ]
            context = projected[0]
            for value in projected[1:]:
                context = context + F.interpolate(
                    value, size=context.shape[-2:], mode="nearest"
                )
            outputs = []
            for feature, gate, refine in zip(
                features, self.gates, self.refinements, strict=True
            ):
                local = F.interpolate(context, size=feature.shape[-2:], mode="nearest")
                modulation = 2.0 * torch.sigmoid(gate(local))
                outputs.append(feature + refine(feature) * modulation)
            self.output_contract.validate_features(outputs, input_hw)
            return outputs

        def estimated_intermediate_elements(self, *, imgsz: int) -> int:
            return int(
                self.fusion_channels * (imgsz // 8) ** 2
                + sum(value * (imgsz // stride) ** 2 for stride, value in zip(
                    self._contract.strides, self.channels, strict=True
                ))
            )

else:

    class MultiScaleFeaturePyramidNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("feature pyramid runtime requires torch")


__all__ = ["MultiScaleFeaturePyramidNeck"]
