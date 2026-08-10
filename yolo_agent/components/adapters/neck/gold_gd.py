"""Isolated Gold-YOLO gather-distribute neck adaptation for YOLO26."""

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

    class GoldGatherDistributeNeck(nn.Module, ModelGraphPlugin):
        """Gather at P4, mix global context, then gate and inject into each native scale."""

        plugin_id = "neck.gold_gather_distribute"
        plugin_version = "gold_gather_distribute.v1"
        paper_ids = ("arxiv:2309.11331",)
        exact_paper_reproduction = False

        def __init__(self, channels: list[int], *, context_channels: int = 64) -> None:
            super().__init__()
            self.channels = list(channels)
            self.context_channels = int(context_channels)
            self._contract = build_feature_contract(self.channels)
            self.gather_projection = nn.ModuleList(
                [nn.Conv2d(value, self.context_channels, 1) for value in channels]
            )
            self.context_mixer = nn.Sequential(
                nn.Conv2d(self.context_channels * 3, self.context_channels, 3, padding=1),
                nn.BatchNorm2d(self.context_channels),
                nn.SiLU(inplace=True),
            )
            self.gates = nn.ModuleList(
                [nn.Conv2d(self.context_channels, value, 1) for value in channels]
            )
            self.embeddings = nn.ModuleList(
                [nn.Conv2d(self.context_channels, value, 1) for value in channels]
            )
            for gate, embedding in zip(self.gates, self.embeddings, strict=True):
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
                nn.init.zeros_(embedding.weight)
                nn.init.zeros_(embedding.bias)

        @property
        def input_contract(self) -> FeaturePyramidContract:
            return self._contract

        @property
        def output_contract(self) -> FeaturePyramidContract:
            return self._contract

        def forward(self, features: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor]:
            input_hw = self.input_contract.input_hw_from_finest_feature(features)
            self.input_contract.validate_features(features, input_hw)
            target_size = features[1].shape[-2:]
            gathered: list[Tensor] = []
            for index, (feature, projection) in enumerate(
                zip(features, self.gather_projection, strict=True)
            ):
                value = projection(feature)
                if index == 0:
                    value = F.adaptive_avg_pool2d(value, target_size)
                elif index == 2:
                    value = F.interpolate(value, size=target_size, mode="nearest")
                gathered.append(value)
            context = self.context_mixer(torch.cat(gathered, dim=1))
            outputs: list[Tensor] = []
            for feature, gate, embedding in zip(
                features, self.gates, self.embeddings, strict=True
            ):
                distributed = F.interpolate(context, size=feature.shape[-2:], mode="nearest")
                outputs.append(
                    feature * (2.0 * torch.sigmoid(gate(distributed)))
                    + embedding(distributed)
                )
            self.output_contract.validate_features(outputs, input_hw)
            return outputs

        def estimated_intermediate_elements(self, *, imgsz: int) -> int:
            p4_area = (imgsz // 16) ** 2
            distributed = sum(
                2 * channels * (imgsz // stride) ** 2
                for stride, channels in zip(self._contract.strides, self.channels, strict=True)
            )
            return int(p4_area * self.context_channels * 4 + distributed)

else:

    class GoldGatherDistributeNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("GoldGatherDistributeNeck requires torch")


__all__ = ["GoldGatherDistributeNeck"]
