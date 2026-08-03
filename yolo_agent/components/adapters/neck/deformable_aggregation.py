"""Dependency-bound deformable feature aggregation for YOLO26 feature scales."""

from __future__ import annotations

import importlib
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


def load_deformable_operator(module_name: str, class_name: str) -> type[Any]:
    """Load an explicit local operator without substituting a normal convolution."""
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            f"deformable operator module is unavailable: {module_name}"
        ) from exc
    operator = getattr(module, class_name, None)
    if not isinstance(operator, type):
        raise ImportError(
            f"deformable operator class is unavailable: {module_name}:{class_name}"
        )
    return operator


if nn is not None:

    class DeformableAggregationBlock(nn.Module):
        def __init__(
            self,
            channels: int,
            *,
            operator_type: type[Any],
            groups: int = 1,
            kernel_size: int = 3,
        ) -> None:
            super().__init__()
            if channels % groups:
                raise ValueError("deformable groups must divide feature channels")
            self.offset = nn.Conv2d(
                channels,
                2 * groups * kernel_size * kernel_size,
                kernel_size,
                padding=kernel_size // 2,
            )
            self.deform = operator_type(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            )
            self.output = nn.Conv2d(channels, channels, 1)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
            self.operator_calls = 0

        def forward(self, value: Tensor) -> Tensor:
            offsets = self.offset(value)
            with torch.autocast(device_type=value.device.type, enabled=False):
                update = self.deform(value.float(), offsets.float())
            self.operator_calls += 1
            return value + self.output(update)


    class DeformableFeatureAggregationNeck(nn.Module, ModelGraphPlugin):
        plugin_id = "neck.deformable_feature_aggregation"
        plugin_version = "deformable_feature_aggregation.v1"
        paper_ids: tuple[str, ...] = ()
        exact_paper_reproduction = False

        def __init__(
            self,
            channels: list[int],
            *,
            operator_module: str,
            operator_class: str = "DeformConv2d",
            groups: int = 1,
        ) -> None:
            super().__init__()
            self.channels = list(channels)
            self.operator_module = operator_module
            self.operator_class = operator_class
            self.groups = int(groups)
            self._contract = build_feature_contract(self.channels)
            operator_type = load_deformable_operator(operator_module, operator_class)
            self.blocks = nn.ModuleList(
                [
                    DeformableAggregationBlock(
                        value,
                        operator_type=operator_type,
                        groups=groups,
                    )
                    for value in channels
                ]
            )

        @property
        def input_contract(self) -> FeaturePyramidContract:
            return self._contract

        @property
        def output_contract(self) -> FeaturePyramidContract:
            return self._contract

        @property
        def operator_calls(self) -> int:
            return sum(block.operator_calls for block in self.blocks)

        def forward(self, features: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor]:
            imgsz = int(features[0].shape[-1]) * self._contract.strides[0]
            self.input_contract.validate_features(features, imgsz)
            outputs = [
                block(feature)
                for block, feature in zip(self.blocks, features, strict=True)
            ]
            self.output_contract.validate_features(outputs, imgsz)
            return outputs

        def estimated_intermediate_elements(self, *, imgsz: int) -> int:
            return int(
                sum(
                    (channels + 18 * self.groups) * (imgsz // stride) ** 2
                    for stride, channels in zip(
                        self._contract.strides,
                        self.channels,
                        strict=True,
                    )
                )
            )

else:

    class DeformableAggregationBlock:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("DeformableAggregationBlock requires torch")

    class DeformableFeatureAggregationNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("DeformableFeatureAggregationNeck requires torch")


__all__ = [
    "DeformableAggregationBlock",
    "DeformableFeatureAggregationNeck",
    "load_deformable_operator",
]
