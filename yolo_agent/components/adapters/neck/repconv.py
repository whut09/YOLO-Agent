"""Re-parameterized convolution blocks inserted before native YOLO26 Detect."""

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

    class ReparameterizableConvBlock(nn.Module):
        """3x3, 1x1, and identity branches that fuse into one deploy convolution."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.channels = int(channels)
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.branch_identity = nn.BatchNorm2d(channels)
            self.activation = nn.SiLU(inplace=False)
            self.deploy_conv: nn.Conv2d | None = None

        def forward(self, value: Tensor) -> Tensor:
            if self.deploy_conv is not None:
                return self.activation(self.deploy_conv(value))
            return self.activation(
                self.branch_3x3(value)
                + self.branch_1x1(value)
                + self.branch_identity(value)
            )

        def equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
            kernel_3x3, bias_3x3 = _fuse_conv_bn(
                self.branch_3x3[0],
                self.branch_3x3[1],
            )
            kernel_1x1, bias_1x1 = _fuse_conv_bn(
                self.branch_1x1[0],
                self.branch_1x1[1],
            )
            kernel_1x1 = F.pad(kernel_1x1, [1, 1, 1, 1])
            identity = torch.zeros_like(kernel_3x3)
            indices = torch.arange(self.channels, device=identity.device)
            identity[indices, indices, 1, 1] = 1.0
            kernel_identity, bias_identity = _fuse_kernel_bn(
                identity,
                self.branch_identity,
            )
            return (
                kernel_3x3 + kernel_1x1 + kernel_identity,
                bias_3x3 + bias_1x1 + bias_identity,
            )

        def switch_to_deploy(self) -> None:
            if self.deploy_conv is not None:
                return
            kernel, bias = self.equivalent_kernel_bias()
            deploy = nn.Conv2d(
                self.channels,
                self.channels,
                3,
                padding=1,
                bias=True,
            ).to(device=kernel.device, dtype=kernel.dtype)
            deploy.weight.data.copy_(kernel)
            deploy.bias.data.copy_(bias)
            self.deploy_conv = deploy
            del self.branch_3x3
            del self.branch_1x1
            del self.branch_identity


    class ReparameterizedConvolutionNeck(nn.Module, ModelGraphPlugin):
        plugin_id = "block.reparameterized_convolution"
        plugin_version = "reparameterized_convolution.v1"
        paper_ids: tuple[str, ...] = ()
        exact_paper_reproduction = False

        def __init__(self, channels: list[int]) -> None:
            super().__init__()
            self.channels = list(channels)
            self._contract = build_feature_contract(self.channels)
            self.blocks = nn.ModuleList(
                [ReparameterizableConvBlock(value) for value in channels]
            )
            self.output = nn.ModuleList(
                [nn.Conv2d(value, value, 1) for value in channels]
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

        def forward(self, features: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor]:
            input_hw = self.input_contract.input_hw_from_finest_feature(features)
            self.input_contract.validate_features(features, input_hw)
            outputs = [
                source + projection(block(source))
                for source, block, projection in zip(
                    features,
                    self.blocks,
                    self.output,
                    strict=True,
                )
            ]
            self.output_contract.validate_features(outputs, input_hw)
            return outputs

        def switch_to_deploy(self) -> None:
            for block in self.blocks:
                block.switch_to_deploy()

        def estimated_intermediate_elements(self, *, imgsz: int) -> int:
            return int(
                sum(
                    3 * channels * (imgsz // stride) ** 2
                    for stride, channels in zip(
                        self._contract.strides,
                        self.channels,
                        strict=True,
                    )
                )
            )


    def _fuse_conv_bn(conv: nn.Conv2d, norm: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
        return _fuse_kernel_bn(conv.weight, norm)


    def _fuse_kernel_bn(kernel: Tensor, norm: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
        scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
        fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
        fused_bias = norm.bias - norm.running_mean * scale
        return fused_kernel, fused_bias

else:

    class ReparameterizableConvBlock:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("ReparameterizableConvBlock requires torch")

    class ReparameterizedConvolutionNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("ReparameterizedConvolutionNeck requires torch")


__all__ = ["ReparameterizableConvBlock", "ReparameterizedConvolutionNeck"]
