"""Controlled P2 feature head for small-object experiments.

The module is intentionally framework-neutral.  It provides the tensor and
checkpoint contract needed by the harness; wiring it into an Ultralytics
trainer remains an explicit adapter milestone and is not enabled by this
module alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)

try:  # torch is an optional dependency for the core harness
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised in minimal installations
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc, assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


class P2HeadConfig(BaseModel):
    """Explicit graph and checkpoint policy for a P2 experiment."""

    p2_stride: int = Field(default=4, ge=1)
    source_strides: list[int] = Field(default_factory=lambda: [8, 16, 32])
    p2_channels: int = Field(default=128, ge=1)
    num_classes: int = Field(default=80, ge=1)
    in_channels: list[int] = Field(default_factory=lambda: [64, 128, 256, 512], min_length=4, max_length=4)
    checkpoint_policy: str = "partial_load_new_head"
    imgsz: int = 640

    @model_validator(mode="after")
    def _protocol(self) -> "P2HeadConfig":
        if self.imgsz != 640:
            raise ValueError("P2 head experiments require fixed imgsz=640")
        if self.p2_stride >= min(self.source_strides):
            raise ValueError("p2_stride must be finer than all source feature strides")
        if self.checkpoint_policy not in {"partial_load_new_head", "strict", "reject"}:
            raise ValueError("unsupported checkpoint_policy")
        return self


class P2HeadCheckpointReport(BaseModel):
    """Auditable result of loading a checkpoint into a changed graph."""

    policy: str
    loaded: bool
    partial: bool
    missing_keys: list[str] = Field(default_factory=list)
    unexpected_keys: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


if nn is not None:

    class P2Head(nn.Module):
        """Small-object head that fuses a stride-4 feature with P3/P4/P5.

        Inputs are ordered ``[P2, P3, P4, P5]`` and their spatial sizes must
        follow the declared strides.  The returned tensor is at P2 resolution.
        """

        def __init__(self, in_channels: list[int], config: P2HeadConfig | None = None) -> None:
            super().__init__()
            self.config = config or P2HeadConfig()
            if len(in_channels) != 4:
                raise ValueError("P2Head expects channels for P2, P3, P4, and P5")
            self.projections = nn.ModuleList(
                [nn.Conv2d(channels, self.config.p2_channels, 1) for channels in in_channels]
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(self.config.p2_channels * 4, self.config.p2_channels, 3, padding=1),
                nn.BatchNorm2d(self.config.p2_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, features: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
            if len(features) != 4:
                raise ValueError("P2Head expects four feature maps [P2, P3, P4, P5]")
            target = features[0].shape[-2:]
            projected = []
            for feature, projection in zip(features, self.projections):
                value = projection(feature)
                if value.shape[-2:] != target:
                    value = F.interpolate(value, size=target, mode="nearest")
                projected.append(value)
            return self.fuse(torch.cat(projected, dim=1))

        @staticmethod
        def validate_feature_strides(
            features: list[Tensor] | tuple[Tensor, ...],
            input_size: int | tuple[int, int],
            expected_strides: list[int] | tuple[int, ...] = (4, 8, 16, 32),
        ) -> dict[str, int]:
            """Validate strides from actual feature shapes, not config labels."""
            if len(features) != len(expected_strides):
                raise ValueError("feature count must match expected_strides")
            height, width = (input_size, input_size) if isinstance(input_size, int) else input_size
            actual: dict[str, int] = {}
            for index, (feature, expected) in enumerate(zip(features, expected_strides)):
                stride_h = height // feature.shape[-2]
                stride_w = width // feature.shape[-1]
                if stride_h != stride_w or stride_h != expected:
                    raise ValueError(f"feature {index} has stride {(stride_h, stride_w)}; expected {expected}")
                actual[f"p{index + 2}"] = stride_h
            return actual

else:

    class P2Head:  # type: ignore[no-redef]
        """Placeholder that explains the optional torch dependency."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("P2Head requires the optional torch dependency")


class P2HeadAdapter(ComponentAdapter):
    """Dry-run-safe adapter describing a changed detection graph."""

    adapter_version = "p2_head.v1"
    source_commit = "local"
    strategy = "custom_module"
    modified_model_fields = frozenset({"p2_head"})
    modified_training_fields = frozenset()

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        if torch is None:
            return AdapterValidationReport(ok=False, errors=["torch is required for P2 head shape/backward checks"])
        return AdapterValidationReport(ok=True, checks={"torch": torch.__version__})

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        if context.imgsz != 640:
            return AdapterValidationReport(ok=False, errors=["P2 head requires fixed imgsz=640"])
        options = P2HeadConfig.model_validate(context.options or {})
        warnings = ["P2 changes the model graph; pretrained checkpoints require partial-load accounting."]
        if context.detector_family == "yolo26" and context.head == "one_to_one":
            warnings.append("YOLO26 one-to-one head integration requires a trainer/model adapter.")
        return AdapterValidationReport(ok=True, warnings=warnings, checks={"p2_stride": options.p2_stride, "imgsz": 640})

    def patch_model_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        options = P2HeadConfig.model_validate(context.options or {})
        config["p2_head"] = options.model_dump(mode="json")
        return config

    def patch_training_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        return config

    def build_module(self, context: AdapterContext) -> P2Head:
        options = P2HeadConfig.model_validate(context.options or {})
        return P2Head(options.in_channels, options)

    def load_pretrained_weights(self, module: Any, weights: Path | str | None, context: AdapterContext) -> WeightLoadResult:
        if weights is None:
            return WeightLoadResult(loaded=False, message="No checkpoint supplied; new P2 layers require initialization")
        path = Path(weights)
        if not path.is_file():
            return WeightLoadResult(loaded=False, source=path, message="Checkpoint not found")
        return WeightLoadResult(
            loaded=False,
            source=path,
            message="P2 graph changes require caller-owned partial checkpoint loading; strict loading is not implicit",
        )

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        if torch is None:
            return SmokeTestResult(passed=False, errors=["torch is required"])
        try:
            options = P2HeadConfig.model_validate(context.options or {})
            channels = [64, 128, 256, 512]
            module = P2Head(channels, options)
            features = [torch.randn(2, c, 160 // (2 ** index), 160 // (2 ** index), requires_grad=True) for index, c in enumerate(channels)]
            strides = P2Head.validate_feature_strides(features, 640)
            output = module(features)
            output.mean().backward()
            return SmokeTestResult(passed=True, checks={"feature_strides": str(strides), "shape": str(tuple(output.shape)), "backward": True, "checkpoint_policy": options.checkpoint_policy})
        except (RuntimeError, ValueError) as exc:
            return SmokeTestResult(passed=False, errors=[str(exc)])

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        return [ExpectedArtifact(name="p2_head_manifest", relative_path=Path("artifacts/p2_head_manifest.json"))]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(actions=["remove p2_head model patch and discard new head weights"], files_to_remove=[Path("artifacts/p2_head_manifest.json")])


__all__ = ["P2Head", "P2HeadAdapter", "P2HeadCheckpointReport", "P2HeadConfig"]
