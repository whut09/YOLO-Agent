"""Typed contracts and hard gates for isolated model-graph components."""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib.util
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


GraphExecutionClass = Literal["implementation_request", "runtime_candidate"]


class FeaturePyramidContract(BaseModel):
    """Tensor boundary for a feature-pyramid plugin inserted before Detect."""

    model_config = ConfigDict(extra="forbid")

    strides: list[int] = Field(min_length=1)
    channels: list[int] = Field(min_length=1)
    insertion_point: Literal["before_detect"] = "before_detect"

    @model_validator(mode="after")
    def validate_boundary(self) -> "FeaturePyramidContract":
        if len(self.strides) != len(self.channels):
            raise ValueError("feature strides and channels must have equal length")
        if self.strides != sorted(set(self.strides)):
            raise ValueError("feature strides must be unique and ordered fine-to-coarse")
        if any(value <= 0 for value in [*self.strides, *self.channels]):
            raise ValueError("feature strides and channels must be positive")
        return self

    def validate_features(self, features: list[Any] | tuple[Any, ...], imgsz: int) -> None:
        """Fail closed when runtime tensors do not match the declared boundary."""
        if len(features) != len(self.strides):
            raise ValueError(
                f"feature count {len(features)} does not match contract {len(self.strides)}"
            )
        for index, (feature, stride, channels) in enumerate(
            zip(features, self.strides, self.channels, strict=True)
        ):
            if getattr(feature, "ndim", None) != 4:
                raise ValueError(f"feature {index} must be a BCHW tensor")
            actual_channels = int(feature.shape[1])
            actual_stride_h = imgsz // int(feature.shape[-2])
            actual_stride_w = imgsz // int(feature.shape[-1])
            if actual_channels != channels:
                raise ValueError(
                    f"feature {index} has {actual_channels} channels; expected {channels}"
                )
            if actual_stride_h != stride or actual_stride_w != stride:
                raise ValueError(
                    f"feature {index} has stride {(actual_stride_h, actual_stride_w)}; "
                    f"expected {stride}"
                )


class PartialCheckpointAudit(BaseModel):
    """Explicit accounting for base weights after a graph component is inserted."""

    policy: str = "partial_load_graph_extension"
    loaded: bool
    partial: bool
    checkpoint_path: str | None = None
    checkpoint_sha256: str
    matched_keys: list[str] = Field(default_factory=list)
    missing_keys: list[str] = Field(default_factory=list)
    unexpected_keys: list[str] = Field(default_factory=list)
    shape_mismatches: list[str] = Field(default_factory=list)
    key_mapping: dict[str, str] = Field(default_factory=dict)
    matched_parameter_count: int = Field(default=0, ge=0)
    total_parameter_count: int = Field(default=0, ge=0)
    matched_parameter_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    newly_initialized_keys: list[str] = Field(default_factory=list)


class ModelGraphResourceLimits(BaseModel):
    """Maximum allowed regressions before a graph plugin is allowed to train."""

    model_config = ConfigDict(extra="forbid")

    max_latency_regression: float = Field(default=2.0, ge=0.0)
    max_vram_regression: float = Field(default=3.0, ge=0.0)
    max_parameter_regression: float = Field(default=0.5, ge=0.0)
    max_model_size_regression: float = Field(default=0.5, ge=0.0)


class ModelGraphResourceReport(BaseModel):
    """Measured or shape-derived resource deltas and deterministic guard results."""

    base_latency_ms: float = Field(ge=0.0)
    candidate_latency_ms: float = Field(ge=0.0)
    latency_regression: float
    base_vram_estimate_mb: float = Field(ge=0.0)
    candidate_vram_estimate_mb: float = Field(ge=0.0)
    vram_regression: float
    base_parameter_count: int = Field(ge=0)
    candidate_parameter_count: int = Field(ge=0)
    parameter_regression: float
    base_model_size_mb: float = Field(ge=0.0)
    candidate_model_size_mb: float = Field(ge=0.0)
    model_size_regression: float
    limits: ModelGraphResourceLimits
    checks: dict[str, bool]
    passed: bool


class ModelGraphGuardError(RuntimeError):
    """Raised when a model-graph candidate exceeds a hard resource guard."""


class ModelGraphImplementationRequest(BaseModel):
    """Non-executable result for an unavailable optional graph operator."""

    execution_class: Literal["implementation_request"] = "implementation_request"
    component_id: str
    missing_dependency: str
    reason: str
    acceptance_tests: list[str] = Field(
        default_factory=lambda: ["shape", "backward", "amp", "export", "resource_guards"]
    )


class ModelGraphDependencyDecision(BaseModel):
    """Dependency gate outcome; missing deformable ops never silently fall back."""

    execution_class: GraphExecutionClass
    available: bool
    implementation_request: ModelGraphImplementationRequest | None = None


class ModelGraphDependencyGate:
    """Resolve optional graph dependencies without importing or installing them."""

    @staticmethod
    def evaluate(
        *,
        component_id: str,
        deformable_module: str | None,
    ) -> ModelGraphDependencyDecision:
        if not deformable_module:
            return ModelGraphDependencyDecision(
                execution_class="runtime_candidate",
                available=True,
            )
        try:
            available = importlib.util.find_spec(deformable_module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if available:
            return ModelGraphDependencyDecision(
                execution_class="runtime_candidate",
                available=True,
            )
        return ModelGraphDependencyDecision(
            execution_class="implementation_request",
            available=False,
            implementation_request=ModelGraphImplementationRequest(
                component_id=component_id,
                missing_dependency=deformable_module,
                reason=(
                    "deformable graph operator is unavailable; install and validate a local "
                    "operator adapter before creating a training node"
                ),
            ),
        )


class ModelGraphPlugin(ABC):
    """Interface implemented by isolated feature-pyramid graph components."""

    plugin_id: str
    plugin_version: str
    paper_ids: tuple[str, ...] = ()
    exact_paper_reproduction: bool = False

    @property
    @abstractmethod
    def input_contract(self) -> FeaturePyramidContract:
        """Return the expected input tensor boundary."""

    @property
    @abstractmethod
    def output_contract(self) -> FeaturePyramidContract:
        """Return the produced output tensor boundary."""

    @abstractmethod
    def forward(self, features: list[Any] | tuple[Any, ...]) -> list[Any]:
        """Transform feature maps while preserving the declared boundary."""

    @abstractmethod
    def estimated_intermediate_elements(self, *, imgsz: int) -> int:
        """Return a deterministic activation-size estimate used by the VRAM guard."""


def regression_ratio(base: float | int, candidate: float | int) -> float:
    """Return signed relative change, handling a zero baseline conservatively."""
    base_value = float(base)
    candidate_value = float(candidate)
    if base_value == 0.0:
        return 0.0 if candidate_value == 0.0 else float("inf")
    return (candidate_value - base_value) / base_value


def evaluate_resource_guards(
    *,
    base_latency_ms: float,
    candidate_latency_ms: float,
    base_vram_estimate_mb: float,
    candidate_vram_estimate_mb: float,
    base_parameter_count: int,
    candidate_parameter_count: int,
    base_model_size_mb: float,
    candidate_model_size_mb: float,
    limits: ModelGraphResourceLimits,
) -> ModelGraphResourceReport:
    """Evaluate all model-graph resource limits as hard, independent checks."""
    regressions = {
        "latency": regression_ratio(base_latency_ms, candidate_latency_ms),
        "vram": regression_ratio(base_vram_estimate_mb, candidate_vram_estimate_mb),
        "parameters": regression_ratio(base_parameter_count, candidate_parameter_count),
        "model_size": regression_ratio(base_model_size_mb, candidate_model_size_mb),
    }
    checks = {
        "latency": regressions["latency"] <= limits.max_latency_regression,
        "vram": regressions["vram"] <= limits.max_vram_regression,
        "parameters": regressions["parameters"] <= limits.max_parameter_regression,
        "model_size": regressions["model_size"] <= limits.max_model_size_regression,
    }
    return ModelGraphResourceReport(
        base_latency_ms=base_latency_ms,
        candidate_latency_ms=candidate_latency_ms,
        latency_regression=regressions["latency"],
        base_vram_estimate_mb=base_vram_estimate_mb,
        candidate_vram_estimate_mb=candidate_vram_estimate_mb,
        vram_regression=regressions["vram"],
        base_parameter_count=base_parameter_count,
        candidate_parameter_count=candidate_parameter_count,
        parameter_regression=regressions["parameters"],
        base_model_size_mb=base_model_size_mb,
        candidate_model_size_mb=candidate_model_size_mb,
        model_size_regression=regressions["model_size"],
        limits=limits,
        checks=checks,
        passed=all(checks.values()),
    )


__all__ = [
    "FeaturePyramidContract",
    "GraphExecutionClass",
    "ModelGraphDependencyDecision",
    "ModelGraphDependencyGate",
    "ModelGraphGuardError",
    "ModelGraphImplementationRequest",
    "ModelGraphPlugin",
    "ModelGraphResourceLimits",
    "ModelGraphResourceReport",
    "PartialCheckpointAudit",
    "evaluate_resource_guards",
    "regression_ratio",
]
