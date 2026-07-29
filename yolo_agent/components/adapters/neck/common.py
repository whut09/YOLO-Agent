"""Shared runtime machinery for guarded YOLO26 feature-pyramid plugins."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from statistics import median
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from yolo_agent.components.model_graph import (
    FeaturePyramidContract,
    ModelGraphGuardError,
    ModelGraphPlugin,
    ModelGraphResourceLimits,
    ModelGraphResourceReport,
    PartialCheckpointAudit,
    evaluate_resource_guards,
)

try:  # torch remains optional for the metadata-only harness
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - minimal installations
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


NeckKind = Literal["multi_scale_fusion", "gold_gather_distribute", "rtmdet_large_kernel"]
YOLO26_NECK_STRIDES = [8, 16, 32]


class YOLO26NeckConfig(BaseModel):
    """Common fixed protocol and resource policy for one neck component."""

    kind: NeckKind
    component_id: str
    imgsz: int = 640
    expected_strides: list[int] = Field(default_factory=lambda: list(YOLO26_NECK_STRIDES))
    expected_channels: list[int] = Field(default_factory=list)
    insertion_point: Literal["before_detect"] = "before_detect"
    audit_imgsz: int = Field(default=64, ge=64)
    latency_warmup: int = Field(default=1, ge=0, le=10)
    latency_iterations: int = Field(default=2, ge=1, le=20)
    resource_limits: ModelGraphResourceLimits = Field(default_factory=ModelGraphResourceLimits)
    deformable_module: str | None = None
    kernel_size: int = Field(default=5, ge=3, le=15)
    context_channels: int = Field(default=64, ge=8)

    @model_validator(mode="after")
    def validate_protocol(self) -> "YOLO26NeckConfig":
        if self.imgsz != 640:
            raise ValueError("multi-scale neck experiments require fixed imgsz=640")
        if self.expected_strides != YOLO26_NECK_STRIDES:
            raise ValueError("YOLO26 neck plugins require P3/P4/P5 strides [8, 16, 32]")
        if self.expected_channels and len(self.expected_channels) != len(self.expected_strides):
            raise ValueError("expected_channels must describe P3/P4/P5")
        if self.audit_imgsz % max(self.expected_strides):
            raise ValueError("audit_imgsz must be divisible by the coarsest feature stride")
        if self.kernel_size % 2 == 0:
            raise ValueError("large-kernel neck requires an odd kernel size")
        return self


class YOLO26NeckManifest(BaseModel):
    """Runtime proof for graph boundary, checkpoint transfer, and hard guards."""

    schema_version: str = "yolo26_neck_manifest.v1"
    component_id: str
    neck_kind: NeckKind
    adapter_class: str
    adapter_version: str
    plugin_class: str
    plugin_version: str
    adapter_hash: str
    protocol_hash: str
    paper_ids: list[str] = Field(default_factory=list)
    exact_paper_reproduction: bool = False
    insertion_point: str
    input_strides: list[int]
    input_channels: list[int]
    output_strides: list[int]
    output_channels: list[int]
    native_end2end: bool
    native_reg_max: int
    dfl_disabled: bool
    external_nms_added: bool = False
    checkpoint: PartialCheckpointAudit
    resources: ModelGraphResourceReport
    export_dry_run: bool


if nn is not None:

    class DetectWithFeaturePyramidNeck(nn.Module):
        """Apply one isolated neck component immediately before native Detect."""

        def __init__(self, detect: nn.Module, neck: nn.Module) -> None:
            super().__init__()
            if not isinstance(neck, ModelGraphPlugin):
                raise TypeError("neck must implement ModelGraphPlugin")
            self.detect = detect
            self.neck = neck
            self.f = detect.f
            self.i = detect.i
            self.type = f"{type(neck).__name__}+{getattr(detect, 'type', type(detect).__name__)}"
            self.np = sum(int(parameter.numel()) for parameter in self.parameters())

        def forward(self, features: list[Any] | tuple[Any, ...]) -> Any:
            transformed = self.neck.forward(features)
            return self.detect(transformed)

        def __getattr__(self, name: str) -> Any:
            try:
                return super().__getattr__(name)
            except AttributeError:
                detect = super().__getattr__("detect")
                return getattr(detect, name)

        @property
        def export(self) -> bool:
            return bool(self.detect.export)

        @export.setter
        def export(self, value: bool) -> None:
            self.detect.export = value

        @property
        def format(self) -> str:
            return str(self.detect.format)

        @format.setter
        def format(self, value: str) -> None:
            self.detect.format = value

else:

    class DetectWithFeaturePyramidNeck:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("YOLO26 neck runtime requires torch")


def infer_detect_channels(detect: Any) -> list[int]:
    """Infer native Detect input channels from its regression branches."""
    channels: list[int] = []
    for branch in getattr(detect, "cv2", []):
        first = branch[0]
        convolution = getattr(first, "conv", first)
        value = getattr(convolution, "in_channels", None)
        if value is None:
            raise ValueError("unable to infer Detect input channels")
        channels.append(int(value))
    if len(channels) != 3:
        raise ValueError(f"YOLO26 neck requires three Detect scales, got {channels}")
    return channels


def assert_native_yolo26_graph(model: Any) -> tuple[Any, list[int]]:
    """Return native Detect and channels after auditing YOLO26 invariants."""
    detect = model.model[-1]
    strides = [int(value) for value in detect.stride.tolist()]
    if strides != YOLO26_NECK_STRIDES:
        raise ValueError(f"YOLO26 neck requires strides {YOLO26_NECK_STRIDES}, got {strides}")
    if not bool(getattr(model, "end2end", False)) or not bool(getattr(detect, "end2end", False)):
        raise ValueError("YOLO26 neck requires the native end-to-end/NMS-free Detect path")
    if int(getattr(detect, "reg_max", -1)) != 1:
        raise ValueError("YOLO26 neck requires native DFL-free reg_max=1")
    if type(getattr(detect, "dfl", None)).__name__ != "Identity":
        raise ValueError("YOLO26 neck must preserve the native DFL-free regression path")
    return detect, infer_detect_channels(detect)


def build_feature_contract(channels: list[int]) -> FeaturePyramidContract:
    return FeaturePyramidContract(
        strides=list(YOLO26_NECK_STRIDES),
        channels=list(channels),
        insertion_point="before_detect",
    )


def audit_partial_checkpoint(
    *,
    source_state: dict[str, Any],
    target_state: dict[str, Any],
    detect_index: int,
    checkpoint_path: Path | None,
) -> PartialCheckpointAudit:
    """Map the wrapped Detect tensors to their original keys and expose new neck keys."""
    selected: dict[str, str] = {}
    shape_mismatches: list[str] = []
    consumed: set[str] = set()
    prefix = f"model.{detect_index}."
    wrapped_prefix = f"model.{detect_index}.detect."
    neck_prefix = f"model.{detect_index}.neck."
    for target_key, target_value in target_state.items():
        if target_key.startswith(neck_prefix):
            continue
        source_key = (
            prefix + target_key.removeprefix(wrapped_prefix)
            if target_key.startswith(wrapped_prefix)
            else target_key
        )
        source_value = source_state.get(source_key)
        if source_value is None:
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            shape_mismatches.append(
                f"{source_key}->{target_key}:{tuple(source_value.shape)}!={tuple(target_value.shape)}"
            )
            continue
        selected[target_key] = source_key
        consumed.add(source_key)
    missing = sorted(set(target_state) - set(selected))
    unexpected = sorted(set(source_state) - consumed)
    matched_parameters = sum(int(target_state[key].numel()) for key in selected)
    total_parameters = sum(int(value.numel()) for value in target_state.values())
    return PartialCheckpointAudit(
        loaded=bool(selected),
        partial=bool(missing or unexpected or shape_mismatches),
        checkpoint_path=checkpoint_path.as_posix() if checkpoint_path else None,
        checkpoint_sha256=checkpoint_sha256(source_state, checkpoint_path),
        matched_keys=sorted(selected),
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatches=sorted(shape_mismatches),
        key_mapping=selected,
        matched_parameter_count=matched_parameters,
        total_parameter_count=total_parameters,
        matched_parameter_fraction=(matched_parameters / total_parameters if total_parameters else 0.0),
        newly_initialized_keys=sorted(key for key in missing if key.startswith(neck_prefix)),
    )


def build_resource_report(
    *,
    base_latency_ms: float,
    candidate_latency_ms: float,
    base_parameter_count: int,
    candidate_parameter_count: int,
    base_model_size_mb: float,
    candidate_model_size_mb: float,
    contract: FeaturePyramidContract,
    neck: ModelGraphPlugin,
    limits: ModelGraphResourceLimits,
    imgsz: int = 640,
) -> ModelGraphResourceReport:
    base_elements = sum(
        channels * (imgsz // stride) * (imgsz // stride)
        for stride, channels in zip(contract.strides, contract.channels, strict=True)
    )
    candidate_elements = base_elements + neck.estimated_intermediate_elements(imgsz=imgsz)
    bytes_per_element = 4
    denominator = 1024 * 1024
    return evaluate_resource_guards(
        base_latency_ms=base_latency_ms,
        candidate_latency_ms=candidate_latency_ms,
        base_vram_estimate_mb=base_elements * bytes_per_element / denominator,
        candidate_vram_estimate_mb=candidate_elements * bytes_per_element / denominator,
        base_parameter_count=base_parameter_count,
        candidate_parameter_count=candidate_parameter_count,
        base_model_size_mb=base_model_size_mb,
        candidate_model_size_mb=candidate_model_size_mb,
        limits=limits,
    )


def enforce_resource_report(report: ModelGraphResourceReport) -> None:
    """Raise with stable check names instead of allowing an over-budget graph."""
    if report.passed:
        return
    failed = sorted(name for name, passed in report.checks.items() if not passed)
    raise ModelGraphGuardError(f"model graph resource guards failed: {', '.join(failed)}")


def latency_ms(model: Any, config: YOLO26NeckConfig) -> float:
    """Measure a small same-device forward for a deterministic pre-training guard."""
    if torch is None:
        raise ImportError("latency audit requires torch")
    was_training = model.training
    model.eval()
    parameter = next(model.parameters())
    image = torch.zeros(
        1,
        3,
        config.audit_imgsz,
        config.audit_imgsz,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    with torch.no_grad():
        for _ in range(config.latency_warmup):
            model(image)
        timings: list[float] = []
        for _ in range(config.latency_iterations):
            if parameter.device.type == "cuda":
                torch.cuda.synchronize(parameter.device)
            started = time.perf_counter()
            model(image)
            if parameter.device.type == "cuda":
                torch.cuda.synchronize(parameter.device)
            timings.append((time.perf_counter() - started) * 1000.0)
    model.train(was_training)
    return float(median(timings))


def serialized_state_size_mb(state: dict[str, Any]) -> float:
    if torch is None:
        raise ImportError("checkpoint size audit requires torch")
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return len(buffer.getvalue()) / (1024 * 1024)


def checkpoint_path(model: Any) -> Path | None:
    raw = getattr(model, "pt_path", None)
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_file() else None


def checkpoint_sha256(state: dict[str, Any], path: Path | None) -> str:
    if path is not None:
        return sha256_path(path)
    if torch is None:
        raise ImportError("checkpoint hashing requires torch")
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def ranked_path(path: Path) -> Path:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))
    if rank in {-1, 0}:
        return path
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


__all__ = [
    "DetectWithFeaturePyramidNeck",
    "NeckKind",
    "YOLO26_NECK_STRIDES",
    "YOLO26NeckConfig",
    "YOLO26NeckManifest",
    "assert_native_yolo26_graph",
    "audit_partial_checkpoint",
    "build_feature_contract",
    "build_resource_report",
    "checkpoint_path",
    "enforce_resource_report",
    "infer_detect_channels",
    "latency_ms",
    "ranked_path",
    "serialized_state_size_mb",
    "sha256_path",
    "write_json_atomic",
]
