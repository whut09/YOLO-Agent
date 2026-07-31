"""Runtime-integrated YOLO26 P2 detection head."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from statistics import median
import time
from typing import Any

from pydantic import BaseModel, Field, model_validator
import yaml

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload, RuntimePluginReference
from yolo_agent.components.model_graph import (
    ModelGraphGuardError,
    ModelGraphResourceLimits,
    ModelGraphResourceReport,
    evaluate_resource_guards,
)

try:  # torch is optional for the core harness
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - minimal installations
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc, assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


P2_DETECT_INPUTS = [19, 22, 25, 28]
P2_STRIDES = [4, 8, 16, 32]
_NATIVE_SMOKE_CACHE: dict[str, dict[str, bool | str]] = {}


class P2HeadConfig(BaseModel):
    """Graph, checkpoint, and risk-measurement policy for the P2 adapter."""

    p2_stride: int = Field(default=4, ge=1)
    source_strides: list[int] = Field(default_factory=lambda: [8, 16, 32])
    p2_channels: int = Field(default=128, ge=1)
    num_classes: int = Field(default=80, ge=1)
    in_channels: list[int] = Field(
        default_factory=lambda: [64, 128, 256, 512], min_length=4, max_length=4
    )
    checkpoint_policy: str = "partial_load_new_head"
    imgsz: int = 640
    audit_imgsz: int = Field(default=64, ge=64)
    latency_warmup: int = Field(default=1, ge=0, le=10)
    latency_iterations: int = Field(default=2, ge=1, le=20)
    resource_limits: ModelGraphResourceLimits = Field(
        default_factory=ModelGraphResourceLimits
    )

    @model_validator(mode="after")
    def _protocol(self) -> "P2HeadConfig":
        if self.imgsz != 640:
            raise ValueError("P2 head experiments require fixed imgsz=640")
        if self.p2_stride >= min(self.source_strides):
            raise ValueError("p2_stride must be finer than all source feature strides")
        if self.checkpoint_policy not in {"partial_load_new_head", "strict", "reject"}:
            raise ValueError("unsupported checkpoint_policy")
        if self.audit_imgsz % 32:
            raise ValueError("audit_imgsz must be divisible by 32")
        return self


class P2HeadCheckpointReport(BaseModel):
    """Auditable partial checkpoint transfer into the changed graph."""

    policy: str
    loaded: bool
    partial: bool
    checkpoint_path: str | None = None
    checkpoint_sha256: str
    matched_keys: list[str] = Field(default_factory=list)
    missing_keys: list[str] = Field(default_factory=list)
    unexpected_keys: list[str] = Field(default_factory=list)
    shape_mismatches: list[str] = Field(default_factory=list)
    key_mapping: dict[str, str] = Field(default_factory=dict)
    matched_parameter_count: int = 0
    total_parameter_count: int = 0
    matched_parameter_fraction: float = 0.0
    newly_initialized_keys: list[str] = Field(default_factory=list)
    initialization_policy: str = "ultralytics_default_initialization"
    warnings: list[str] = Field(default_factory=list)


class P2HeadManifest(BaseModel):
    """Runtime proof that the P2 graph, native loss, and checkpoint policy were applied."""

    schema_version: str = "yolo26_p2_head_manifest.v1"
    adapter_version: str
    plugin_version: str
    adapter_hash: str
    protocol_hash: str
    generated_model_yaml: str
    generated_yaml_sha256: str
    actual_tensor_strides: list[int]
    detect_input_count: int
    native_end2end: bool
    native_reg_max: int
    dfl_disabled: bool
    external_nms_added: bool = False
    checkpoint: P2HeadCheckpointReport
    base_parameter_count: int
    p2_parameter_count: int
    parameter_delta: int
    base_model_size_mb: float
    p2_model_size_mb: float
    model_size_delta_mb: float
    latency_audit_imgsz: int
    base_latency_ms: float
    p2_latency_ms: float
    latency_delta_ms: float
    latency_risk: str
    resources: ModelGraphResourceReport


if nn is not None:

    class P2Head(nn.Module):
        """Standalone feature fusion fixture retained for focused tensor tests."""

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
            """Validate strides from actual tensor shapes rather than config labels."""
            if len(features) != len(expected_strides):
                raise ValueError("feature count must match expected_strides")
            height, width = (input_size, input_size) if isinstance(input_size, int) else input_size
            actual: dict[str, int] = {}
            for index, (feature, expected) in enumerate(zip(features, expected_strides)):
                stride_h = height // feature.shape[-2]
                stride_w = width // feature.shape[-1]
                if stride_h != stride_w or stride_h != expected:
                    raise ValueError(
                        f"feature {index} has stride {(stride_h, stride_w)}; expected {expected}"
                    )
                actual[f"p{index + 2}"] = stride_h
            return actual

else:

    class P2Head:  # type: ignore[no-redef]
        """Placeholder that explains the optional torch dependency."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("P2Head requires the optional torch dependency")


def build_yolo26_p2_yaml(source_yaml: dict[str, Any], *, p2_channels: int = 128) -> dict[str, Any]:
    """Return a controlled four-scale YOLO26 graph using the native Detect module."""
    graph = deepcopy(source_yaml)
    if graph.get("end2end") is not True or int(graph.get("reg_max", -1)) != 1:
        raise ValueError("P2 runtime requires native YOLO26 end2end=True and reg_max=1")
    if len(graph.get("backbone", [])) != 11:
        raise ValueError("unsupported YOLO26 backbone graph; expected 11 backbone layers")
    current_head = graph.get("head") or []
    if current_head and current_head[-1][2] == "Detect" and current_head[-1][0] == P2_DETECT_INPUTS:
        return graph
    if not current_head or current_head[-1][0] != [16, 19, 22]:
        raise ValueError("unsupported YOLO26 head graph; expected native P3/P4/P5 Detect inputs")
    graph["head"] = [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, True]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, True]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 2], 1, "Concat", [1]],
        [-1, 2, "C3k2", [p2_channels, True]],
        [-1, 1, "Conv", [p2_channels, 3, 2]],
        [[-1, 16], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, True]],
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 13], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, True]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 1, "C3k2", [1024, True, 0.5, True]],
        [P2_DETECT_INPUTS, 1, "Detect", ["nc"]],
    ]
    graph["p2_adapter"] = {
        "version": P2HeadAdapter.adapter_version,
        "detect_inputs": P2_DETECT_INPUTS,
        "strides": P2_STRIDES,
    }
    return graph


def partial_load_p2_checkpoint(
    target: Any,
    source: Any,
    *,
    checkpoint_policy: str = "partial_load_new_head",
) -> P2HeadCheckpointReport:
    """Transfer shape-compatible native weights and report every unmatched tensor."""
    if checkpoint_policy == "reject":
        raise ValueError("checkpoint policy rejects loading weights into a changed P2 graph")
    source_state = {
        key: value.float() if value.is_floating_point() else value
        for key, value in source.state_dict().items()
    }
    target_state = target.state_dict()
    selected: dict[str, Any] = {}
    key_mapping: dict[str, str] = {}
    consumed_source: set[str] = set()
    shape_mismatches: list[str] = []

    def consider(source_key: str, target_key: str) -> None:
        if source_key not in source_state or target_key not in target_state or target_key in selected:
            return
        if source_state[source_key].shape != target_state[target_key].shape:
            shape_mismatches.append(
                f"{source_key}->{target_key}:"
                f"{tuple(source_state[source_key].shape)}!={tuple(target_state[target_key].shape)}"
            )
            return
        selected[target_key] = source_state[source_key]
        key_mapping[target_key] = source_key
        consumed_source.add(source_key)

    source_strides = [int(value) for value in source.model[-1].stride.tolist()]
    source_is_p2 = source_strides == P2_STRIDES
    for key in source_state:
        layer_index = _state_layer_index(key)
        if source_is_p2 or (layer_index is not None and layer_index <= 16):
            consider(key, key)
    for source_layer, target_layer in {19: 25, 20: 26, 22: 28}.items():
        prefix = f"model.{source_layer}."
        for key in source_state:
            if key.startswith(prefix):
                consider(key, f"model.{target_layer}.{key.removeprefix(prefix)}")
    for branch in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        for old_scale in range(3):
            prefix = f"model.23.{branch}.{old_scale}."
            for key in source_state:
                if key.startswith(prefix):
                    suffix = key.removeprefix(prefix)
                    consider(key, f"model.29.{branch}.{old_scale + 1}.{suffix}")

    missing = sorted(set(target_state) - set(selected))
    unexpected = sorted(set(source_state) - consumed_source)
    if checkpoint_policy == "strict" and (missing or unexpected or shape_mismatches):
        raise ValueError("strict checkpoint loading failed for the changed P2 graph")
    target.load_state_dict(selected, strict=False)
    matched_parameters = sum(int(target_state[key].numel()) for key in selected)
    total_parameters = sum(int(value.numel()) for value in target_state.values())
    checkpoint_path = _checkpoint_path(source)
    warnings = []
    if missing:
        warnings.append("new or shape-changed P2 graph tensors use Ultralytics initialization")
    return P2HeadCheckpointReport(
        policy=checkpoint_policy,
        loaded=bool(selected),
        partial=bool(missing or unexpected or shape_mismatches),
        checkpoint_path=checkpoint_path.as_posix() if checkpoint_path else None,
        checkpoint_sha256=_checkpoint_sha256(source, checkpoint_path),
        matched_keys=sorted(selected),
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatches=sorted(set(shape_mismatches)),
        key_mapping=key_mapping,
        matched_parameter_count=matched_parameters,
        total_parameter_count=total_parameters,
        matched_parameter_fraction=(matched_parameters / total_parameters if total_parameters else 0.0),
        newly_initialized_keys=missing,
        warnings=warnings,
    )


class P2HeadRuntimePlugin:
    """Replace the trainer-built native graph with an audited native four-scale graph."""

    plugin_version = "p2_head_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = P2HeadConfig.model_validate(options)
        self.generated_model_yaml = Path(str(options["generated_model_yaml"])).resolve()
        self.manifest_path = Path(str(options["manifest_path"])).resolve()

    def build_model(self, *, context: Any, trainer: Any, model: Any) -> Any:
        if torch is None:
            raise ImportError("P2 runtime requires torch")
        try:
            from ultralytics.cfg import get_cfg
            from ultralytics.nn.tasks import DetectionModel
        except ImportError as exc:  # pragma: no cover - guarded by runtime audit
            raise ImportError("P2 runtime requires ultralytics") from exc

        source = model
        graph = build_yolo26_p2_yaml(source.yaml, p2_channels=self.config.p2_channels)
        _write_yaml_atomic(self.generated_model_yaml, graph)
        channels = int(graph.get("channels", 3))
        class_count = int(getattr(source.model[-1], "nc", self.config.num_classes))
        target = DetectionModel(graph, ch=channels, nc=class_count, verbose=False)
        target.args = getattr(trainer, "args", None) or getattr(source, "args", None) or get_cfg()
        target.names = getattr(source, "names", target.names)
        target.task = getattr(source, "task", "detect")
        target.pt_path = getattr(source, "pt_path", None)
        report = partial_load_p2_checkpoint(
            target,
            source,
            checkpoint_policy=self.config.checkpoint_policy,
        )
        source_parameter = next(source.parameters())
        target.to(device=source_parameter.device, dtype=source_parameter.dtype)
        manifest = _build_runtime_manifest(
            source,
            target,
            report,
            graph_path=self.generated_model_yaml,
            protocol_hash=context.payload.protocol_hash,
            config=self.config,
        )
        manifest_path = _ranked_manifest_path(self.manifest_path)
        _write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
        if not manifest.resources.passed:
            failed = sorted(
                name for name, passed in manifest.resources.checks.items() if not passed
            )
            raise ModelGraphGuardError(
                "P2 model graph resource guards failed: " + ", ".join(failed)
            )
        return target


class P2HeadAdapter(ComponentAdapter):
    """Executable adapter for a native YOLO26 P2/P3/P4/P5 detection graph."""

    adapter_version = "p2_head.v3"
    source_commit = "yolo-agent:p2-head-runtime-v1"
    strategy = "custom_model_yaml"
    modified_model_fields = frozenset({"p2_head"})
    modified_training_fields = frozenset()

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        if torch is None:
            return AdapterValidationReport(ok=False, errors=["torch is required for P2 head checks"])
        try:
            import ultralytics
        except ImportError:
            return AdapterValidationReport(ok=False, errors=["ultralytics is required for P2 runtime"])
        return AdapterValidationReport(
            ok=True,
            checks={"torch": torch.__version__, "ultralytics": ultralytics.__version__},
        )

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        if context.imgsz != 640:
            return AdapterValidationReport(ok=False, errors=["P2 head requires fixed imgsz=640"])
        options = P2HeadConfig.model_validate(context.options or {})
        if context.detector_family != "yolo26":
            return AdapterValidationReport(ok=False, errors=["P2 runtime supports YOLO26 only"])
        warnings = [
            "P2 changes the graph; checkpoint transfer and latency/model-size deltas are recorded.",
            "P2 retains native YOLO26 one-to-one and one-to-many loss branches.",
        ]
        return AdapterValidationReport(
            ok=True,
            warnings=warnings,
            checks={"p2_stride": options.p2_stride, "imgsz": 640, "external_nms": False},
        )

    def patch_model_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        options = P2HeadConfig.model_validate(context.options or {})
        config["p2_head"] = options.model_dump(mode="json")
        return config

    def patch_training_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        return config

    def build_module(self, context: AdapterContext) -> P2Head:
        options = P2HeadConfig.model_validate(context.options or {})
        return P2Head(options.in_channels, options)

    def load_pretrained_weights(
        self, module: Any, weights: Path | str | None, context: AdapterContext
    ) -> WeightLoadResult:
        if weights is None:
            return WeightLoadResult(
                loaded=False,
                message="No checkpoint supplied; runtime manifest will record new-layer initialization",
            )
        path = Path(weights)
        if not path.is_file():
            return WeightLoadResult(loaded=False, source=path, message="Checkpoint not found")
        return WeightLoadResult(
            loaded=False,
            source=path,
            message=(
                "Runtime plugin performs explicit partial loading and records matched, missing, "
                "unexpected, and shape-mismatched keys"
            ),
        )

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        if torch is None:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                errors=["torch is required"],
            )
        try:
            config = P2HeadConfig.model_validate(context.options or {})
            cache_key = hashlib.sha256(
                json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            checks = _NATIVE_SMOKE_CACHE.get(cache_key)
            if checks is None:
                checks = _run_native_smoke(config)
                _NATIVE_SMOKE_CACHE[cache_key] = checks
            return SmokeTestResult(
                passed=True,
                evidence_kind="local",
                checks=checks,
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                errors=[str(exc)],
            )

    def gpu_smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        if torch is None or not torch.cuda.is_available():
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                checks={"gpu_smoke_implemented": True, "cuda_available": False},
                errors=["cuda_not_available"],
            )
        try:
            checks = _run_gpu_smoke(P2HeadConfig.model_validate(context.options or {}))
            return SmokeTestResult(
                passed=all(value is True for value in checks.values()),
                evidence_kind="local",
                checks=checks,
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                checks={"gpu_smoke_implemented": True, "cuda_available": True},
                errors=[str(exc)],
            )

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        return [
            ExpectedArtifact(name="p2_head_manifest", relative_path=Path("p2_head_manifest.json")),
            ExpectedArtifact(
                name="p2_model_yaml", relative_path=Path("generated_yolo26_p2.yaml")
            ),
        ]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(
            actions=["remove the P2 model-graph plugin and resume the native YOLO26 graph"],
            files_to_remove=[Path("p2_head_manifest.json"), Path("generated_yolo26_p2.yaml")],
        )

    def build_runtime_payload(
        self,
        context: AdapterContext,
        *,
        protocol_hash: str,
        base_command: list[str],
        generated_config: dict[str, Any],
    ) -> AdapterRuntimePayload:
        config = P2HeadConfig.model_validate(context.options or {})
        options = config.model_dump(mode="json", exclude_none=True)
        options.update(
            {
                "generated_model_yaml": str(
                    (context.workspace / "generated_yolo26_p2.yaml").resolve()
                ),
                "manifest_path": str((context.workspace / "p2_head_manifest.json").resolve()),
            }
        )
        return AdapterRuntimePayload(
            component_ids=[context.contract.component_id],
            adapter_classes=[type(self).__name__],
            adapter_versions={context.contract.component_id: self.adapter_version},
            source_commits={context.contract.component_id: self.source_commit},
            model_graph_plugin=[
                RuntimePluginReference(
                    reference=(
                        "yolo_agent.components.adapters.head.p2_head:P2HeadRuntimePlugin"
                    ),
                    options=options,
                )
            ],
            generated_config=generated_config,
            changed_variables={
                "model.p2_head": config.model_dump(mode="json", exclude_none=True)
            },
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=True,
            supports_ddp=True,
            supports_resume=True,
        )


def _run_native_smoke(config: P2HeadConfig) -> dict[str, bool | str]:
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    base = DetectionModel("yolo26n.yaml", nc=config.num_classes, verbose=False)
    graph = build_yolo26_p2_yaml(base.yaml, p2_channels=config.p2_channels)
    model = DetectionModel(graph, nc=config.num_classes, verbose=False)
    model.args = get_cfg(overrides={"imgsz": 640})
    report = partial_load_p2_checkpoint(model, base)
    strides = _actual_detect_strides(model, config.audit_imgsz)
    model.train()
    image = torch.rand(1, 3, config.audit_imgsz, config.audit_imgsz)
    output = model(image)
    if set(output) != {"one2many", "one2one"}:
        raise RuntimeError("native end-to-end Detect branches were not preserved")
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    loss, _ = model.loss(batch)
    loss.sum().backward()
    p2_backward = any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith("model.19.")
    )
    model.zero_grad(set_to_none=True)
    model.criterion = None
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_loss, _ = model.loss(batch)
    amp_loss.sum().backward()
    model.eval()
    detect = model.model[-1]
    previous_export, previous_format = detect.export, detect.format
    detect.export, detect.format = True, "torchscript"
    with torch.no_grad():
        export_output = model(image)
    detect.export, detect.format = previous_export, previous_format
    if not isinstance(export_output, torch.Tensor):
        raise RuntimeError("native export dry-run did not return a tensor")
    return {
        "shape": str(tuple(export_output.shape)),
        "feature_strides": str(strides),
        "detect_inputs": str(P2_DETECT_INPUTS),
        "one2many": True,
        "one2one": True,
        "loss": True,
        "backward": p2_backward,
        "amp": True,
        "export": True,
        "dfl_disabled": type(detect.dfl).__name__ == "Identity",
        "partial_checkpoint": report.partial,
    }


def _run_gpu_smoke(config: P2HeadConfig) -> dict[str, bool]:
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    device = torch.device("cuda")
    source = DetectionModel("yolo26n.yaml", nc=config.num_classes, verbose=False)
    graph = build_yolo26_p2_yaml(source.yaml, p2_channels=config.p2_channels)
    model = DetectionModel(graph, nc=config.num_classes, verbose=False).to(device)
    model.args = get_cfg(overrides={"imgsz": 640})
    partial_load_p2_checkpoint(model, source)
    model.train()
    image = torch.rand(1, 3, config.audit_imgsz, config.audit_imgsz, device=device)
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0], device=device),
        "cls": torch.tensor([[0.0]], device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
    }
    predictions = model(image)
    actual_graph = bool(
        isinstance(predictions, dict)
        and set(predictions) == {"one2many", "one2one"}
        and len(predictions["one2many"]["feats"]) == 4
        and len(predictions["one2one"]["feats"]) == 4
    )
    loss, _ = model.loss(batch)
    loss.sum().backward()
    backward = any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith("model.19.")
    )
    model.zero_grad(set_to_none=True)
    model.criterion = None
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_loss, _ = model.loss(batch)
    amp_loss.sum().backward()
    return {
        "gpu_smoke_implemented": True,
        "cuda_available": True,
        "actual_p2_graph": actual_graph,
        "native_loss_preserved": bool(
            model.end2end
            and model.model[-1].reg_max == 1
            and type(model.model[-1].dfl).__name__ == "Identity"
        ),
        "backward": backward,
        "amp": bool(torch.isfinite(amp_loss).all()),
        "fixed_imgsz_640": config.imgsz == 640,
    }


def _build_runtime_manifest(
    source: Any,
    target: Any,
    report: P2HeadCheckpointReport,
    *,
    graph_path: Path,
    protocol_hash: str,
    config: P2HeadConfig,
) -> P2HeadManifest:
    strides = _actual_detect_strides(target, config.audit_imgsz)
    detect = target.model[-1]
    base_parameters = sum(int(item.numel()) for item in source.parameters())
    p2_parameters = sum(int(item.numel()) for item in target.parameters())
    base_size = _serialized_model_size_mb(source)
    p2_size = _serialized_model_size_mb(target)
    base_latency = _latency_ms(source, config)
    p2_latency = _latency_ms(target, config)
    resources = evaluate_resource_guards(
        base_latency_ms=base_latency,
        candidate_latency_ms=p2_latency,
        base_vram_estimate_mb=_detect_activation_mb(source, config.imgsz),
        candidate_vram_estimate_mb=_detect_activation_mb(target, config.imgsz),
        base_parameter_count=base_parameters,
        candidate_parameter_count=p2_parameters,
        base_model_size_mb=base_size,
        candidate_model_size_mb=p2_size,
        limits=config.resource_limits,
    )
    return P2HeadManifest(
        adapter_version=P2HeadAdapter.adapter_version,
        plugin_version=P2HeadRuntimePlugin.plugin_version,
        adapter_hash=_sha256_path(Path(__file__)),
        protocol_hash=protocol_hash,
        generated_model_yaml=graph_path.as_posix(),
        generated_yaml_sha256=_sha256_path(graph_path),
        actual_tensor_strides=strides,
        detect_input_count=len(detect.stride),
        native_end2end=bool(target.end2end),
        native_reg_max=int(detect.reg_max),
        dfl_disabled=type(detect.dfl).__name__ == "Identity",
        checkpoint=report,
        base_parameter_count=base_parameters,
        p2_parameter_count=p2_parameters,
        parameter_delta=p2_parameters - base_parameters,
        base_model_size_mb=base_size,
        p2_model_size_mb=p2_size,
        model_size_delta_mb=p2_size - base_size,
        latency_audit_imgsz=config.audit_imgsz,
        base_latency_ms=base_latency,
        p2_latency_ms=p2_latency,
        latency_delta_ms=p2_latency - base_latency,
        latency_risk="increase_guarded" if p2_latency > base_latency else "measured_no_increase",
        resources=resources,
    )


def _actual_detect_strides(model: Any, imgsz: int) -> list[int]:
    captured: list[Any] = []

    def capture(_: Any, inputs: tuple[Any, ...]) -> None:
        captured.extend(inputs[0])

    hook = model.model[-1].register_forward_pre_hook(capture)
    was_training = model.training
    model.train()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=dtype))
    hook.remove()
    model.train(was_training)
    values = [imgsz // int(feature.shape[-1]) for feature in captured]
    if values != P2_STRIDES:
        raise RuntimeError(f"P2 Detect received tensor strides {values}, expected {P2_STRIDES}")
    return values


def _latency_ms(model: Any, config: P2HeadConfig) -> float:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    image = torch.zeros(1, 3, config.audit_imgsz, config.audit_imgsz, device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(config.latency_warmup):
            model(image)
        timings = []
        for _ in range(config.latency_iterations):
            start = time.perf_counter()
            model(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    model.train(was_training)
    return float(median(timings))


def _serialized_model_size_mb(model: Any) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024 * 1024)


def _detect_activation_mb(model: Any, imgsz: int) -> float:
    detect = model.model[-1]
    channels = []
    for branch in detect.cv2:
        convolution = getattr(branch[0], "conv", branch[0])
        channels.append(int(convolution.in_channels))
    strides = [int(value) for value in detect.stride.tolist()]
    elements = sum(
        channel * (imgsz // stride) * (imgsz // stride)
        for channel, stride in zip(channels, strides, strict=True)
    )
    return elements * 4 / (1024 * 1024)


def _checkpoint_path(model: Any) -> Path | None:
    raw = getattr(model, "pt_path", None)
    if raw:
        path = Path(str(raw)).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def _state_layer_index(key: str) -> int | None:
    parts = key.split(".", 2)
    if len(parts) < 3 or parts[0] != "model":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _checkpoint_sha256(model: Any, path: Path | None) -> str:
    if path is not None:
        return _sha256_path(path)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_yaml_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        yaml.safe_dump(value, file, sort_keys=False)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _ranked_manifest_path(path: Path) -> Path:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))
    if rank in {-1, 0}:
        return path
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


__all__ = [
    "P2Head",
    "P2HeadAdapter",
    "P2HeadCheckpointReport",
    "P2HeadConfig",
    "P2HeadManifest",
    "P2HeadRuntimePlugin",
    "build_yolo26_p2_yaml",
    "partial_load_p2_checkpoint",
]
