"""Ultralytics runtime bridge for isolated guarded YOLO26 neck components."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, ClassVar

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.neck.common import (
    DetectWithFeaturePyramidNeck,
    NeckKind,
    YOLO26NeckConfig,
    YOLO26NeckManifest,
    assert_native_yolo26_graph,
    audit_partial_checkpoint,
    build_resource_report,
    checkpoint_path,
    enforce_resource_report,
    latency_ms,
    ranked_path,
    serialized_state_size_mb,
    sha256_path,
    write_json_atomic,
)
from yolo_agent.components.adapters.neck.gold_gd import GoldGatherDistributeNeck
from yolo_agent.components.adapters.neck.multi_scale_fusion import MultiScaleFusionNeck
from yolo_agent.components.adapters.neck.rtmdet_large_kernel import RTMDetLargeKernelNeck
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload, RuntimePluginReference
from yolo_agent.components.model_graph import (
    ModelGraphDependencyGate,
    ModelGraphImplementationRequest,
    ModelGraphPlugin,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


_SMOKE_CACHE: dict[str, dict[str, bool | str]] = {}


def build_neck_component(
    kind: NeckKind,
    channels: list[int],
    config: YOLO26NeckConfig,
) -> ModelGraphPlugin:
    """Build exactly one component without importing another detector graph."""
    if kind == "multi_scale_fusion":
        return MultiScaleFusionNeck(channels, fusion_channels=config.context_channels)
    if kind == "gold_gather_distribute":
        return GoldGatherDistributeNeck(channels, context_channels=config.context_channels)
    if kind == "rtmdet_large_kernel":
        return RTMDetLargeKernelNeck(channels, kernel_size=config.kernel_size)
    raise ValueError(f"unsupported neck kind: {kind}")


class YOLO26NeckRuntimePlugin:
    """Insert one audited feature-pyramid plugin before native YOLO26 Detect."""

    plugin_version = "yolo26_neck_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = YOLO26NeckConfig.model_validate(options)
        self.manifest_path = Path(str(options["manifest_path"])).resolve()
        self.adapter_class = str(options["adapter_class"])
        self.adapter_version = str(options["adapter_version"])
        self.adapter_hash = str(options["adapter_hash"])

    def build_model(self, *, context: Any, trainer: Any, model: Any) -> Any:
        if torch is None:
            raise ImportError("YOLO26 neck runtime requires torch")
        dependency = ModelGraphDependencyGate.evaluate(
            component_id=self.config.component_id,
            deformable_module=self.config.deformable_module,
        )
        if not dependency.available:
            request = dependency.implementation_request
            raise RuntimeError(
                "implementation_request: "
                + (request.model_dump_json() if request is not None else "missing deformable op")
            )
        detect, channels = assert_native_yolo26_graph(model)
        if self.config.expected_channels and self.config.expected_channels != channels:
            raise ValueError(
                f"Detect channels {channels} do not match configured {self.config.expected_channels}"
            )
        source_state = dict(model.state_dict())
        source_checkpoint = checkpoint_path(model)
        base_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
        base_size = serialized_state_size_mb(source_state)
        base_latency = latency_ms(model, self.config)

        neck = build_neck_component(self.config.kind, channels, self.config)
        parameter = next(model.parameters())
        neck.to(device=parameter.device, dtype=parameter.dtype)
        wrapper = DetectWithFeaturePyramidNeck(detect, neck)
        detect_index = int(detect.i)
        model.model[-1] = wrapper
        target_state = dict(model.state_dict())
        checkpoint = audit_partial_checkpoint(
            source_state=source_state,
            target_state=target_state,
            detect_index=detect_index,
            checkpoint_path=source_checkpoint,
        )
        candidate_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
        candidate_size = serialized_state_size_mb(target_state)
        candidate_latency = latency_ms(model, self.config)
        resources = build_resource_report(
            base_latency_ms=base_latency,
            candidate_latency_ms=candidate_latency,
            base_parameter_count=base_parameters,
            candidate_parameter_count=candidate_parameters,
            base_model_size_mb=base_size,
            candidate_model_size_mb=candidate_size,
            contract=neck.input_contract,
            neck=neck,
            limits=self.config.resource_limits,
        )
        export_ok = _export_dry_run(model, self.config.audit_imgsz)
        manifest = YOLO26NeckManifest(
            component_id=self.config.component_id,
            neck_kind=self.config.kind,
            adapter_class=self.adapter_class,
            adapter_version=self.adapter_version,
            plugin_class=type(neck).__name__,
            plugin_version=neck.plugin_version,
            adapter_hash=self.adapter_hash,
            protocol_hash=context.payload.protocol_hash,
            paper_ids=list(neck.paper_ids),
            exact_paper_reproduction=neck.exact_paper_reproduction,
            insertion_point=neck.input_contract.insertion_point,
            input_strides=neck.input_contract.strides,
            input_channels=neck.input_contract.channels,
            output_strides=neck.output_contract.strides,
            output_channels=neck.output_contract.channels,
            native_end2end=bool(model.end2end),
            native_reg_max=int(wrapper.reg_max),
            dfl_disabled=type(wrapper.dfl).__name__ == "Identity",
            checkpoint=checkpoint,
            resources=resources,
            export_dry_run=export_ok,
        )
        write_json_atomic(ranked_path(self.manifest_path), manifest.model_dump(mode="json"))
        enforce_resource_report(resources)
        return model


class GuardedYOLO26NeckAdapter(ComponentAdapter):
    """Base SDK implementation shared by the three independent neck adapters."""

    adapter_version = "guarded_yolo26_neck.v1"
    source_commit = "yolo-agent:guarded-neck-runtime-v1"
    strategy = "custom_module"
    modified_model_fields = frozenset({"neck_plugin"})
    modified_training_fields = frozenset()
    component_id: ClassVar[str]
    neck_kind: ClassVar[NeckKind]

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        if torch is None:
            return AdapterValidationReport(ok=False, errors=["torch is required"])
        try:
            import ultralytics
        except ImportError:
            return AdapterValidationReport(ok=False, errors=["ultralytics is required"])
        config = self._config(context)
        dependency = ModelGraphDependencyGate.evaluate(
            component_id=self.component_id,
            deformable_module=config.deformable_module,
        )
        if not dependency.available:
            request = dependency.implementation_request
            return AdapterValidationReport(
                ok=False,
                errors=["implementation_request:deformable_dependency_missing"],
                checks={
                    "execution_class": "implementation_request",
                    "missing_dependency": request.missing_dependency if request else "unknown",
                },
            )
        return AdapterValidationReport(
            ok=True,
            checks={
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "execution_class": "runtime_candidate",
            },
        )

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        if context.imgsz != 640:
            return AdapterValidationReport(
                ok=False,
                errors=["multi-scale neck requires fixed imgsz=640"],
            )
        if context.detector_family != "yolo26":
            return AdapterValidationReport(
                ok=False,
                errors=["guarded neck runtime supports YOLO26 only"],
            )
        config = self._config(context)
        return AdapterValidationReport(
            ok=True,
            warnings=[
                "This adapter moves one isolated neck component, not a complete detector.",
                "Paper claims remain priors until matched local reproduction.",
            ],
            checks={
                "imgsz": config.imgsz,
                "input_strides": str(config.expected_strides),
                "insertion_point": config.insertion_point,
                "native_head_preserved": True,
            },
        )

    def patch_model_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        config["neck_plugin"] = self._config(context).model_dump(mode="json")
        return config

    def patch_training_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        return config

    def build_module(self, context: AdapterContext) -> ModelGraphPlugin:
        config = self._config(context)
        channels = config.expected_channels or [64, 128, 256]
        return build_neck_component(self.neck_kind, channels, config)

    def load_pretrained_weights(
        self,
        module: Any,
        weights: Path | str | None,
        context: AdapterContext,
    ) -> WeightLoadResult:
        path = Path(weights) if weights is not None else None
        return WeightLoadResult(
            loaded=False,
            source=path,
            message=(
                "Base YOLO26 weights are retained in place; runtime manifest records the "
                "wrapped Detect key mapping and newly initialized neck tensors."
            ),
        )

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        if torch is None:
            return SmokeTestResult(passed=False, errors=["torch is required"])
        try:
            config = self._config(context)
            key = hashlib.sha256(
                json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            checks = _SMOKE_CACHE.get(key)
            if checks is None:
                checks = _run_module_smoke(build_neck_component(
                    self.neck_kind,
                    config.expected_channels or [64, 128, 256],
                    config,
                ), config)
                _SMOKE_CACHE[key] = checks
            return SmokeTestResult(passed=True, checks=checks)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            return SmokeTestResult(passed=False, errors=[str(exc)])

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        return [
            ExpectedArtifact(
                name=f"{self.component_id}_manifest",
                relative_path=Path(f"{self.component_id.replace('.', '_')}_manifest.json"),
            )
        ]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(
            actions=["remove the pre-Detect neck wrapper and restore native Detect"],
            files_to_remove=[item.relative_path for item in self.expected_artifacts(context)],
        )

    def build_runtime_payload(
        self,
        context: AdapterContext,
        *,
        protocol_hash: str,
        base_command: list[str],
        generated_config: dict[str, Any],
    ) -> AdapterRuntimePayload:
        config = self._config(context)
        manifest = self.expected_artifacts(context)[0].relative_path
        options = config.model_dump(mode="json", exclude_none=True)
        options.update(
            {
                "manifest_path": str((context.workspace / manifest).resolve()),
                "adapter_class": type(self).__name__,
                "adapter_version": self.adapter_version,
                "adapter_hash": self._adapter_hash(),
            }
        )
        return AdapterRuntimePayload(
            component_ids=[context.contract.component_id],
            adapter_classes=[type(self).__name__],
            adapter_versions={context.contract.component_id: self.adapter_version},
            source_commits={context.contract.component_id: self.source_commit},
            model_graph_plugin=[RuntimePluginReference(
                reference=(
                    "yolo_agent.components.adapters.neck.runtime:YOLO26NeckRuntimePlugin"
                ),
                options=options,
            )],
            generated_config=generated_config,
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=True,
            supports_ddp=True,
            supports_resume=False,
        )

    def implementation_request(
        self,
        context: AdapterContext,
    ) -> ModelGraphImplementationRequest | None:
        config = self._config(context)
        return ModelGraphDependencyGate.evaluate(
            component_id=self.component_id,
            deformable_module=config.deformable_module,
        ).implementation_request

    def _config(self, context: AdapterContext) -> YOLO26NeckConfig:
        values = dict(context.options or {})
        values.update({"kind": self.neck_kind, "component_id": self.component_id})
        return YOLO26NeckConfig.model_validate(values)

    def _adapter_hash(self) -> str:
        source = inspect.getsourcefile(type(self))
        return sha256_path(Path(source)) if source else "unknown"


def _run_module_smoke(
    neck: ModelGraphPlugin,
    config: YOLO26NeckConfig,
) -> dict[str, bool | str]:
    channels = neck.input_contract.channels
    features = [
        torch.randn(
            1,
            channel,
            config.audit_imgsz // stride,
            config.audit_imgsz // stride,
            requires_grad=True,
        )
        for stride, channel in zip(neck.input_contract.strides, channels, strict=True)
    ]
    outputs = neck.forward(features)
    sum(value.float().mean() for value in outputs).backward()
    backward = any(parameter.grad is not None for parameter in neck.parameters())
    neck.zero_grad(set_to_none=True)
    amp_features = [value.detach() for value in features]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_outputs = neck.forward(amp_features)
    amp = all(value.shape == source.shape for value, source in zip(amp_outputs, features, strict=True))
    neck.eval()
    with torch.no_grad():
        exported_program = torch.export.export(neck, (tuple(amp_features),))
        exported = exported_program.module()(tuple(amp_features))
    export = all(value.shape == source.shape for value, source in zip(exported, features, strict=True))
    return {
        "shape": True,
        "backward": backward,
        "amp": amp,
        "export": export,
        "input_strides": str(neck.input_contract.strides),
        "input_channels": str(neck.input_contract.channels),
        "output_strides": str(neck.output_contract.strides),
        "output_channels": str(neck.output_contract.channels),
    }


def _export_dry_run(model: Any, imgsz: int) -> bool:
    # YOLO26 export keeps max_det constant. P3/P4/P5 at 64 produce fewer than
    # 300 anchors, so use the smallest 32-aligned audit input with >300 anchors.
    export_imgsz = max(imgsz, 160)
    was_training = model.training
    model.eval()
    detect = model.model[-1]
    previous_export = detect.export
    previous_format = detect.format
    detect.export = True
    detect.format = "torchscript"
    parameter = next(model.parameters())
    with torch.no_grad():
        output = model(torch.zeros(
            1,
            3,
            export_imgsz,
            export_imgsz,
            device=parameter.device,
            dtype=parameter.dtype,
        ))
    detect.export = previous_export
    detect.format = previous_format
    model.train(was_training)
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("neck export dry-run did not preserve native tensor output")
    return True


__all__ = [
    "GuardedYOLO26NeckAdapter",
    "YOLO26NeckRuntimePlugin",
    "build_neck_component",
]
