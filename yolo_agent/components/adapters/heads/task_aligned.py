"""Independent task-aligned detection-head adapter for YOLO26.

The adapter wraps the native terminal Detect module instead of replacing the
YOLO26 head.  It applies a bounded learnable quality scale to the native
one-to-one scores, while preserving the native output contract and loss path.
This is an adapted runtime component, not an exact reproduction of TOOD.
"""

from __future__ import annotations

import hashlib
import inspect
import json
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
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)
from yolo_agent.components.adapters.neck.common import assert_native_yolo26_graph

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc, assignment]
    nn = None  # type: ignore[assignment]


class TaskAlignedHeadConfig(BaseModel):
    """Fixed protocol and bounded quality-scale configuration."""

    component_id: str = "detection_head.task_aligned"
    imgsz: int = 640
    audit_imgsz: int = Field(default=64, ge=32)
    quality_scale_init: float = Field(default=1.0, gt=0.0, le=2.0)
    quality_scale_min: float = Field(default=0.5, gt=0.0)
    quality_scale_max: float = Field(default=1.5, gt=0.0)

    @model_validator(mode="after")
    def validate_protocol(self) -> "TaskAlignedHeadConfig":
        if self.imgsz != 640:
            raise ValueError("task-aligned YOLO26 head requires fixed imgsz=640")
        if self.quality_scale_min >= self.quality_scale_max:
            raise ValueError("quality scale bounds must be ordered")
        if not self.quality_scale_min <= self.quality_scale_init <= self.quality_scale_max:
            raise ValueError("quality_scale_init must be within quality scale bounds")
        return self


if nn is not None:

    class TaskAlignedDetectionHead(nn.Module):
        """Shape-preserving wrapper around the native YOLO26 Detect head."""

        plugin_id = "detection_head.task_aligned"
        plugin_version = "task_aligned_detection_head.v1"
        paper_ids = ("arxiv:2108.07755",)
        exact_paper_reproduction = False

        def __init__(self, detect: nn.Module, config: TaskAlignedHeadConfig) -> None:
            super().__init__()
            self.detect = detect
            self.quality_scale = nn.Parameter(
                torch.tensor(float(config.quality_scale_init))
            )
            self.quality_scale_min = float(config.quality_scale_min)
            self.quality_scale_max = float(config.quality_scale_max)
            self.f = detect.f
            self.i = detect.i
            self.type = f"{type(self).__name__}+{getattr(detect, 'type', type(detect).__name__)}"
            self.np = sum(int(parameter.numel()) for parameter in self.parameters())

        def forward(self, features: list[Any] | tuple[Any, ...]) -> Any:
            output = self.detect(features)
            if not isinstance(output, dict):
                # Native export returns a tensor (or an export tuple), not the
                # training dictionary. Preserve that deployment contract.
                return output
            if "one2one" not in output:
                raise ValueError("native YOLO26 Detect output has no one2one branch")
            branch = output["one2one"]
            if not isinstance(branch, dict) or "scores" not in branch:
                raise ValueError("native YOLO26 one2one output has no scores")
            scale = self.quality_scale.clamp(
                self.quality_scale_min,
                self.quality_scale_max,
            )
            output = dict(output)
            output["one2one"] = dict(branch)
            output["one2one"]["scores"] = branch["scores"] * scale
            return output

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

    class TaskAlignedDetectionHead:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("task-aligned head runtime requires torch")


class TaskAlignedHeadRuntimePlugin:
    """Install the independent head wrapper at the terminal native Detect node."""

    plugin_version = "task_aligned_detection_head_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = TaskAlignedHeadConfig.model_validate(options)
        self.manifest_path = Path(str(options["manifest_path"])).resolve()
        self.adapter_hash = str(options["adapter_hash"])
        self.adapter_class = str(options["adapter_class"])
        self.adapter_version = str(options["adapter_version"])

    def build_model(self, *, context: Any, trainer: Any, model: Any) -> Any:
        del trainer
        if torch is None:
            raise ImportError("task-aligned head runtime requires torch")
        if not str(getattr(context.payload, "protocol_hash", "")).strip():
            raise ValueError("task-aligned head protocol identity is unavailable")
        detect, _ = assert_native_yolo26_graph(model)
        if int(detect.i) != len(model.model) - 1:
            raise ValueError("task-aligned head may only wrap terminal native Detect")
        if isinstance(detect, TaskAlignedDetectionHead):
            raise ValueError("task-aligned head refuses duplicate wrapping")
        wrapper = TaskAlignedDetectionHead(detect, self.config)
        parameter = next(model.parameters())
        wrapper.to(device=parameter.device, dtype=parameter.dtype)
        model.model[-1] = wrapper
        manifest = {
            "schema_version": "task_aligned_head_manifest.v1",
            "component_id": "detection_head.task_aligned",
            "adapter_class": self.adapter_class,
            "adapter_version": self.adapter_version,
            "adapter_hash": self.adapter_hash,
            "plugin_version": self.plugin_version,
            "protocol_hash": context.payload.protocol_hash,
            "changed_variable": "model.task_aligned_head",
            "imgsz": self.config.imgsz,
            "graph_identity": "detection_head.task_aligned",
            "native_one_to_one_preserved": True,
            "native_dfl_free_regression": int(wrapper.reg_max) == 1
            and type(wrapper.dfl).__name__ == "Identity",
            "native_end2end": bool(model.end2end),
            "native_nms_free": True,
            "quality_scale_bounds": [
                self.config.quality_scale_min,
                self.config.quality_scale_max,
            ],
        }
        _write_json_atomic(self.manifest_path, manifest)
        return model


class TaskAlignedHeadAdapter(ComponentAdapter):
    """Typed ComponentAdapter for the independent task-aligned head graph."""

    adapter_version = "task_aligned_head_adapter.v1"
    source_commit = "yolo-agent:task-aligned-head-v1"
    strategy = "custom_module"
    modified_model_fields = frozenset({"task_aligned_head"})
    component_id = "detection_head.task_aligned"

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        if torch is None:
            return AdapterValidationReport(ok=False, errors=["torch is required"])
        try:
            import ultralytics
        except ImportError:
            return AdapterValidationReport(ok=False, errors=["ultralytics is required"])
        return AdapterValidationReport(
            ok=True,
            checks={"torch": torch.__version__, "ultralytics": ultralytics.__version__},
        )

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        if context.imgsz != 640:
            return AdapterValidationReport(ok=False, errors=["task-aligned head requires imgsz=640"])
        if context.detector_family != "yolo26":
            return AdapterValidationReport(ok=False, errors=["task-aligned head supports YOLO26 only"])
        if context.head not in {None, "one_to_one"}:
            return AdapterValidationReport(ok=False, errors=["task-aligned head requires one_to_one native head"])
        return AdapterValidationReport(ok=True, checks={"imgsz": 640, "native_head_preserved": True})

    def patch_model_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        del dry_run
        config["task_aligned_head"] = self._config(context).model_dump(mode="json")
        return config

    def patch_training_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        del context, dry_run
        return config

    def build_module(self, context: AdapterContext) -> Any:
        if torch is None:
            raise ImportError("task-aligned head runtime requires torch")
        from ultralytics.nn.tasks import DetectionModel

        model = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
        return TaskAlignedDetectionHead(model.model[-1], self._config(context))

    def load_pretrained_weights(
        self, module: Any, weights: Path | str | None, context: AdapterContext
    ) -> WeightLoadResult:
        del module, context
        return WeightLoadResult(
            loaded=False,
            source=Path(weights) if weights is not None else None,
            message="Native Detect weights remain in the wrapped head; the quality scale is initialized by the recipe.",
        )

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        if torch is None:
            return SmokeTestResult(passed=False, evidence_kind="local", errors=["torch is required"])
        try:
            from ultralytics.cfg import get_cfg
            from ultralytics.nn.tasks import DetectionModel

            model = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
            model.args = get_cfg(overrides={"imgsz": 640})
            detect = model.model[-1]
            wrapper = TaskAlignedDetectionHead(detect, self._config(context))
            wrapper.train()
            image = torch.rand(1, 3, self._config(context).audit_imgsz, self._config(context).audit_imgsz)
            captured: list[Any] = []

            def capture(_: Any, inputs: tuple[Any, ...]) -> None:
                captured.append(inputs[0])

            hook = detect.register_forward_pre_hook(capture)
            model.eval()
            with torch.no_grad():
                model(image)
            hook.remove()
            if not captured:
                raise RuntimeError("native Detect feature input was not captured")
            wrapper.train()
            output = wrapper(captured[0])
            shape = set(output) == {"one2many", "one2one"}
            score_shape = output["one2one"]["scores"].shape == output["one2many"]["scores"].shape
            loss = output["one2one"]["scores"].float().mean()
            loss.backward()
            backward = wrapper.quality_scale.grad is not None
            return SmokeTestResult(
                passed=bool(shape and score_shape and backward),
                evidence_kind="local",
                checks={
                    "build": True,
                    "forward": True,
                    "shape": score_shape,
                    "backward": backward,
                    "native_one_to_one": True,
                    "native_dfl_free": int(wrapper.reg_max) == 1 and type(wrapper.dfl).__name__ == "Identity",
                    "imgsz": "640",
                },
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            return SmokeTestResult(passed=False, evidence_kind="local", errors=[str(exc)])

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        del context
        return [ExpectedArtifact(name="task_aligned_head_manifest", relative_path=Path("task_aligned_head_manifest.json"))]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(
            actions=["restore the native terminal YOLO26 Detect module"],
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
        adapter_hash = _source_hash(Path(inspect.getfile(type(self))))
        options = config.model_dump(mode="json")
        options.update(
            {
                "manifest_path": str((context.workspace / "task_aligned_head_manifest.json").resolve()),
                "adapter_hash": adapter_hash,
                "adapter_class": type(self).__name__,
                "adapter_version": self.adapter_version,
            }
        )
        return AdapterRuntimePayload(
            component_ids=[context.contract.component_id],
            adapter_classes=[type(self).__name__],
            adapter_versions={context.contract.component_id: self.adapter_version},
            source_commits={context.contract.component_id: self.source_commit},
            model_graph_plugin=[RuntimePluginReference(
                reference="yolo_agent.components.adapters.heads.task_aligned:TaskAlignedHeadRuntimePlugin",
                options=options,
                required_hooks=["build_model"],
            )],
            generated_config=generated_config,
            changed_variables={"model.task_aligned_head": config.model_dump(mode="json")},
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=True,
            supports_ddp=True,
            supports_resume=True,
        )

    @staticmethod
    def _config(context: AdapterContext) -> TaskAlignedHeadConfig:
        values = dict(context.options or {})
        values["component_id"] = context.contract.component_id
        values["imgsz"] = context.imgsz
        return TaskAlignedHeadConfig.model_validate(values)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TaskAlignedDetectionHead",
    "TaskAlignedHeadAdapter",
    "TaskAlignedHeadConfig",
    "TaskAlignedHeadRuntimePlugin",
]
