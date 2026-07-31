"""Local Ultralytics trainer subclass that dispatches component plugin hooks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MethodType
from typing import Any

from yolo_agent.adapters.ultralytics.plugin_context import (
    PluginRuntimeEvidence,
    RuntimePluginDescriptor,
    UltralyticsPluginContext,
    audit_installed_ultralytics,
    plugin_source_hash,
    runtime_evidence_path,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload

try:
    from ultralytics.models.yolo.detect.train import DetectionTrainer as _DetectionTrainer
except ImportError:  # pragma: no cover - optional train dependency
    class _DetectionTrainer:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("PluginDetectionTrainer requires ultralytics")


TRAINING_HOOKS = {
    "build_model",
    "build_train_dataset",
    "build_train_dataloader",
    "build_validator",
    "build_criterion",
    "compute_loss",
    "on_train_batch_start",
    "on_train_batch_end",
    "on_checkpoint_save",
    "on_checkpoint_load",
}
TRANSFORM_ARGUMENT = {
    "build_model": "model",
    "build_train_dataset": "dataset",
    "build_train_dataloader": "dataloader",
    "build_validator": "validator",
    "build_criterion": "criterion",
    "compute_loss": "loss_output",
}


class PluginExecutionError(RuntimeError):
    """Raised when a plugin fails; the bridge never falls back to plain training."""


class _LoadedPlugin:
    def __init__(self, reference: str, instance: Any, descriptor: RuntimePluginDescriptor) -> None:
        self.reference = reference
        self.instance = instance
        self.descriptor = descriptor


class PluginCriterionWrapper:
    """Call the native criterion, then dispatch the typed compute_loss hook."""

    def __init__(
        self,
        criterion: Any,
        bridge: "UltralyticsTrainerPluginBridge",
        model: Any,
        trainer: Any,
    ) -> None:
        self.criterion = criterion
        self.bridge = bridge
        self.model = model
        self.trainer = trainer

    def __call__(self, predictions: Any, batch: Any) -> Any:
        output = self.criterion(predictions, batch)
        return self.bridge.invoke_transform(
            "compute_loss",
            output,
            trainer=self.trainer,
            model=self.model,
            criterion=self.criterion,
            predictions=predictions,
            batch=batch,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.criterion, name)

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        """Keep training-only bridge state out of inference checkpoints."""
        import copy

        return copy.deepcopy(self.criterion, memo)


class UltralyticsTrainerPluginBridge:
    """Load, validate, execute, and account for component runtime plugins."""

    def __init__(self, payload_path: Path | str) -> None:
        self.payload_path = Path(payload_path).resolve()
        self.payload = AdapterRuntimePayload.read(self.payload_path, verify_imports=True)
        self.audit = audit_installed_ultralytics()
        if not self.audit.compatible:
            self._persist_initial_failure("; ".join(self.audit.blocked_by))
            raise PluginExecutionError("; ".join(self.audit.blocked_by))
        try:
            self.plugins = self._load_plugins()
        except PluginExecutionError as exc:
            self._persist_initial_failure(str(exc))
            raise
        if not any(set(item.descriptor.hooks) & TRAINING_HOOKS for item in self.plugins):
            self._persist_initial_failure("runtime payload has no executable trainer hooks")
            raise PluginExecutionError("runtime payload has no executable trainer hooks")
        evidence_path = runtime_evidence_path(self.payload_path)
        evidence = self._load_or_create_evidence(evidence_path)
        self.context = UltralyticsPluginContext(
            payload=self.payload,
            payload_path=self.payload_path,
            audit=self.audit,
            evidence_path=evidence_path,
            evidence=evidence,
        )
        self.context.persist()

    def validate_training_args(self, arguments: dict[str, Any]) -> None:
        """Enforce the fixed comparison protocol before Trainer construction."""
        imgsz = arguments.get("imgsz")
        if imgsz != 640:
            raise PluginExecutionError(f"trainer plugins require fixed imgsz=640, got {imgsz!r}")
        multi_scale = arguments.get("multi_scale", 0.0)
        if float(multi_scale or 0.0) != 0.0:
            raise PluginExecutionError("trainer plugins reject multi_scale because imgsz must remain fixed at 640")
        if bool(arguments.get("amp", True)) and not self.payload.supports_amp:
            raise PluginExecutionError("runtime payload does not support AMP")
        if arguments.get("resume") not in {None, False, "False", "false", ""} and not self.payload.supports_resume:
            raise PluginExecutionError("runtime payload does not support checkpoint resume")
        device = arguments.get("device")
        multi_device = (
            isinstance(device, (list, tuple)) and len(device) > 1
        ) or (
            isinstance(device, str) and "," in device
        )
        if multi_device and not self.payload.supports_ddp:
            raise PluginExecutionError("runtime payload does not support DDP")

    def prepare_command(
        self,
        command: list[str],
        env: dict[str, str],
    ) -> tuple[list[str], dict[str, str]]:
        """Run optional command preparation hooks before parsing train arguments."""
        current_command = list(command)
        current_env = dict(env)
        for plugin in self.plugins:
            method = getattr(plugin.instance, "prepare_command", None)
            if not callable(method):
                continue
            self.context.record_call(plugin.reference, "prepare_command")
            try:
                result = method(
                    payload=self.payload,
                    command=current_command,
                    env=current_env,
                )
            except Exception as exc:
                self.context.record_failure(plugin.reference, "prepare_command", exc)
                raise PluginExecutionError(
                    f"plugin hook failed: {plugin.reference}:prepare_command: {exc}"
                ) from exc
            if result is not None:
                current_command, current_env = result
        return current_command, current_env

    def invoke_transform(self, hook: str, value: Any, **kwargs: Any) -> Any:
        """Apply transform hooks sequentially and fail closed on any exception."""
        argument = TRANSFORM_ARGUMENT[hook]
        current = value
        for plugin in self.plugins:
            method = getattr(plugin.instance, hook, None)
            if not callable(method):
                continue
            self.context.record_call(plugin.reference, hook)
            try:
                result = method(context=self.context, **kwargs, **{argument: current})
            except Exception as exc:
                self.context.record_failure(plugin.reference, hook, exc)
                raise PluginExecutionError(f"plugin hook failed: {plugin.reference}:{hook}: {exc}") from exc
            if result is not None:
                current = result
        return current

    def invoke_event(self, hook: str, **kwargs: Any) -> None:
        """Invoke event hooks without suppressing plugin failures."""
        for plugin in self.plugins:
            method = getattr(plugin.instance, hook, None)
            if not callable(method):
                continue
            self.context.record_call(plugin.reference, hook)
            try:
                method(context=self.context, **kwargs)
            except Exception as exc:
                self.context.record_failure(plugin.reference, hook, exc)
                raise PluginExecutionError(f"plugin hook failed: {plugin.reference}:{hook}: {exc}") from exc

    def install_model_hooks(self, model: Any, *, trainer: Any) -> Any:
        """Install criterion hooks on this model instance only."""
        model = self.invoke_transform("build_model", model, trainer=trainer)
        if not self.has_hook("build_criterion") and not self.has_hook("compute_loss"):
            return model
        original_init = getattr(model, "init_criterion", None)
        if not callable(original_init):
            raise PluginExecutionError("model has no init_criterion required by loss plugins")

        def init_criterion(instance: Any) -> PluginCriterionWrapper:
            criterion = original_init()
            criterion = self.invoke_transform(
                "build_criterion",
                criterion,
                trainer=trainer,
                model=instance,
            )
            return PluginCriterionWrapper(criterion, self, instance, trainer)

        model.init_criterion = MethodType(init_criterion, model)
        existing = getattr(model, "criterion", None)
        if existing is not None:
            criterion = self.invoke_transform(
                "build_criterion",
                existing,
                trainer=trainer,
                model=model,
            )
            model.criterion = PluginCriterionWrapper(criterion, self, model, trainer)
        return model

    def has_hook(self, hook: str) -> bool:
        return any(hook in item.descriptor.hooks for item in self.plugins)

    def verify_required_hooks(self) -> None:
        """Fail unless every plugin's efficacy hook ran in this payload execution."""
        try:
            evidence = PluginRuntimeEvidence.model_validate_json(
                self.context.evidence_path.read_text(encoding="utf-8-sig")
            )
        except (OSError, ValueError) as exc:
            raise PluginExecutionError("runtime hook evidence is unavailable") from exc
        if (
            evidence.payload_hash != self.payload.payload_hash
            or evidence.protocol_hash != self.payload.protocol_hash
            or evidence.component_ids != self.payload.component_ids
            or evidence.changed_variables != self.payload.changed_variables
        ):
            raise PluginExecutionError("runtime hook evidence identity does not match payload")
        missing: list[str] = []
        for reference in self.payload.plugin_references:
            counts = evidence.hook_call_counts.get(reference.reference, {})
            missing.extend(
                f"{reference.reference}:{hook}"
                for hook in reference.required_hooks
                if counts.get(hook, 0) < 1
            )
        if missing:
            message = "required runtime hooks were not called: " + ", ".join(missing)
            self.context.record_failure("runtime_entrypoint", "required_hooks", message)
            raise PluginExecutionError(message)

    def _load_plugins(self) -> list[_LoadedPlugin]:
        loaded: list[_LoadedPlugin] = []
        seen: set[str] = set()
        for reference in self.payload.plugin_references:
            identity = json.dumps(reference.model_dump(mode="json"), sort_keys=True)
            if identity in seen:
                continue
            seen.add(identity)
            implementation = reference.resolve()
            version = str(getattr(implementation, "plugin_version", "")).strip()
            if not version:
                raise PluginExecutionError(
                    f"runtime plugin must declare plugin_version: {reference.reference}"
                )
            try:
                instance = (
                    implementation(**reference.options)
                    if isinstance(implementation, type)
                    else implementation
                )
            except Exception as exc:
                raise PluginExecutionError(
                    f"runtime plugin failed to load: {reference.reference}: {exc}"
                ) from exc
            hooks = sorted(
                hook for hook in TRAINING_HOOKS | {"prepare_command"}
                if callable(getattr(instance, hook, None))
            )
            missing_required = sorted(set(reference.required_hooks) - set(hooks))
            if missing_required:
                raise PluginExecutionError(
                    f"runtime plugin is missing required hooks: {reference.reference}:"
                    f"{','.join(missing_required)}"
                )
            descriptor = RuntimePluginDescriptor(
                reference=reference.reference,
                class_name=getattr(implementation, "__qualname__", type(instance).__qualname__),
                module=getattr(implementation, "__module__", type(instance).__module__),
                version=version,
                source_hash=plugin_source_hash(implementation),
                hooks=hooks,
            )
            loaded.append(_LoadedPlugin(reference.reference, instance, descriptor))
        return loaded

    def _load_or_create_evidence(self, path: Path) -> PluginRuntimeEvidence:
        if path.is_file():
            try:
                existing = PluginRuntimeEvidence.model_validate_json(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                existing = None
            if (
                existing is not None
                and existing.payload_hash == self.payload.payload_hash
                and existing.protocol_hash == self.payload.protocol_hash
                and existing.component_ids == self.payload.component_ids
                and existing.changed_variables == self.payload.changed_variables
            ):
                existing.component_ids = list(self.payload.component_ids)
                existing.changed_variables = dict(self.payload.changed_variables)
                existing.plugins = [item.descriptor for item in self.plugins]
                return existing
        return PluginRuntimeEvidence(
            payload_hash=self.payload.payload_hash,
            protocol_hash=self.payload.protocol_hash,
            component_ids=list(self.payload.component_ids),
            changed_variables=dict(self.payload.changed_variables),
            ultralytics_version=self.audit.version,
            signature_hash=self.audit.signature_hash,
            compatible=True,
            rank=int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1"))),
            plugins=[item.descriptor for item in self.plugins],
        )

    def _persist_initial_failure(self, message: str) -> None:
        evidence_path = runtime_evidence_path(self.payload_path)
        context = UltralyticsPluginContext(
            payload=self.payload,
            payload_path=self.payload_path,
            audit=self.audit,
            evidence_path=evidence_path,
            evidence=PluginRuntimeEvidence(
                payload_hash=self.payload.payload_hash,
                protocol_hash=self.payload.protocol_hash,
                component_ids=list(self.payload.component_ids),
                changed_variables=dict(self.payload.changed_variables),
                ultralytics_version=self.audit.version,
                signature_hash=self.audit.signature_hash,
                compatible=self.audit.compatible,
                rank=int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1"))),
                failures=[message],
            ),
        )
        context.persist()


class PluginDetectionTrainer(_DetectionTrainer):
    """Stable importable trainer class used by both direct and DDP processes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        payload_path = os.environ.get("YOLO_AGENT_RUNTIME_PAYLOAD")
        if not payload_path:
            raise PluginExecutionError("YOLO_AGENT_RUNTIME_PAYLOAD is required")
        self.plugin_bridge = UltralyticsTrainerPluginBridge(payload_path)
        overrides = kwargs.get("overrides") or {}
        self.plugin_bridge.validate_training_args(dict(overrides))
        self._plugin_batch: Any = None
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_batch_end", self._run_plugin_batch_end)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True) -> Any:
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        return self.plugin_bridge.install_model_hooks(model, trainer=self)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> Any:
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        if mode == "train":
            dataset = self.plugin_bridge.invoke_transform(
                "build_train_dataset",
                dataset,
                trainer=self,
                image_path=img_path,
                batch_size=batch,
            )
        return dataset

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> Any:
        dataloader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        return self.apply_dataloader_plugins(
            dataloader,
            dataset_path=dataset_path,
            batch_size=batch_size,
            rank=rank,
            mode=mode,
        )

    def apply_dataloader_plugins(
        self,
        dataloader: Any,
        *,
        dataset_path: str,
        batch_size: int,
        rank: int,
        mode: str,
    ) -> Any:
        """Apply train-only dataloader plugins without touching val/test loaders."""
        if mode == "train":
            dataloader = self.plugin_bridge.invoke_transform(
                "build_train_dataloader",
                dataloader,
                trainer=self,
                dataset_path=dataset_path,
                batch_size=batch_size,
                rank=rank,
            )
        return dataloader

    def get_validator(self) -> Any:
        validator = super().get_validator()
        return self.plugin_bridge.invoke_transform(
            "build_validator",
            validator,
            trainer=self,
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        processed = super().preprocess_batch(batch)
        self._plugin_batch = processed
        self.plugin_bridge.invoke_event(
            "on_train_batch_start",
            trainer=self,
            batch=processed,
        )
        return processed

    def save_model(self) -> Any:
        result = super().save_model()
        self.plugin_bridge.invoke_event(
            "on_checkpoint_save",
            trainer=self,
            checkpoints={"last": self.last, "best": self.best},
        )
        return result

    def resume_training(self, ckpt: Any) -> None:
        super().resume_training(ckpt)
        if ckpt is not None and self.resume:
            self.plugin_bridge.invoke_event(
                "on_checkpoint_load",
                trainer=self,
                checkpoint=ckpt,
            )

    def _run_plugin_batch_end(self, _: Any) -> None:
        self.plugin_bridge.invoke_event(
            "on_train_batch_end",
            trainer=self,
            batch=self._plugin_batch,
            loss=getattr(self, "loss", None),
            loss_items=getattr(self, "loss_items", None),
        )


__all__ = [
    "PluginCriterionWrapper",
    "PluginDetectionTrainer",
    "PluginExecutionError",
    "TRAINING_HOOKS",
    "UltralyticsTrainerPluginBridge",
]
