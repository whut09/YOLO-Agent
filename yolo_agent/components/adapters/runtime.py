"""Typed runtime payloads for component adapter execution."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.base import ExpectedArtifact, RollbackPlan


RuntimePluginKind = Literal[
    "dataloader_plugin",
    "trainer_plugin",
    "model_graph_plugin",
    "loss_plugin",
    "assigner_plugin",
    "inference_plugin",
]
RUNTIME_PLUGIN_METHODS = {
    "prepare_command",
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


class RuntimePluginReference(BaseModel):
    """Importable local plugin reference used by the runtime entrypoint."""

    model_config = ConfigDict(extra="forbid")

    reference: str
    options: dict[str, Any] = Field(default_factory=dict)
    required_hooks: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reference(self) -> "RuntimePluginReference":
        module_name, separator, qualname = self.reference.partition(":")
        if not separator or not module_name.strip() or not qualname.strip():
            raise ValueError("runtime plugin reference must use 'module:qualname'")
        if module_name.startswith("ultralytics"):
            raise ValueError("runtime plugins must live outside the installed ultralytics package")
        if len(self.required_hooks) != len(set(self.required_hooks)):
            raise ValueError("runtime plugin required_hooks must be unique")
        unsupported = set(self.required_hooks) - RUNTIME_PLUGIN_METHODS
        if unsupported:
            raise ValueError(
                "runtime plugin has unsupported required hooks: "
                + ", ".join(sorted(unsupported))
            )
        return self

    def resolve(self) -> Any:
        """Import the referenced object without modifying framework source."""
        module_name, _, qualname = self.reference.partition(":")
        value: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            value = getattr(value, part)
        return value


class AdapterRuntimePayload(BaseModel):
    """Serializable contract consumed by the isolated runtime entrypoint."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "adapter_runtime_payload.v1"
    component_ids: list[str] = Field(min_length=1)
    adapter_classes: list[str] = Field(min_length=1)
    adapter_versions: dict[str, str]
    source_commits: dict[str, str]
    dataloader_plugin: list[RuntimePluginReference] = Field(default_factory=list)
    trainer_plugin: list[RuntimePluginReference] = Field(default_factory=list)
    model_graph_plugin: list[RuntimePluginReference] = Field(default_factory=list)
    loss_plugin: list[RuntimePluginReference] = Field(default_factory=list)
    assigner_plugin: list[RuntimePluginReference] = Field(default_factory=list)
    inference_plugin: list[RuntimePluginReference] = Field(default_factory=list)
    runtime_entrypoint: str = "yolo_agent.adapters.ultralytics.runtime_entrypoint"
    generated_config: dict[str, Any] = Field(default_factory=dict)
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    expected_artifacts: list[ExpectedArtifact] = Field(default_factory=list)
    rollback_plan: RollbackPlan
    protocol_hash: str
    base_command: list[str] = Field(min_length=1)
    cwd: Path | None = None
    env: dict[str, str] = Field(default_factory=dict)
    supports_amp: bool = False
    supports_ddp: bool = False
    supports_resume: bool = False

    @model_validator(mode="after")
    def validate_payload(self) -> "AdapterRuntimePayload":
        if not self.protocol_hash.strip():
            raise ValueError("runtime payload requires protocol_hash")
        if len(self.component_ids) != len(set(self.component_ids)):
            raise ValueError("runtime payload component_ids must be unique")
        if len(self.adapter_classes) != len(self.component_ids):
            raise ValueError("runtime payload requires one adapter class per component")
        if set(self.adapter_versions) != set(self.component_ids):
            raise ValueError("runtime payload adapter_versions must match component_ids")
        if set(self.source_commits) != set(self.component_ids):
            raise ValueError("runtime payload source_commits must match component_ids")
        if not self.plugin_references:
            raise ValueError("runtime payload requires at least one plugin reference")
        if not self.changed_variables:
            raise ValueError("runtime payload requires at least one changed variable")
        if any(not str(name).strip() for name in self.changed_variables):
            raise ValueError("runtime payload changed variable names must be non-empty")
        if self.runtime_entrypoint.startswith("ultralytics"):
            raise ValueError("runtime entrypoint must not modify or live inside ultralytics")
        return self

    @property
    def plugin_references(self) -> list[RuntimePluginReference]:
        """Return all declared plugin references in stable execution order."""
        return [
            *self.dataloader_plugin,
            *self.trainer_plugin,
            *self.model_graph_plugin,
            *self.loss_plugin,
            *self.assigner_plugin,
            *self.inference_plugin,
        ]

    @property
    def payload_hash(self) -> str:
        """Return a deterministic hash of the complete runtime contract."""
        content = self.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def verify_imports(self) -> None:
        """Fail closed unless the entrypoint and every plugin are importable."""
        if importlib.util.find_spec(self.runtime_entrypoint) is None:
            raise ImportError(f"runtime entrypoint is not importable: {self.runtime_entrypoint}")
        entrypoint = importlib.import_module(self.runtime_entrypoint)
        if not callable(getattr(entrypoint, "main", None)):
            raise ImportError(f"runtime entrypoint has no callable main: {self.runtime_entrypoint}")
        for plugin in self.plugin_references:
            try:
                implementation = plugin.resolve()
            except (AttributeError, ImportError, ModuleNotFoundError) as exc:
                raise ImportError(f"runtime plugin is not importable: {plugin.reference}") from exc
            if not str(getattr(implementation, "plugin_version", "")).strip():
                raise ImportError(f"runtime plugin has no plugin_version: {plugin.reference}")
            missing_required = [
                hook
                for hook in plugin.required_hooks
                if not callable(getattr(implementation, hook, None))
            ]
            if missing_required:
                raise ImportError(
                    f"runtime plugin is missing required hooks: {plugin.reference}:"
                    f"{','.join(missing_required)}"
                )
            if not any(
                callable(getattr(implementation, method, None))
                for method in RUNTIME_PLUGIN_METHODS
            ):
                raise ImportError(
                    f"runtime plugin has no supported hooks: {plugin.reference}"
                )

    def write(self, path: Path | str) -> Path:
        """Atomically persist the payload for resume and Windows child processes."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                self.model_dump(mode="json", exclude_none=True),
                file,
                sort_keys=False,
            )
        temporary.replace(output)
        return output

    @classmethod
    def read(cls, path: Path | str, *, verify_imports: bool = True) -> "AdapterRuntimePayload":
        """Restore a payload and optionally verify its local runtime dependencies."""
        source = Path(path)
        with source.open("r", encoding="utf-8-sig") as file:
            raw = yaml.safe_load(file) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"runtime payload must contain a mapping: {source}")
        payload = cls.model_validate(raw)
        if verify_imports:
            payload.verify_imports()
        return payload

    @classmethod
    def compose(
        cls,
        payloads: list["AdapterRuntimePayload"],
        *,
        generated_config: dict[str, Any],
        expected_artifacts: list[ExpectedArtifact],
        rollback_plan: RollbackPlan,
    ) -> "AdapterRuntimePayload":
        """Compose multiple compatible component payloads into one process contract."""
        if not payloads:
            raise ValueError("at least one runtime payload is required")
        first = payloads[0]
        for payload in payloads[1:]:
            if payload.protocol_hash != first.protocol_hash:
                raise ValueError("runtime payload protocol hashes do not match")
            if payload.base_command != first.base_command:
                raise ValueError("runtime payload base commands do not match")
            if payload.runtime_entrypoint != first.runtime_entrypoint:
                raise ValueError("runtime payload entrypoints do not match")
        changed_variables: dict[str, Any] = {}
        for payload in payloads:
            for name, value in payload.changed_variables.items():
                if name in changed_variables and changed_variables[name] != value:
                    raise ValueError(
                        f"runtime payload changed variable conflict: {name}"
                    )
                changed_variables[name] = value
        composed = cls(
            component_ids=[item for payload in payloads for item in payload.component_ids],
            adapter_classes=[item for payload in payloads for item in payload.adapter_classes],
            adapter_versions={key: value for payload in payloads for key, value in payload.adapter_versions.items()},
            source_commits={key: value for payload in payloads for key, value in payload.source_commits.items()},
            dataloader_plugin=[item for payload in payloads for item in payload.dataloader_plugin],
            trainer_plugin=[item for payload in payloads for item in payload.trainer_plugin],
            model_graph_plugin=[item for payload in payloads for item in payload.model_graph_plugin],
            loss_plugin=[item for payload in payloads for item in payload.loss_plugin],
            assigner_plugin=[item for payload in payloads for item in payload.assigner_plugin],
            inference_plugin=[item for payload in payloads for item in payload.inference_plugin],
            runtime_entrypoint=first.runtime_entrypoint,
            generated_config=generated_config,
            changed_variables=changed_variables,
            expected_artifacts=expected_artifacts,
            rollback_plan=rollback_plan,
            protocol_hash=first.protocol_hash,
            base_command=first.base_command,
            cwd=first.cwd,
            env={key: value for payload in payloads for key, value in payload.env.items()},
            supports_amp=all(payload.supports_amp for payload in payloads),
            supports_ddp=all(payload.supports_ddp for payload in payloads),
            supports_resume=all(payload.supports_resume for payload in payloads),
        )
        composed.verify_imports()
        return composed


__all__ = [
    "AdapterRuntimePayload",
    "RUNTIME_PLUGIN_METHODS",
    "RuntimePluginKind",
    "RuntimePluginReference",
]
