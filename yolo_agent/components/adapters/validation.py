"""Validation helpers for the Component Adapter SDK."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from yolo_agent.components.adapters.base import (
    ComponentAdapter,
    PatchOperation,
    RollbackPlan,
)
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RUNTIME_PLUGIN_METHODS,
)


_HOOK_REQUIRED_PARAMETERS: dict[str, set[str]] = {
    "prepare_command": {"payload", "command", "env"},
    "build_model": {"context", "trainer", "model"},
    "build_train_dataset": {"context", "trainer", "dataset"},
    "build_train_dataloader": {"context", "trainer", "dataloader"},
    "build_validator": {"context", "trainer", "validator"},
    "build_criterion": {"context", "trainer", "criterion"},
    "compute_loss": {"context", "trainer", "loss_output"},
    "on_train_batch_start": {"context", "trainer"},
    "on_train_batch_end": {"context", "trainer"},
    "on_checkpoint_save": {"context", "trainer"},
    "on_checkpoint_load": {"context", "trainer"},
    "on_model_serialize_start": {"context", "trainer"},
    "on_model_serialize_end": {"context", "trainer"},
}


def diff_config(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    target: str,
    prefix: str = "",
) -> list[PatchOperation]:
    """Return deterministic leaf-level changes between two mappings."""
    operations: list[PatchOperation] = []
    keys = sorted(set(before) | set(after))
    for key in keys:
        field = f"{prefix}.{key}" if prefix else key
        old = before.get(key)
        new = after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            operations.extend(diff_config(old, new, target=target, prefix=field))
        elif old != new:
            operations.append(PatchOperation(target=target, field=field, before=old, after=new))
    return operations


def validate_adapter_metadata(adapter: ComponentAdapter) -> None:
    """Ensure an adapter declares stable provenance and a supported strategy."""
    if not str(getattr(adapter, "adapter_version", "")).strip():
        raise ValueError("Adapter must declare adapter_version")
    if not str(getattr(adapter, "source_commit", "")).strip():
        raise ValueError("Adapter must declare source_commit")
    if getattr(adapter, "strategy", None) not in {
        "custom_module", "custom_model_yaml", "callback", "trainer_subclass",
        "loss_injection", "assigner_injection",
        "inference_adapter",
    }:
        raise ValueError("Adapter must declare a supported local integration strategy")


def validate_declared_operations(adapter: ComponentAdapter, operations: list[PatchOperation]) -> None:
    """Reject changes not declared by the adapter contract."""
    allowed = {
        "model_config." + field for field in adapter.modified_model_fields
    } | {
        "training_config." + field for field in adapter.modified_training_fields
    }
    missing = [
        f"{item.target}.{item.field}"
        for item in operations
        if f"{item.target}.{item.field}" not in allowed
    ]
    if missing:
        raise ValueError(f"Adapter changed undeclared fields: {', '.join(sorted(missing))}")


def validate_rollback_plan(plan: RollbackPlan, workspace: Path) -> None:
    """Prevent rollback declarations from targeting global installed code."""
    if plan.restores_global_source:
        raise ValueError("Adapters cannot modify or restore global framework source")
    root = workspace.resolve()
    for path in plan.files_to_remove:
        candidate = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError(f"Rollback path escapes adapter workspace: {path}")


def validate_runtime_plugin_hooks(
    payload: AdapterRuntimePayload,
) -> dict[str, bool | str | int | float]:
    """Instantiate runtime plugins and enforce the trainer hook signatures."""
    loaded = 0
    hook_count = 0
    identities: list[str] = []
    for reference in payload.plugin_references:
        implementation = reference.resolve()
        instance = (
            implementation(**reference.options)
            if isinstance(implementation, type)
            else implementation
        )
        hooks = sorted(
            method
            for method in RUNTIME_PLUGIN_METHODS
            if callable(getattr(instance, method, None))
        )
        if not hooks:
            raise ValueError(
                f"runtime plugin has no callable hooks: {reference.reference}"
            )
        missing_required = sorted(set(reference.required_hooks) - set(hooks))
        if missing_required:
            raise ValueError(
                f"runtime plugin is missing required hooks: {reference.reference}:"
                f"{','.join(missing_required)}"
            )
        for hook in hooks:
            method = getattr(instance, hook)
            signature = inspect.signature(method)
            parameters = signature.parameters
            accepts_kwargs = any(
                item.kind is inspect.Parameter.VAR_KEYWORD
                for item in parameters.values()
            )
            missing = (
                set()
                if accepts_kwargs
                else _HOOK_REQUIRED_PARAMETERS[hook] - set(parameters)
            )
            if missing:
                raise ValueError(
                    f"runtime plugin hook signature mismatch: {reference.reference}:{hook}:"
                    f"missing={','.join(sorted(missing))}"
                )
        loaded += 1
        hook_count += len(hooks)
        identities.append(f"{reference.reference}={','.join(hooks)}")
    return {
        "runtime_plugins_loaded": loaded,
        "runtime_plugin_hooks_verified": hook_count,
        "runtime_hook_signatures_verified": hook_count,
        "runtime_plugin_identities": ";".join(identities),
    }


__all__ = [
    "diff_config",
    "validate_adapter_metadata",
    "validate_declared_operations",
    "validate_rollback_plan",
    "validate_runtime_plugin_hooks",
]
