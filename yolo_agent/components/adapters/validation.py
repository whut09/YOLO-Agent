"""Validation helpers for the Component Adapter SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yolo_agent.components.adapters.base import (
    ComponentAdapter,
    PatchOperation,
    RollbackPlan,
)


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
    undeclared = [item.field if item.target == "model_config" else item.field for item in operations]
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


__all__ = ["diff_config", "validate_adapter_metadata", "validate_declared_operations", "validate_rollback_plan"]
