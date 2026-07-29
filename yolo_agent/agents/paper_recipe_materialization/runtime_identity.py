"""Fail-closed runtime identity validation for materialized paper recipes."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.paper_recipe_materialization.schemas import (
    MaterializedAdapterIdentity,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.execution_bridge import ComponentExecutionResult
from yolo_agent.core.experiment_graph import ExperimentNode


def certified_runtime_identity(
    result: ComponentExecutionResult,
) -> MaterializedAdapterIdentity:
    if result.status != "executable":
        raise ValueError("component execution bridge did not produce an executable result")
    if not result.adapters or len(result.adapters) != len(result.component_ids):
        raise ValueError("every paper component requires one prepared adapter record")
    if any(not item.smoke_test.passed for item in result.adapters):
        raise ValueError("every runtime adapter must have passed smoke evidence")
    if not result.aggregate_patch_hash:
        raise ValueError("aggregate adapter patch hash is missing")
    if result.runtime_payload_path is None or not result.runtime_payload_path.is_file():
        raise ValueError("adapter runtime payload artifact is missing")
    if not result.runtime_payload_hash or not result.protocol_hash:
        raise ValueError("adapter runtime payload or protocol hash is missing")
    payload = AdapterRuntimePayload.read(result.runtime_payload_path, verify_imports=True)
    if payload.payload_hash != result.runtime_payload_hash:
        raise ValueError("adapter runtime payload hash mismatch")
    command = result.node.command_spec
    if command is None or command.command_type != "train":
        raise ValueError("materialized adapter must wrap a typed training command")
    metadata = command.metadata
    if metadata.get("adapter_runtime_payload_hash") != result.runtime_payload_hash:
        raise ValueError("training command is not bound to the certified runtime payload")
    if not metadata.get("adapter_runtime_entrypoint") or "--payload" not in command.argv:
        raise ValueError("training command silently fell back to plain Ultralytics execution")
    return MaterializedAdapterIdentity(
        component_ids=list(result.component_ids),
        adapter_classes={item.component_id: item.adapter_class for item in result.adapters},
        adapter_versions={item.component_id: item.adapter_version for item in result.adapters},
        adapter_patch_hashes={item.component_id: item.patch_hash for item in result.adapters},
        aggregate_patch_hash=result.aggregate_patch_hash,
        runtime_payload_hash=result.runtime_payload_hash,
        runtime_payload_path=result.runtime_payload_path,
        protocol_hash=result.protocol_hash,
    )


def validate_runtime_identity_binding(
    node: ExperimentNode,
    identity: MaterializedAdapterIdentity,
) -> list[str]:
    """Return fail-closed binding errors between a node and certified runtime."""
    command = node.command_spec
    if command is None or command.command_type != "train":
        return ["certified_adapter_training_command_missing"]
    metadata = command.metadata
    errors: list[str] = []
    component_ids = {
        item.strip()
        for item in str(metadata.get("component_ids") or "").split(",")
        if item.strip()
    }
    if component_ids != set(identity.component_ids):
        errors.append("certified_adapter_component_identity_mismatch")
    if metadata.get("adapter_patch_hash") != identity.aggregate_patch_hash:
        errors.append("certified_adapter_patch_hash_mismatch")
    if metadata.get("adapter_runtime_payload_hash") != identity.runtime_payload_hash:
        errors.append("certified_adapter_payload_hash_mismatch")
    if metadata.get("adapter_runtime_protocol_hash") != identity.protocol_hash:
        errors.append("certified_adapter_protocol_hash_mismatch")
    payload_path = str(metadata.get("adapter_runtime_payload_path") or "")
    if not payload_path or Path(payload_path).resolve() != identity.runtime_payload_path.resolve():
        errors.append("certified_adapter_payload_path_mismatch")
    if not identity.runtime_payload_path.is_file():
        errors.append("certified_adapter_payload_missing")
    elif not errors:
        try:
            payload = AdapterRuntimePayload.read(identity.runtime_payload_path, verify_imports=True)
            if payload.payload_hash != identity.runtime_payload_hash:
                errors.append("certified_adapter_payload_content_mismatch")
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("certified_adapter_payload_invalid")
    entrypoint = str(metadata.get("adapter_runtime_entrypoint") or "")
    if not entrypoint or "--payload" not in command.argv:
        errors.append("plain_ultralytics_fallback_forbidden")
    return errors


def validate_certified_runtime_node(node: ExperimentNode) -> list[str]:
    """Validate runtime materialization directly at an ASHA registration boundary."""
    command = node.command_spec
    if command is None or command.command_type != "train":
        return ["certified_adapter_training_command_missing"]
    metadata = command.metadata
    errors: list[str] = []
    component_ids = [
        item.strip()
        for item in str(metadata.get("component_ids") or "").split(",")
        if item.strip()
    ]
    expected_components = list(node.candidate_config.components)
    if not component_ids or set(component_ids) != set(expected_components):
        errors.append("certified_adapter_component_identity_mismatch")
    if not str(metadata.get("adapter_patch_hash") or ""):
        errors.append("certified_adapter_patch_hash_missing")
    payload_hash = str(metadata.get("adapter_runtime_payload_hash") or "")
    payload_path = Path(str(metadata.get("adapter_runtime_payload_path") or ""))
    protocol_hash = str(metadata.get("adapter_runtime_protocol_hash") or "")
    if not payload_hash or not protocol_hash:
        errors.append("certified_adapter_payload_identity_missing")
    if not payload_path.is_file():
        errors.append("certified_adapter_payload_missing")
    else:
        try:
            payload = AdapterRuntimePayload.read(payload_path, verify_imports=True)
            if payload.payload_hash != payload_hash:
                errors.append("certified_adapter_payload_content_mismatch")
            if set(payload.component_ids) != set(expected_components):
                errors.append("certified_adapter_payload_component_mismatch")
            if payload.protocol_hash != protocol_hash:
                errors.append("certified_adapter_payload_protocol_mismatch")
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("certified_adapter_payload_invalid")
    if not metadata.get("adapter_runtime_entrypoint") or "--payload" not in command.argv:
        errors.append("plain_ultralytics_fallback_forbidden")
    return list(dict.fromkeys(errors))


__all__ = [
    "certified_runtime_identity",
    "validate_certified_runtime_node",
    "validate_runtime_identity_binding",
]
