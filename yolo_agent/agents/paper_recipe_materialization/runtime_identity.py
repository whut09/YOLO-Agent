"""Fail-closed runtime identity validation for materialized paper recipes."""

from __future__ import annotations

from yolo_agent.agents.paper_recipe_materialization.schemas import (
    MaterializedAdapterIdentity,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.execution_bridge import ComponentExecutionResult


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


__all__ = ["certified_runtime_identity"]
