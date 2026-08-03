"""Fail-closed runtime command construction for certified components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from yolo_agent.certification.paper_auto_optimization_research import (
    load_component_contract,
)
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload


@dataclass(frozen=True)
class ComponentRuntimeLaunch:
    """Executable command and artifact contract for one component runtime."""

    component_id: str
    command: list[str]
    payload: AdapterRuntimePayload
    payload_path: Path
    runtime_artifacts: dict[str, Path]


def build_component_runtime_launch(
    *,
    component_id: str,
    base_command: list[str],
    workspace: Path | str,
    protocol_hash: str,
    options: dict[str, Any],
    environment: dict[str, Any] | None = None,
    registry: ComponentAdapterRegistry | None = None,
) -> ComponentRuntimeLaunch:
    """Build an import-verified adapter launch without framework source patches."""
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract = load_component_contract(component_id)
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=root,
        environment=dict(environment or {}),
        options=dict(options),
    )
    adapter = (registry or ComponentAdapterRegistry()).create_for_contract(contract)
    preview = adapter.prepare_patch({}, {}, context, dry_run=False)
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash=protocol_hash,
        base_command=list(base_command),
        generated_config={
            "model_config": preview.patched_model_config,
            "training_config": preview.patched_training_config,
        },
    )
    if payload is None:
        raise RuntimeError(
            f"component adapter has no runtime payload: {component_id}"
        )
    if payload.component_ids != [component_id]:
        raise RuntimeError(
            "component runtime payload identity mismatch: "
            f"expected={component_id} actual={payload.component_ids}"
        )
    if payload.protocol_hash != protocol_hash:
        raise RuntimeError("component runtime payload protocol hash mismatch")
    if payload.base_command != base_command:
        raise RuntimeError("component runtime payload changed the base command")
    payload.verify_imports()
    payload_path = payload.write(root / "adapter_runtime_payload.yaml")
    artifacts = {
        "runtime_payload": payload_path,
        "plugin_runtime_evidence": root / "plugin_runtime_evidence.json",
    }
    for expected in payload.expected_artifacts:
        if expected.name in artifacts:
            raise RuntimeError(
                f"duplicate component runtime artifact name: {expected.name}"
            )
        artifacts[expected.name] = root / expected.relative_path
    command = [
        sys.executable,
        "-m",
        payload.runtime_entrypoint,
        "--payload",
        str(payload_path),
        "--",
        *base_command,
    ]
    return ComponentRuntimeLaunch(
        component_id=component_id,
        command=command,
        payload=payload,
        payload_path=payload_path,
        runtime_artifacts=artifacts,
    )


__all__ = ["ComponentRuntimeLaunch", "build_component_runtime_launch"]
