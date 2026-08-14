"""Fail-closed runtime identity validation for materialized paper recipes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

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
        adapter_hashes={item.component_id: item.adapter_hash for item in result.adapters},
        component_maturity={
            item.component_id: item.component_maturity for item in result.adapters
        },
        maturity_artifact_hashes={
            item.component_id: item.maturity_artifact_hashes
            for item in result.adapters
        },
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
    if _metadata_mapping(metadata, "adapter_hashes") != identity.adapter_hashes:
        errors.append("certified_adapter_source_hash_mismatch")
    if _metadata_mapping(metadata, "component_maturity") != identity.component_maturity:
        errors.append("certified_component_maturity_mismatch")
    if (
        _metadata_mapping(metadata, "maturity_artifact_hashes")
        != identity.maturity_artifact_hashes
    ):
        errors.append("certified_maturity_artifact_hash_mismatch")
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
            _append_payload_shape_errors(errors, payload_path)
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
    errors.extend(_quality_evaluation_contract_errors(node))
    if not component_ids or set(component_ids) != set(expected_components):
        errors.append("certified_adapter_component_identity_mismatch")
    if not str(metadata.get("adapter_patch_hash") or ""):
        errors.append("certified_adapter_patch_hash_missing")
    adapter_hashes = _metadata_mapping(metadata, "adapter_hashes")
    component_maturity = _metadata_mapping(metadata, "component_maturity")
    maturity_artifacts = _metadata_mapping(metadata, "maturity_artifact_hashes")
    if set(adapter_hashes) != set(expected_components) or any(
        not str(value) for value in adapter_hashes.values()
    ):
        errors.append("certified_adapter_source_hash_missing")
    if set(component_maturity) != set(expected_components) or any(
        value not in {
            "smoke_passed",
            "gpu_certified",
            "pilot_reproduced",
            "full_reproduced",
            "confirmed_multi_seed",
        }
        for value in component_maturity.values()
    ):
        errors.append("certified_component_maturity_invalid")
    if set(maturity_artifacts) != set(expected_components) or any(
        not isinstance(value, list) or not value
        for value in maturity_artifacts.values()
    ):
        errors.append("certified_maturity_artifact_hash_missing")
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
            errors.extend(_quality_runtime_contract_errors(node, payload, command.argv))
        except (ImportError, OSError, TypeError, ValueError):
            _append_payload_shape_errors(errors, payload_path)
            errors.append("certified_adapter_payload_invalid")
    imgsz = _command_imgsz(command.argv)
    if imgsz is not None and imgsz != 640:
        errors.append("fixed_imgsz_must_equal_640")
    if not metadata.get("adapter_runtime_entrypoint") or "--payload" not in command.argv:
        errors.append("plain_ultralytics_fallback_forbidden")
    return list(dict.fromkeys(errors))


def _quality_runtime_contract_errors(
    node: ExperimentNode,
    payload: AdapterRuntimePayload,
    argv: list[str],
) -> list[str]:
    """Apply quality-loss-specific runtime checks before ASHA registration."""
    expected_variables = {
        "loss.quality.correlation": "loss.correlation.weight",
        "loss.quality.pseudo_iou": "loss.pseudo_iou.weight",
    }
    expected_evidence = {
        "loss.quality.correlation": "auxiliary_loss_correlation_evidence",
        "loss.quality.pseudo_iou": "auxiliary_loss_pseudo_iou_evidence",
    }
    artifact_names = {artifact.name for artifact in payload.expected_artifacts}
    errors: list[str] = []
    for component_id, variable in expected_variables.items():
        if component_id not in node.candidate_config.components:
            continue
        if variable not in payload.changed_variables:
            errors.append("adapter_changed_variable_missing")
        if expected_evidence[component_id] not in artifact_names:
            errors.append("quality_evidence_artifact_missing")
    if any(component_id in node.candidate_config.components for component_id in expected_variables):
        if not any(
            reference.required_hooks and "compute_loss" in reference.required_hooks
            for reference in payload.loss_plugin
        ):
            errors.append("adapter_runtime_payload_missing")
    return errors


def _quality_evaluation_contract_errors(node: ExperimentNode) -> list[str]:
    quality_components = {
        "loss.quality.correlation",
        "loss.quality.pseudo_iou",
    }
    if not quality_components.intersection(node.candidate_config.components):
        return []
    contract = node.candidate_config.evaluation_contract
    errors: list[str] = []
    if contract.primary_metric != "map50_95":
        errors.append("quality_primary_metric_must_be_map50_95")
    localization_metrics = {
        "ap75",
        "confidence_iou_correlation",
        "localization_error_rate",
        "confidence_localization_mismatch",
    }
    if not localization_metrics.intersection(contract.evaluation_metrics):
        errors.append("quality_localization_metric_missing")
    guards = {
        *contract.stop_conditions,
        *contract.promotion_requirements,
    }
    if not any("latency_guard" in item for item in guards):
        errors.append("quality_latency_guard_missing")
    if not any("model_size_guard" in item for item in guards):
        errors.append("quality_model_size_guard_missing")
    return errors


def _command_imgsz(argv: list[str]) -> int | None:
    for token in argv:
        key, separator, value = token.partition("=")
        if key == "imgsz" and separator:
            try:
                return int(str(value))
            except ValueError:
                return None
    return None


def _append_payload_shape_errors(errors: list[str], payload_path: Path) -> None:
    """Expose stable registration reasons for malformed adapter payloads."""
    try:
        raw = yaml.safe_load(payload_path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, TypeError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    if not raw.get("changed_variables"):
        errors.append("adapter_changed_variable_missing")
    if not raw.get("loss_plugin") and any(
        component_id.startswith("loss.quality.")
        for component_id in raw.get("component_ids", [])
        if isinstance(component_id, str)
    ):
        errors.append("adapter_runtime_payload_missing")


def _metadata_mapping(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key)
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = [
    "certified_runtime_identity",
    "validate_certified_runtime_node",
    "validate_runtime_identity_binding",
]
