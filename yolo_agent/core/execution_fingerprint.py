"""Canonical execution identities shared by candidate routing and memory.

An execution fingerprint answers one question only: would these two nodes run
the same implementation under the same evaluation contract? Paper provenance
is deliberately excluded so multiple papers can share one trial without
causing repeated training.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.paired_experiment import PairedExperimentResult
from yolo_agent.research.component_aliases import (
    ComponentAliasConfig,
    ComponentAliasResolver,
)


EXECUTION_FINGERPRINT_SCHEMA_VERSION = "execution_fingerprint.v1"


def execution_identity_payload(
    node: ExperimentNode,
    *,
    fidelity: str | None = None,
    baseline_protocol_hash: str | None = None,
    dataset_manifest_hash: str | None = None,
    recipe_id: str | None = None,
    recipe_version: str | None = None,
    combination_id: str | None = None,
    combination_fingerprint: str | None = None,
    effective_overrides: dict[str, Any] | None = None,
    runtime_payload_hash: str | None = None,
    teacher_checkpoint_hash: str | None = None,
    graph_identity_hash: str | None = None,
) -> dict[str, Any]:
    """Return the complete semantic identity used for execution dedupe."""
    metadata = _metadata(node)
    candidate = node.candidate_config
    resolved_fidelity = _first(
        fidelity,
        metadata.get("fidelity"),
        metadata.get("round_stage"),
        metadata.get("training_budget_profile"),
        "unknown",
    )
    protocol = _first(
        baseline_protocol_hash,
        metadata.get("baseline_protocol_hash"),
        metadata.get("run_protocol_hash"),
        metadata.get("protocol_hash"),
        "unknown",
    )
    dataset = _first(
        dataset_manifest_hash,
        metadata.get("dataset_manifest_sha256"),
        metadata.get("dataset_manifest_hash"),
        node.data_version,
        "unknown",
    )
    recipe = _first(
        recipe_id,
        metadata.get("component_recipe_id"),
        metadata.get("paper_recipe_id"),
        candidate.action_id,
        "unknown",
    )
    version = _first(
        recipe_version,
        metadata.get("component_recipe_version"),
        candidate.train_overrides.get("recipe_version"),
        "unknown",
    )
    combination = _first(
        combination_id,
        metadata.get("ablation_combination_id"),
        metadata.get("combination_id"),
        "atomic",
    )
    return execution_identity_payload_from_values(
        model_checkpoint_identity=_first(
            metadata.get("model_checkpoint_sha256"),
            metadata.get("checkpoint_hash"),
            candidate.base_model,
            "unknown",
        ),
        component_ids=candidate.components,
        recipe_id=recipe,
        recipe_version=version,
        effective_overrides={
            **candidate.train_overrides,
            **node.effective_overrides,
            **node.changed_variables,
            **(effective_overrides or {}),
        },
        dataset_manifest_hash=dataset,
        baseline_protocol_hash=protocol,
        imgsz=_imgsz(node),
        fidelity=resolved_fidelity,
        seed=node.seed,
        teacher_checkpoint_hash=_first(
            teacher_checkpoint_hash,
            metadata.get("teacher_checkpoint_sha256"),
            metadata.get("teacher_checkpoint_hash"),
            "none",
        ),
        graph_identity_hash=_first(
            graph_identity_hash,
            metadata.get("graph_identity_hash"),
            metadata.get("model_graph_identity_hash"),
            "none",
        ),
        runtime_payload_hash=_first(
            runtime_payload_hash,
            metadata.get("adapter_runtime_payload_hash"),
            metadata.get("runtime_payload_hash"),
            "none",
        ),
        combination_id=combination,
        combination_fingerprint=_first(
            combination_fingerprint,
            metadata.get("combination_fingerprint"),
            metadata.get("coupled_combination_fingerprint"),
            "none",
        ),
    )


def execution_identity_payload_from_values(
    *,
    model_checkpoint_identity: Any,
    component_ids: list[str],
    recipe_id: Any,
    recipe_version: Any,
    effective_overrides: dict[str, Any],
    dataset_manifest_hash: Any,
    baseline_protocol_hash: Any,
    imgsz: Any,
    fidelity: Any,
    seed: Any,
    teacher_checkpoint_hash: Any = "none",
    graph_identity_hash: Any = "none",
    runtime_payload_hash: Any = "none",
    combination_id: Any = "atomic",
    combination_fingerprint: Any = "none",
) -> dict[str, Any]:
    """Build the same identity payload for non-node policy proposals."""
    return {
        "schema_version": EXECUTION_FINGERPRINT_SCHEMA_VERSION,
        "model_checkpoint_identity": _first(model_checkpoint_identity, "unknown"),
        "canonical_component_ids": canonical_component_ids(component_ids),
        "recipe_id": _first(recipe_id, "unknown"),
        "recipe_version": _first(recipe_version, "unknown"),
        "effective_overrides": _normalized_mapping(effective_overrides),
        "dataset_manifest_hash": _first(dataset_manifest_hash, "unknown"),
        "baseline_protocol_hash": _first(baseline_protocol_hash, "unknown"),
        "imgsz": imgsz,
        "fidelity": _first(fidelity, "unknown"),
        "seed": seed,
        "teacher_checkpoint_hash": _first(teacher_checkpoint_hash, "none"),
        "graph_identity_hash": _first(graph_identity_hash, "none"),
        "runtime_payload_hash": _first(runtime_payload_hash, "none"),
        "combination_id": _first(combination_id, "atomic"),
        "combination_fingerprint": _first(combination_fingerprint, "none"),
    }


def execution_fingerprint(node: ExperimentNode, **kwargs: Any) -> str:
    """Hash a canonical node identity; paper IDs never affect this value."""
    payload = execution_identity_payload(node, **kwargs)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def canonical_component_ids(component_ids: list[str]) -> list[str]:
    """Normalize spelling while retaining dotted canonical taxonomy IDs.

    Alias resolution is performed before candidates reach this layer. This
    function intentionally does not infer semantic aliases from substrings;
    unresolved identifiers remain explicit and therefore cannot collide with a
    known component merely because their normalized text looks similar.
    """
    resolver = _alias_resolver()
    resolved: set[str] = set()
    for raw in component_ids:
        value = str(raw).strip()
        if not value:
            continue
        match = resolver.resolve(value)
        if match.resolved:
            resolved.update(item.canonical_component_id for item in match.mappings)
        else:
            resolved.add(value)
    return sorted(resolved)


def paired_evidence_is_valid(
    paired: PairedExperimentResult | None,
    *,
    expected_candidate_id: str | None = None,
    expected_protocol_hash: str | None = None,
    expected_dataset_manifest_hash: str | None = None,
    expected_imgsz: int = 640,
    expected_fidelity: str | None = None,
) -> bool:
    """Accept only current, matched, protocol-compatible paired evidence."""
    if paired is None or not paired.verified:
        return False
    if paired.protocol_match_status != "matched" or not paired.matched_control.matched:
        return False
    if expected_candidate_id and paired.candidate_id != expected_candidate_id:
        return False
    delta = paired.metric_deltas.get("map50_95")
    if delta is None or not delta.verified:
        return False
    key = paired.matched_control.match_key
    if key is None or key.imgsz != expected_imgsz:
        return False
    if expected_protocol_hash and key.protocol_hash != expected_protocol_hash:
        return False
    if expected_dataset_manifest_hash and key.dataset_manifest_sha256 != expected_dataset_manifest_hash:
        return False
    if expected_fidelity and key.fidelity != expected_fidelity:
        return False
    return True


def _metadata(node: ExperimentNode) -> dict[str, Any]:
    return dict(node.command_spec.metadata) if node.command_spec is not None else {}


@lru_cache(maxsize=1)
def _alias_resolver() -> ComponentAliasResolver:
    return ComponentAliasResolver(ComponentAliasConfig.from_yaml())


def _imgsz(node: ExperimentNode) -> int | str:
    metadata = _metadata(node)
    value = metadata.get("imgsz", node.effective_overrides.get("imgsz", 640))
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _first(*values: Any) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return "unknown"


def _normalized_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _normalize_value(value)
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
        if str(key) not in {"imgsz", "seed"}
    }


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


__all__ = [
    "EXECUTION_FINGERPRINT_SCHEMA_VERSION",
    "canonical_component_ids",
    "execution_fingerprint",
    "execution_identity_payload",
    "execution_identity_payload_from_values",
    "paired_evidence_is_valid",
]
