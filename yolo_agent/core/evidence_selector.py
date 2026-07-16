"""Strict provenance selectors for candidate-level evidence decisions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.core.experiment_graph import MetricEvidence


class EvidenceSelector(BaseModel):
    """Explicit scope and protocol constraints for evidence queries."""

    current_run_id: str | None = None
    current_run_only: bool = False
    current_node_only: list[str] = Field(default_factory=list)
    inherited_context: bool | None = None
    baseline_reference: bool | None = None
    same_protocol_hash: str | None = None
    same_dataset_manifest: str | None = None
    same_subset_manifest: str | None = None
    same_split: str | None = None
    same_seed: int | str | None = None
    same_epochs: int | None = None
    same_fidelity: str | None = None
    same_batch_policy_hash: str | None = None
    same_ultralytics_version: str | None = None
    same_imgsz: int | None = None
    same_eval_protocol_hash: str | None = None
    candidate_id: str | None = None
    metric_name: str | None = None
    verified: bool | None = True


class EvidenceSelection(BaseModel):
    """Selected records plus deterministic rejection counts."""

    records: list[MetricEvidence] = Field(default_factory=list)
    rejected_by: dict[str, int] = Field(default_factory=dict)


def select_metric_evidence(
    records: Iterable[MetricEvidence],
    selector: EvidenceSelector,
) -> EvidenceSelection:
    """Select records only when every requested identity constraint matches."""
    values = list(records)
    companions = _companion_identity(values)
    selected: list[MetricEvidence] = []
    rejected: dict[str, int] = {}
    for record in values:
        reason = _rejection_reason(record, selector, companions)
        if reason is None:
            selected.append(record)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    return EvidenceSelection(records=selected, rejected_by=rejected)


def _rejection_reason(
    record: MetricEvidence,
    selector: EvidenceSelector,
    companions: dict[tuple[str, str], dict[str, str]],
) -> str | None:
    key = (record.candidate_id, record.node_id)
    identity = companions.get(key, {})
    inherited = _is_inherited(record, selector.current_run_id)
    if selector.current_run_only:
        if selector.current_run_id is None:
            return "missing_current_run_id"
        if record.run_id != selector.current_run_id or inherited:
            return "not_current_run"
        if record.evidence_role != "current_observation":
            return "not_current_observation"
    if selector.current_node_only and record.node_id not in set(selector.current_node_only):
        return "not_current_node"
    if selector.inherited_context is True and not inherited:
        return "not_inherited_context"
    if selector.inherited_context is False and inherited:
        return "inherited_context_excluded"
    is_baseline = record.evidence_role == "baseline_reference"
    if selector.baseline_reference is True and not is_baseline:
        return "not_baseline_reference"
    if selector.baseline_reference is False and is_baseline:
        return "baseline_reference_excluded"
    if selector.same_protocol_hash is not None:
        actual = record.protocol_hash or identity.get("protocol_hash")
        if actual != selector.same_protocol_hash:
            return "protocol_hash_mismatch"
    if selector.same_dataset_manifest is not None:
        actual = record.dataset_manifest_sha256 or identity.get("dataset_manifest_sha256")
        if actual != selector.same_dataset_manifest:
            return "dataset_manifest_mismatch"
    if selector.same_subset_manifest is not None and record.subset_manifest_sha256 != selector.same_subset_manifest:
        return "subset_manifest_mismatch"
    if selector.same_split is not None and record.split != selector.same_split:
        return "split_mismatch"
    if selector.same_seed is not None:
        actual_seed = record.seed if record.seed is not None else identity.get("seed")
        if str(actual_seed) != str(selector.same_seed):
            return "seed_mismatch"
    if selector.same_epochs is not None and record.epochs != selector.same_epochs:
        return "epochs_mismatch"
    if selector.same_fidelity is not None and record.fidelity != selector.same_fidelity:
        return "fidelity_mismatch"
    if selector.same_batch_policy_hash is not None and record.batch_policy_hash != selector.same_batch_policy_hash:
        return "batch_policy_mismatch"
    if selector.same_ultralytics_version is not None and record.ultralytics_version != selector.same_ultralytics_version:
        return "ultralytics_version_mismatch"
    if selector.same_imgsz is not None and record.imgsz != selector.same_imgsz:
        return "imgsz_mismatch"
    if selector.same_eval_protocol_hash is not None and record.eval_protocol_hash != selector.same_eval_protocol_hash:
        return "eval_protocol_mismatch"
    if selector.candidate_id is not None and record.candidate_id != selector.candidate_id:
        return "candidate_mismatch"
    if selector.metric_name is not None and record.metric_name != selector.metric_name:
        return "metric_mismatch"
    if selector.verified is not None and record.verified is not selector.verified:
        return "verification_mismatch"
    return None


def _is_inherited(record: MetricEvidence, current_run_id: str | None) -> bool:
    if record.inheritance_depth > 0 or record.evidence_role in {"inherited_context", "baseline_reference"}:
        return True
    if current_run_id is None:
        return False
    return record.origin_run_id not in {None, current_run_id}


def _companion_identity(records: list[MetricEvidence]) -> dict[tuple[str, str], dict[str, str]]:
    identities: dict[tuple[str, str], dict[str, str]] = {}
    metric_keys: dict[str, Literal["protocol_hash", "dataset_manifest_sha256", "seed"]] = {
        "baseline_protocol_hash": "protocol_hash",
        "dataset_manifest_sha256": "dataset_manifest_sha256",
        "fast_baseline_seed": "seed",
    }
    for record in records:
        identity = identities.setdefault((record.candidate_id, record.node_id), {})
        if record.protocol_hash:
            identity["protocol_hash"] = record.protocol_hash
        if record.dataset_manifest_sha256:
            identity["dataset_manifest_sha256"] = record.dataset_manifest_sha256
        if record.seed is not None:
            identity["seed"] = str(record.seed)
        target = metric_keys.get(record.metric_name)
        if target is not None and record.value is not None:
            identity[target] = str(record.value)
    return identities
