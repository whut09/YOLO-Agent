"""Strict matched-control identities and paired metric deltas."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field, model_validator

from yolo_agent.core.experiment_graph import MetricEvidence


MATCHED_BASELINE_SCHEMA_VERSION = "matched_baseline.v1"


class MatchedBaselineKey(BaseModel):
    """Identity dimensions that must agree before observations are comparable."""

    schema_version: str = MATCHED_BASELINE_SCHEMA_VERSION
    dataset_manifest_sha256: str
    protocol_hash: str
    subset_manifest_sha256: str
    seed: str
    epochs: int = Field(ge=1)
    fidelity: str
    batch_policy_hash: str
    ultralytics_version: str
    imgsz: int = Field(ge=1)
    eval_protocol_hash: str
    split: str

    @property
    def match_key_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class MatchedBaselineControl(BaseModel):
    """Result of matching one candidate observation to a baseline control."""

    candidate_run_id: str | None = None
    candidate_id: str
    candidate_node_id: str
    baseline_run_id: str | None = None
    baseline_candidate_id: str | None = None
    baseline_node_id: str | None = None
    match_key: MatchedBaselineKey | None = None
    matched: bool = False
    status: str = "needs_matched_baseline"
    missing_dimensions: list[str] = Field(default_factory=list)
    mismatch_reasons: list[str] = Field(default_factory=list)


class PairedMetricDelta(BaseModel):
    """A metric effect computed only from an exact matched control pair."""

    metric_name: str
    baseline_value: float
    candidate_value: float
    paired_delta: float
    effect_delta: float
    higher_is_better: bool
    baseline_run_id: str | None = None
    baseline_candidate_id: str
    baseline_node_id: str
    candidate_run_id: str | None = None
    candidate_id: str
    candidate_node_id: str
    baseline_source: str
    candidate_source: str
    match_key: MatchedBaselineKey
    match_key_hash: str
    verified: bool = True

    @model_validator(mode="after")
    def validate_hash(self) -> "PairedMetricDelta":
        if self.match_key_hash != self.match_key.match_key_hash:
            raise ValueError("match_key_hash does not match match_key")
        return self


def build_match_key(record: MetricEvidence) -> tuple[MatchedBaselineKey | None, list[str]]:
    """Build a complete match key, returning explicit missing dimensions."""
    values: dict[str, Any] = {
        "dataset_manifest_sha256": record.dataset_manifest_sha256,
        "protocol_hash": record.protocol_hash,
        "subset_manifest_sha256": record.subset_manifest_sha256,
        "seed": None if record.seed is None else str(record.seed),
        "epochs": record.epochs,
        "fidelity": record.fidelity,
        "batch_policy_hash": record.batch_policy_hash,
        "ultralytics_version": record.ultralytics_version,
        "imgsz": record.imgsz,
        "eval_protocol_hash": record.eval_protocol_hash,
        "split": record.split,
    }
    missing = [name for name, value in values.items() if value is None or value == ""]
    if values["imgsz"] is not None and values["imgsz"] != 640:
        missing.append("imgsz_not_fixed_640")
    if missing:
        return None, missing
    return MatchedBaselineKey.model_validate(values), []


def match_baseline_control(
    candidate: MetricEvidence,
    baseline_records: Iterable[MetricEvidence],
) -> tuple[MatchedBaselineControl, MetricEvidence | None]:
    """Find the newest verified baseline-reference record with an exact key match."""
    candidate_key, missing = build_match_key(candidate)
    control = MatchedBaselineControl(
        candidate_run_id=candidate.run_id,
        candidate_id=candidate.candidate_id,
        candidate_node_id=candidate.node_id,
        match_key=candidate_key,
        missing_dimensions=missing,
    )
    if candidate.evidence_role != "current_observation" or candidate.inheritance_depth > 0:
        control.mismatch_reasons.append("candidate_not_current_observation")
        return control, None
    if candidate.run_id is None or (candidate.origin_run_id or candidate.run_id) != candidate.run_id:
        control.mismatch_reasons.append("candidate_not_current_run")
        return control, None
    if not candidate.verified:
        control.mismatch_reasons.append("candidate_not_verified")
        return control, None
    if candidate_key is None:
        return control, None

    mismatches: set[str] = set()
    matches: list[MetricEvidence] = []
    for baseline in baseline_records:
        if baseline.metric_name != candidate.metric_name or not baseline.verified:
            continue
        if baseline.evidence_role != "baseline_reference":
            mismatches.add("baseline_not_explicit_reference")
            continue
        candidate_run_id = candidate.origin_run_id or candidate.run_id
        baseline_run_id = baseline.origin_run_id or baseline.run_id
        if candidate.run_id is None or candidate_run_id != candidate.run_id:
            mismatches.add("candidate_not_current_run")
            continue
        if baseline.run_id != candidate.run_id or baseline_run_id != candidate.run_id:
            mismatches.add("baseline_not_current_run")
            continue
        if baseline.inheritance_depth > 0:
            mismatches.add("baseline_inherited")
            continue
        baseline_key, baseline_missing = build_match_key(baseline)
        if baseline_key is None:
            mismatches.update(f"baseline_missing_{name}" for name in baseline_missing)
            continue
        for name in MatchedBaselineKey.model_fields:
            if name == "schema_version":
                continue
            if getattr(baseline_key, name) != getattr(candidate_key, name):
                mismatches.add(f"{name}_mismatch")
        if baseline_key == candidate_key:
            matches.append(baseline)
    if not matches:
        control.mismatch_reasons = sorted(mismatches) or ["no_baseline_reference_for_metric"]
        return control, None

    baseline = max(matches, key=lambda item: item.created_at)
    control.baseline_run_id = baseline.origin_run_id or baseline.run_id
    control.baseline_candidate_id = baseline.candidate_id
    control.baseline_node_id = baseline.node_id
    control.matched = True
    control.status = "matched"
    return control, baseline


def paired_metric_delta(
    candidate: MetricEvidence,
    baseline_records: Iterable[MetricEvidence],
) -> tuple[MatchedBaselineControl, PairedMetricDelta | None]:
    """Compute a paired delta, or return a diagnostic without a delta."""
    control, baseline = match_baseline_control(candidate, baseline_records)
    if baseline is None or control.match_key is None:
        return control, None
    if isinstance(candidate.value, bool) or not isinstance(candidate.value, (int, float)):
        control.matched = False
        control.status = "needs_numeric_candidate_metric"
        control.mismatch_reasons = ["candidate_metric_not_numeric"]
        return control, None
    if isinstance(baseline.value, bool) or not isinstance(baseline.value, (int, float)):
        control.matched = False
        control.status = "needs_numeric_baseline_metric"
        control.mismatch_reasons = ["baseline_metric_not_numeric"]
        return control, None
    raw_delta = float(candidate.value) - float(baseline.value)
    higher_is_better = bool(candidate.higher_is_better)
    delta = PairedMetricDelta(
        metric_name=candidate.metric_name,
        baseline_value=float(baseline.value),
        candidate_value=float(candidate.value),
        paired_delta=raw_delta,
        effect_delta=raw_delta if higher_is_better else -raw_delta,
        higher_is_better=higher_is_better,
        baseline_run_id=baseline.origin_run_id or baseline.run_id,
        baseline_candidate_id=baseline.candidate_id,
        baseline_node_id=baseline.node_id,
        candidate_run_id=candidate.run_id,
        candidate_id=candidate.candidate_id,
        candidate_node_id=candidate.node_id,
        baseline_source=baseline.source,
        candidate_source=candidate.source,
        match_key=control.match_key,
        match_key_hash=control.match_key.match_key_hash,
    )
    return control, delta


def stable_identity_hash(payload: dict[str, Any]) -> str:
    """Hash a protocol identity without relying on display names."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
