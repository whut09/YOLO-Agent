"""Verified paired candidate/control results for promotion and learning."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.matched_baseline import (
    MatchedBaselineControl,
    MatchedBaselineKey,
    PairedMetricDelta,
    paired_metric_delta,
)


ProtocolMatchStatus = Literal["matched", "mismatch", "incomplete"]


class PairedErrorFactDelta(BaseModel):
    """One target diagnosis delta from an exact current-run control pair."""

    fact_key: str
    fact_type: str
    subject: str
    metric_name: str | None = None
    baseline_value: float
    candidate_value: float
    paired_delta: float
    effect_delta: float
    higher_is_better: bool
    improved: bool
    baseline_node_id: str
    candidate_node_id: str
    match_key_hash: str
    verified: bool = True


class PairedBootstrapCI(BaseModel):
    """Image-level paired bootstrap interval already tied to the candidate protocol."""

    metric_name: str = "diagnostic_map50"
    confidence_interval_low: float
    confidence_interval_high: float
    probability_improvement: float | None = Field(default=None, ge=0.0, le=1.0)
    direction: str | None = None
    matched_image_count: int | None = Field(default=None, ge=0)
    verified: bool = True


class PairedExperimentResult(BaseModel):
    """Single authoritative paired result consumed by downstream decisions."""

    schema_version: str = "paired_experiment.v1"
    run_id: str
    candidate_id: str
    candidate_node_id: str
    baseline_candidate_id: str | None = None
    baseline_node_id: str | None = None
    protocol_match_status: ProtocolMatchStatus = "incomplete"
    matched_control: MatchedBaselineControl
    metric_deltas: dict[str, PairedMetricDelta] = Field(default_factory=dict)
    target_error_fact_deltas: list[PairedErrorFactDelta] = Field(default_factory=list)
    latency_delta: PairedMetricDelta | None = None
    model_size_delta: PairedMetricDelta | None = None
    paired_bootstrap_ci: PairedBootstrapCI | None = None
    verified: bool = False
    blockers: list[str] = Field(default_factory=list)
    result_hash: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def fill_hash(self) -> "PairedExperimentResult":
        payload = self.model_dump(mode="json", exclude={"result_hash", "created_at"})
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        if self.result_hash and self.result_hash != expected:
            raise ValueError("paired experiment result_hash does not match payload")
        self.result_hash = expected
        return self

    def to_json(self, path: Path | str) -> Path:
        """Persist the authoritative pair atomically for replay and auditing."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(output)
        return output


def build_paired_experiment_result(
    *,
    run_id: str,
    candidate_id: str,
    candidate_node_id: str,
    metric_records: list[MetricEvidence],
    error_facts: list[ErrorFact],
    primary_metric: str = "map50_95",
    target_error_facts: list[dict[str, Any]] | None = None,
    latency_metric: str = "latency_ms",
    model_size_metric: str = "model_size_mb",
) -> PairedExperimentResult:
    """Build a paired result without cross-run, inherited, or absolute-value fallback."""
    candidate_metrics = [
        record
        for record in metric_records
        if _current_metric(record, run_id, candidate_id, candidate_node_id)
    ]
    primary_candidates = [record for record in candidate_metrics if record.metric_name == primary_metric]
    if not primary_candidates and primary_metric == "map50_95":
        primary_candidates = [record for record in candidate_metrics if record.metric_name == "coco_ap50_95"]
    placeholder = MatchedBaselineControl(
        candidate_run_id=run_id,
        candidate_id=candidate_id,
        candidate_node_id=candidate_node_id,
        mismatch_reasons=["missing_current_candidate_primary_metric"],
    )
    if not primary_candidates:
        return PairedExperimentResult(
            run_id=run_id,
            candidate_id=candidate_id,
            candidate_node_id=candidate_node_id,
            matched_control=placeholder,
            blockers=["missing_current_candidate_primary_metric"],
        )

    primary_candidate = max(primary_candidates, key=lambda record: record.created_at)
    control, primary_delta = paired_metric_delta(primary_candidate, metric_records)
    blockers = list(control.missing_dimensions) + list(control.mismatch_reasons)
    deltas: dict[str, PairedMetricDelta] = {}
    if primary_delta is not None:
        deltas[primary_metric] = primary_delta

    for metric_name in {latency_metric, model_size_metric}:
        candidates = [record for record in candidate_metrics if record.metric_name == metric_name]
        if not candidates:
            blockers.append(f"missing_current_candidate_metric:{metric_name}")
            continue
        _, delta = paired_metric_delta(max(candidates, key=lambda record: record.created_at), metric_records)
        if delta is None:
            blockers.append(f"missing_matched_baseline_control:{metric_name}")
        else:
            deltas[metric_name] = delta

    fact_deltas, fact_blockers = _paired_error_fact_deltas(
        run_id=run_id,
        candidate_id=candidate_id,
        candidate_node_id=candidate_node_id,
        error_facts=error_facts,
        target_error_facts=list(target_error_facts or []),
    )
    blockers.extend(fact_blockers)
    bootstrap = _paired_bootstrap_ci(candidate_metrics)
    protocol_status: ProtocolMatchStatus = "matched" if primary_delta is not None else "mismatch"
    verified = bool(
        primary_delta is not None
        and deltas.get(latency_metric) is not None
        and deltas.get(model_size_metric) is not None
        and all(item.verified for item in deltas.values())
        and not fact_blockers
    )
    return PairedExperimentResult(
        run_id=run_id,
        candidate_id=candidate_id,
        candidate_node_id=candidate_node_id,
        baseline_candidate_id=control.baseline_candidate_id,
        baseline_node_id=control.baseline_node_id,
        protocol_match_status=protocol_status,
        matched_control=control,
        metric_deltas=deltas,
        target_error_fact_deltas=fact_deltas,
        latency_delta=deltas.get(latency_metric),
        model_size_delta=deltas.get(model_size_metric),
        paired_bootstrap_ci=bootstrap,
        verified=verified,
        blockers=list(dict.fromkeys(blockers)),
    )


def _current_metric(record: MetricEvidence, run_id: str, candidate_id: str, node_id: str) -> bool:
    return bool(
        record.run_id == run_id
        and (record.origin_run_id or record.run_id) == run_id
        and record.candidate_id == candidate_id
        and record.node_id == node_id
        and record.evidence_role == "current_observation"
        and record.inheritance_depth == 0
        and record.verified
    )


def _paired_error_fact_deltas(
    *,
    run_id: str,
    candidate_id: str,
    candidate_node_id: str,
    error_facts: list[ErrorFact],
    target_error_facts: list[dict[str, Any]],
) -> tuple[list[PairedErrorFactDelta], list[str]]:
    candidates = [
        fact
        for fact in error_facts
        if fact.run_id == run_id
        and (fact.origin_run_id or fact.run_id) == run_id
        and fact.inheritance_depth == 0
        and fact.candidate_id == candidate_id
        and fact.node_id == candidate_node_id
        and fact.evidence_role == "current_observation"
    ]
    requested = {_target_key(item) for item in target_error_facts if _target_key(item) is not None}
    results: list[PairedErrorFactDelta] = []
    blockers: list[str] = []
    matched_requested: set[tuple[str, str, str, str, str, str]] = set()
    for candidate in candidates:
        key = _fact_key(candidate)
        if requested and key not in requested:
            continue
        candidate_match_key = _error_fact_match_key(candidate)
        if candidate_match_key is None:
            blockers.append(f"candidate_error_fact_identity_incomplete:{'|'.join(key)}")
            continue
        baselines = [
            fact
            for fact in error_facts
            if fact.run_id == run_id
            and (fact.origin_run_id or fact.run_id) == run_id
            and fact.inheritance_depth == 0
            and fact.evidence_role == "baseline_reference"
            and _fact_key(fact) == key
            and _error_fact_match_key(fact) == candidate_match_key
        ]
        if not baselines:
            blockers.append(f"missing_matched_baseline_error_fact:{'|'.join(key)}")
            continue
        baseline = max(baselines, key=lambda fact: fact.created_at)
        baseline_value = _fact_numeric(baseline)
        candidate_value = _fact_numeric(candidate)
        if baseline_value is None or candidate_value is None:
            blockers.append(f"non_numeric_paired_error_fact:{'|'.join(key)}")
            continue
        higher_is_better = _error_fact_higher_is_better(candidate)
        raw_delta = candidate_value - baseline_value
        effect_delta = raw_delta if higher_is_better else -raw_delta
        results.append(
            PairedErrorFactDelta(
                fact_key="|".join(key),
                fact_type=candidate.fact_type,
                subject=candidate.subject,
                metric_name=candidate.metric_name,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                paired_delta=raw_delta,
                effect_delta=effect_delta,
                higher_is_better=higher_is_better,
                improved=effect_delta > 0,
                baseline_node_id=baseline.node_id,
                candidate_node_id=candidate.node_id,
                match_key_hash=candidate_match_key.match_key_hash,
            )
        )
        matched_requested.add(key)
    for key in requested - matched_requested:
        blockers.append(f"missing_target_error_fact_pair:{'|'.join(key)}")
    return results, blockers


def _error_fact_match_key(fact: ErrorFact) -> MatchedBaselineKey | None:
    payload = {
        "dataset_manifest_sha256": fact.dataset_manifest_sha256,
        "protocol_hash": fact.protocol_hash,
        "subset_manifest_sha256": fact.subset_manifest_sha256,
        "seed": None if fact.seed is None else str(fact.seed),
        "epochs": fact.epochs,
        "fidelity": fact.fidelity,
        "batch_policy_hash": fact.batch_policy_hash,
        "ultralytics_version": fact.ultralytics_version,
        "imgsz": fact.imgsz,
        "eval_protocol_hash": fact.eval_protocol_hash,
        "split": fact.split,
    }
    if any(value is None or value == "" for value in payload.values()) or payload["imgsz"] != 640:
        return None
    return MatchedBaselineKey.model_validate(payload)


def _paired_bootstrap_ci(records: list[MetricEvidence]) -> PairedBootstrapCI | None:
    values = {record.metric_name: record.value for record in records if record.metric_name.startswith("bootstrap/")}
    low = _numeric(values.get("bootstrap/diagnostic_map50_ci_low"))
    high = _numeric(values.get("bootstrap/diagnostic_map50_ci_high"))
    if low is None or high is None:
        return None
    return PairedBootstrapCI(
        confidence_interval_low=low,
        confidence_interval_high=high,
        probability_improvement=_numeric(values.get("bootstrap/diagnostic_map50_probability_improvement")),
        direction=str(values.get("bootstrap/diagnostic_map50_direction") or "") or None,
        matched_image_count=_integer(values.get("bootstrap/matched_image_count")),
    )


def _target_key(item: dict[str, Any]) -> tuple[str, str, str, str, str, str] | None:
    key = (
        str(item.get("fact_type") or ""),
        str(item.get("subject") or ""),
        str(item.get("class_name") or ""),
        str(item.get("class_pair") or ""),
        str(item.get("area") or ""),
        str(item.get("metric_name") or ""),
    )
    return key if key[0] and key[1] else None


def _fact_key(fact: ErrorFact) -> tuple[str, str, str, str, str, str]:
    return (
        fact.fact_type,
        fact.subject,
        fact.class_name or "",
        fact.class_pair or "",
        fact.area or "",
        fact.metric_name or "",
    )


def _fact_numeric(fact: ErrorFact) -> float | None:
    value = _numeric(fact.value)
    return value if value is not None else (float(fact.count) if fact.count is not None else None)


def _error_fact_higher_is_better(fact: ErrorFact) -> bool:
    return fact.fact_type not in {
        "false_negative_heavy_class",
        "localization_heavy_class",
        "class_confusion_pair",
        "background_false_positive_class",
    }


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: object) -> int | None:
    numeric = _numeric(value)
    return int(numeric) if numeric is not None else None


__all__ = [
    "PairedBootstrapCI",
    "PairedErrorFactDelta",
    "PairedExperimentResult",
    "build_paired_experiment_result",
]
