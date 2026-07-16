"""Long-term policy memory learned from error-fact deltas."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, model_validator


PolicyConfidence = Literal["low", "medium", "high"]
PolicyTrend = Literal["improved", "regressed", "unchanged", "resolved", "new", "current"]
PolicyFidelity = Literal["debug", "pilot", "full", "unknown"]

CONFIDENCE_RANK: dict[PolicyConfidence, int] = {"low": 0, "medium": 1, "high": 2}


class PolicyActionCost(BaseModel):
    """Runtime and deployment cost observed for one action effect."""

    latency_before_ms: float | None = None
    latency_after_ms: float | None = None
    latency_delta_ms: float | None = None
    latency_delta_pct: float | None = None
    model_size_before_mb: float | None = None
    model_size_after_mb: float | None = None
    model_size_delta_mb: float | None = None
    model_size_delta_pct: float | None = None
    gpu_hours: float | None = None


class ActionFingerprint(BaseModel):
    """Normalized identity of an executed action, independent of candidate naming."""

    schema_version: str = "action_fingerprint.v1"
    action: str
    recipe_id: str | None = None
    component_versions: dict[str, str] = Field(default_factory=dict)
    changed_variable: str = "unknown"
    before_value: Any = None
    after_value: Any = None
    model_family: str = "unknown"
    dataset_signature: str = "unversioned"
    protocol_hash: str = "unknown"
    fidelity: PolicyFidelity = "unknown"
    matched_control_hash: str | None = None

    @property
    def fingerprint_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def transfer_sha256(self) -> str:
        """Return an identity shared by pilot/full observations of the same action."""
        payload = self.model_dump(
            mode="json",
            exclude={"fidelity", "dataset_signature", "protocol_hash"},
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def posterior_sha256(self) -> str:
        """Return an action/fidelity bucket that can weight similar datasets together."""
        payload = self.model_dump(
            mode="json",
            exclude={"dataset_signature", "protocol_hash"},
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class PolicyCostDistribution(BaseModel):
    """Observed deployment-cost distribution for an action posterior."""

    mean: float | None = None
    variance: float | None = None
    p50: float | None = None
    p90: float | None = None
    minimum: float | None = None
    maximum: float | None = None


class PolicyMemoryRecord(BaseModel):
    """One learned action-effect observation from a closed-loop run."""

    schema_version: str = "policy_memory.v2"
    record_id: str = ""
    run_id: str
    parent_run_id: str | None = None
    dataset_version: str = "unversioned"
    split: str = "val"
    scenario: str | None = None
    action: str
    action_fingerprint: ActionFingerprint | None = None
    action_fingerprint_sha256: str = ""
    target: str
    target_fact_type: str | None = None
    target_subject: str | None = None
    class_name: str | None = None
    class_pair: str | None = None
    area: str | None = None
    metric_name: str | None = None
    before: float | None = None
    after: float | None = None
    delta: float | None = None
    effect_delta: float | None = None
    higher_is_better: bool = True
    trend: PolicyTrend = "unchanged"
    candidate_id: str | None = None
    node_id: str | None = None
    cost: PolicyActionCost = Field(default_factory=PolicyActionCost)
    confidence: PolicyConfidence = "low"
    confidence_reason: str = "single observation"
    seed_count: int = 1
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    inferred_action: bool = False
    source: str = "error_fact_delta"
    matched_control_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def fill_derived_fields(self) -> "PolicyMemoryRecord":
        """Fill deterministic id and normalized effect direction."""
        if self.effect_delta is None and self.delta is not None:
            self.effect_delta = self.delta if self.higher_is_better else -self.delta
        if self.action_fingerprint is None:
            changed_variable, after_value = _legacy_action_transition(self.action, self.changed_variables)
            self.action_fingerprint = ActionFingerprint(
                action=self.action,
                changed_variable=changed_variable,
                after_value=after_value,
                dataset_signature=self.dataset_version,
            )
        if not self.action_fingerprint_sha256:
            self.action_fingerprint_sha256 = self.action_fingerprint.fingerprint_sha256
        if not self.record_id:
            self.record_id = _record_id(self)
        return self


class PolicyMemorySummary(BaseModel):
    """Aggregated historical effect for one action/target/metric bucket."""

    action: str
    action_fingerprint_sha256: str = ""
    posterior_key_sha256: str = ""
    action_fingerprint: ActionFingerprint | None = None
    target: str | None = None
    metric_name: str | None = None
    record_count: int
    mean_delta: float | None = None
    mean_effect_delta: float | None = None
    mean_latency_delta_pct: float | None = None
    mean_model_size_delta_pct: float | None = None
    mean_target_metric_gain: float | None = None
    mean_error_fact_gain: float | None = None
    effect_variance: float | None = None
    effect_stddev: float | None = None
    confidence_interval_95: tuple[float, float] | None = None
    posterior_confidence: PolicyConfidence = "low"
    seed_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    pilot_mean_effect_delta: float | None = None
    full_mean_effect_delta: float | None = None
    pilot_to_full_correlation: float | None = None
    pilot_to_full_gain_ratio: float | None = None
    latency_cost_distribution: PolicyCostDistribution = Field(default_factory=PolicyCostDistribution)
    model_size_cost_distribution: PolicyCostDistribution = Field(default_factory=PolicyCostDistribution)
    mean_dataset_similarity_weight: float = 1.0
    effective_sample_size: float = 0.0
    confidence_counts: dict[str, int] = Field(default_factory=dict)
    latest_record_ids: list[str] = Field(default_factory=list)


class PolicyMemoryStore:
    """Append-only JSONL memory at ``runs/policy_memory.jsonl``."""

    def __init__(self, root: Path | str = "runs") -> None:
        self.root = Path(root)

    @property
    def path(self) -> Path:
        """Return the policy memory JSONL path."""
        return self.root / "policy_memory.jsonl"

    def append(self, records: Iterable[PolicyMemoryRecord]) -> list[PolicyMemoryRecord]:
        """Append new records idempotently and return records actually written."""
        materialized = list(records)
        if not materialized:
            return []
        existing_ids = {record.record_id for record in self.read()}
        new_records = [record for record in materialized if record.record_id not in existing_ids]
        if not new_records:
            return []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            for record in new_records:
                file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        return new_records

    def read(self) -> list[PolicyMemoryRecord]:
        """Read all memory records."""
        if not self.path.is_file():
            return []
        records: list[PolicyMemoryRecord] = []
        with self.path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                text = line.strip()
                if text:
                    records.append(PolicyMemoryRecord.model_validate(json.loads(text)))
        return records

    def query(
        self,
        action: str | None = None,
        target: str | None = None,
        metric_name: str | None = None,
        dataset_version: str | None = None,
        run_id: str | None = None,
        min_confidence: PolicyConfidence | None = None,
        action_fingerprint_sha256: str | None = None,
        fidelity: PolicyFidelity | None = None,
    ) -> list[PolicyMemoryRecord]:
        """Return records matching all supplied filters."""
        records = self.read()
        if min_confidence is not None:
            min_rank = CONFIDENCE_RANK[min_confidence]
        else:
            min_rank = None
        return [
            record
            for record in records
            if (action is None or record.action == action)
            and (target is None or record.target == target)
            and (metric_name is None or record.metric_name == metric_name)
            and (dataset_version is None or record.dataset_version == dataset_version)
            and (run_id is None or record.run_id == run_id)
            and (
                action_fingerprint_sha256 is None
                or record.action_fingerprint_sha256 == action_fingerprint_sha256
            )
            and (
                fidelity is None
                or (record.action_fingerprint is not None and record.action_fingerprint.fidelity == fidelity)
            )
            and (min_rank is None or CONFIDENCE_RANK[record.confidence] >= min_rank)
        ]

    def summarize(
        self,
        action: str | None = None,
        target: str | None = None,
        metric_name: str | None = None,
        dataset_version: str | None = None,
        dataset_signature: str | None = None,
        scenario: str | None = None,
        model_family: str | None = None,
    ) -> list[PolicyMemorySummary]:
        """Aggregate action effects into weighted posterior summaries."""
        groups: dict[tuple[str, str | None, str | None], list[PolicyMemoryRecord]] = defaultdict(list)
        for record in self.query(
            action=action,
            target=target,
            metric_name=metric_name,
            dataset_version=dataset_version,
        ):
            fingerprint = record.action_fingerprint
            posterior_key = fingerprint.posterior_sha256 if fingerprint is not None else record.action_fingerprint_sha256
            groups[(posterior_key, record.target, record.metric_name)].append(record)
        all_records = self.read()
        summaries: list[PolicyMemorySummary] = []
        for (posterior_key_sha256, group_target, group_metric), records in sorted(
            groups.items(),
            key=lambda item: tuple(str(value or "") for value in item[0]),
        ):
            confidence_counts: dict[str, int] = defaultdict(int)
            for record in records:
                confidence_counts[record.confidence] += 1
            weights = [
                _dataset_similarity_weight(
                    record,
                    dataset_signature=dataset_signature,
                    scenario=scenario,
                    model_family=model_family,
                )
                for record in records
            ]
            effects = [record.effect_delta for record in records]
            mean_effect = _weighted_mean(effects, weights)
            variance = _weighted_variance(effects, weights, mean_effect)
            interval = _confidence_interval_95(effects, weights, mean_effect, variance)
            total_seed_count = sum(max(record.seed_count, 1) for record in records)
            fingerprint = records[0].action_fingerprint
            transfer = _pilot_full_stats(
                all_records,
                transfer_sha256=fingerprint.transfer_sha256 if fingerprint is not None else "",
                target=group_target,
                metric_name=group_metric,
            )
            summaries.append(
                PolicyMemorySummary(
                    action=records[0].action,
                    action_fingerprint_sha256=(fingerprint.fingerprint_sha256 if fingerprint is not None else ""),
                    posterior_key_sha256=posterior_key_sha256,
                    action_fingerprint=fingerprint,
                    target=group_target,
                    metric_name=group_metric,
                    record_count=len(records),
                    mean_delta=_weighted_mean([record.delta for record in records], weights),
                    mean_effect_delta=mean_effect,
                    mean_target_metric_gain=mean_effect if group_metric else None,
                    mean_error_fact_gain=mean_effect,
                    mean_latency_delta_pct=_weighted_mean([record.cost.latency_delta_pct for record in records], weights),
                    mean_model_size_delta_pct=_weighted_mean([record.cost.model_size_delta_pct for record in records], weights),
                    effect_variance=variance,
                    effect_stddev=round(math.sqrt(variance), 6) if variance is not None else None,
                    confidence_interval_95=interval,
                    posterior_confidence=_posterior_confidence(total_seed_count, interval),
                    seed_count=total_seed_count,
                    success_count=sum(1 for record in records if (record.effect_delta or 0.0) > 0),
                    failure_count=sum(1 for record in records if record.effect_delta is not None and record.effect_delta <= 0),
                    pilot_mean_effect_delta=transfer["pilot_mean"],
                    full_mean_effect_delta=transfer["full_mean"],
                    pilot_to_full_correlation=transfer["correlation"],
                    pilot_to_full_gain_ratio=transfer["gain_ratio"],
                    latency_cost_distribution=_distribution(record.cost.latency_delta_pct for record in records),
                    model_size_cost_distribution=_distribution(record.cost.model_size_delta_pct for record in records),
                    mean_dataset_similarity_weight=round(sum(weights) / len(weights), 6),
                    effective_sample_size=_effective_sample_size(weights),
                    confidence_counts=dict(confidence_counts),
                    latest_record_ids=[record.record_id for record in sorted(records, key=lambda item: item.created_at)[-5:]],
                )
            )
        return summaries


def stable_negative_action_reasons(
    records: Iterable[PolicyMemoryRecord],
    actions: set[str],
) -> list[str]:
    """Return hard-negative priors only after repeated, statistically stable evidence."""
    groups: dict[tuple[str, str, str | None, str | None], list[PolicyMemoryRecord]] = defaultdict(list)
    for record in records:
        if record.action not in actions or record.effect_delta is None:
            continue
        fingerprint = record.action_fingerprint
        transfer = fingerprint.transfer_sha256 if fingerprint is not None else record.action
        groups[(record.action, transfer, record.target, record.metric_name)].append(record)
    reasons: list[str] = []
    for (action, _, target, _), grouped in groups.items():
        effects = [record.effect_delta for record in grouped]
        weights = [1.0] * len(grouped)
        mean = _weighted_mean(effects, weights)
        variance = _weighted_variance(effects, weights, mean)
        interval = _confidence_interval_95(effects, weights, mean, variance)
        seed_count = sum(max(record.seed_count, 1) for record in grouped)
        if seed_count >= 3 and interval is not None and interval[1] <= 0:
            reasons.append(
                f"stable_historical_no_gain:{action}:{target}:{mean}:ci95={interval[0]},{interval[1]}:seeds={seed_count}"
            )
    return reasons


def _record_id(record: PolicyMemoryRecord) -> str:
    payload = {
        "run_id": record.run_id,
        "parent_run_id": record.parent_run_id,
        "dataset_version": record.dataset_version,
        "split": record.split,
        "action": record.action,
        "action_fingerprint_sha256": record.action_fingerprint_sha256,
        "target": record.target,
        "metric_name": record.metric_name,
        "before": record.before,
        "after": record.after,
        "delta": record.delta,
        "candidate_id": record.candidate_id,
        "node_id": record.node_id,
        "changed_variables": record.changed_variables,
        "inferred_action": record.inferred_action,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 6)


def _legacy_action_transition(action: str, changed_variables: dict[str, Any]) -> tuple[str, Any]:
    for key, value in sorted(changed_variables.items()):
        values = value if isinstance(value, list) else [value]
        if action in {str(item) for item in values if item is not None}:
            return str(key), value
    if changed_variables:
        key = sorted(changed_variables)[0]
        return str(key), changed_variables[key]
    return "unknown", action


def _weighted_mean(values: Iterable[float | None], weights: list[float]) -> float | None:
    pairs = [(float(value), weight) for value, weight in zip(values, weights) if value is not None and weight > 0]
    if not pairs:
        return None
    total = sum(weight for _, weight in pairs)
    return round(sum(value * weight for value, weight in pairs) / total, 6)


def _weighted_variance(
    values: Iterable[float | None],
    weights: list[float],
    mean: float | None,
) -> float | None:
    pairs = [(float(value), weight) for value, weight in zip(values, weights) if value is not None and weight > 0]
    if mean is None or len(pairs) < 2:
        return None
    total = sum(weight for _, weight in pairs)
    return round(sum(weight * (value - mean) ** 2 for value, weight in pairs) / total, 9)


def _confidence_interval_95(
    values: Iterable[float | None],
    weights: list[float],
    mean: float | None,
    variance: float | None,
) -> tuple[float, float] | None:
    numeric_count = sum(1 for value, weight in zip(values, weights) if value is not None and weight > 0)
    effective_n = _effective_sample_size(weights)
    if mean is None or variance is None or numeric_count < 2 or effective_n <= 1:
        return None
    margin = 1.96 * math.sqrt(variance / effective_n)
    return round(mean - margin, 6), round(mean + margin, 6)


def _effective_sample_size(weights: list[float]) -> float:
    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    if squared <= 0:
        return 0.0
    return round(total * total / squared, 6)


def _posterior_confidence(
    seed_count: int,
    interval: tuple[float, float] | None,
) -> PolicyConfidence:
    if seed_count >= 3 and interval is not None and (interval[0] > 0 or interval[1] < 0):
        return "high"
    if seed_count >= 2:
        return "medium"
    return "low"


def _distribution(values: Iterable[float | None]) -> PolicyCostDistribution:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return PolicyCostDistribution()
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / len(numeric) if len(numeric) > 1 else 0.0
    return PolicyCostDistribution(
        mean=round(mean, 6),
        variance=round(variance, 9),
        p50=_percentile(numeric, 0.5),
        p90=_percentile(numeric, 0.9),
        minimum=round(numeric[0], 6),
        maximum=round(numeric[-1], 6),
    )


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return round(values[0], 6)
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(values[lower], 6)
    fraction = position - lower
    return round(values[lower] + (values[upper] - values[lower]) * fraction, 6)


def _dataset_similarity_weight(
    record: PolicyMemoryRecord,
    *,
    dataset_signature: str | None,
    scenario: str | None,
    model_family: str | None,
) -> float:
    fingerprint = record.action_fingerprint
    if dataset_signature is None and scenario is None and model_family is None:
        return 1.0
    weight = 0.25
    if dataset_signature and fingerprint is not None and fingerprint.dataset_signature == dataset_signature:
        weight = 1.0
    elif dataset_signature and record.dataset_version == dataset_signature:
        weight = 0.9
    elif scenario and record.scenario == scenario:
        weight = 0.65
    if model_family and fingerprint is not None and fingerprint.model_family == model_family:
        weight = min(1.0, weight + 0.15)
    return weight


def _pilot_full_stats(
    records: list[PolicyMemoryRecord],
    *,
    transfer_sha256: str,
    target: str | None,
    metric_name: str | None,
) -> dict[str, float | None]:
    matching = [
        record
        for record in records
        if record.action_fingerprint is not None
        and record.action_fingerprint.transfer_sha256 == transfer_sha256
        and record.target == target
        and record.metric_name == metric_name
    ]
    pilot = [record.effect_delta for record in matching if record.action_fingerprint.fidelity == "pilot"]
    full = [record.effect_delta for record in matching if record.action_fingerprint.fidelity == "full"]
    pilot_mean = _mean(pilot)
    full_mean = _mean(full)
    ratio = None
    if pilot_mean not in {None, 0.0} and full_mean is not None:
        ratio = round(full_mean / pilot_mean, 6)
    pairs: list[tuple[float, float]] = []
    by_context: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in matching:
        effect = record.effect_delta
        if effect is None or record.action_fingerprint.fidelity not in {"pilot", "full"}:
            continue
        key = (record.action_fingerprint.dataset_signature, record.action_fingerprint.protocol_hash)
        by_context[key][record.action_fingerprint.fidelity].append(effect)
    for values in by_context.values():
        if values.get("pilot") and values.get("full"):
            pairs.append((sum(values["pilot"]) / len(values["pilot"]), sum(values["full"]) / len(values["full"])))
    return {
        "pilot_mean": pilot_mean,
        "full_mean": full_mean,
        "correlation": _pearson(pairs),
        "gain_ratio": ratio,
    }


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)
