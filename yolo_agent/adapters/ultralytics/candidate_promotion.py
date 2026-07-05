"""Candidate pilot promotion gate for full-budget Ultralytics runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import Evidence, MetricEvidence, MetricValue


class CandidatePromotionConfig(BaseModel):
    """Policy for promoting a candidate from pilot to full COCO budget."""

    enabled: bool = True
    required_debug_metric: str = "fast_baseline_sanity_passed"
    required_pilot_metric: str = "fast_baseline_pilot_passed"
    debug_profile: str = "debug"
    pilot_profile: str = "pilot"
    baseline_candidate_patterns: list[str] = Field(default_factory=lambda: ["baseline"])
    minimum_improved_error_facts: int = Field(default=1, ge=0)
    max_latency_regression_ratio: float = Field(default=0.10, ge=0.0)
    max_epoch_time_regression_ratio: float = Field(default=0.20, ge=0.0)
    max_runtime_throughput_drop_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    require_runtime_comparison: bool = True
    latency_metric: str = "latency_ms"
    runtime_throughput_metric: str = "runtime_avg_it_per_sec"
    epoch_time_metric: str = "runtime_epoch_time_seconds"


class ImprovedErrorFact(BaseModel):
    """One target error fact improved by candidate pilot evidence."""

    fact_key: str
    trend: str
    baseline_value: float | None = None
    candidate_value: float | None = None
    baseline_severity: str | None = None
    candidate_severity: str | None = None
    action_candidates: list[str] = Field(default_factory=list)


class CandidatePromotionResult(BaseModel):
    """Decision from candidate pilot promotion."""

    candidate_id: str
    candidate_full_allowed: bool
    candidate_promotion_rejection_reason: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug_nodes: list[str] = Field(default_factory=list)
    pilot_nodes: list[str] = Field(default_factory=list)
    baseline_nodes: list[str] = Field(default_factory=list)
    improved_error_facts: list[ImprovedErrorFact] = Field(default_factory=list)
    runtime_comparisons: dict[str, dict[str, float]] = Field(default_factory=dict)
    target_actions: list[str] = Field(default_factory=list)
    target_error_facts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CandidatePromotionGate:
    """Decide whether a candidate pilot can be promoted to candidate_full."""

    def __init__(self, config: CandidatePromotionConfig | None = None) -> None:
        self.config = config or CandidatePromotionConfig()

    def check(
        self,
        evidence: Evidence,
        error_facts: list[ErrorFact],
        candidate_id: str,
        target_actions: list[str] | None = None,
        target_error_facts: list[dict[str, Any]] | None = None,
    ) -> CandidatePromotionResult:
        """Return whether a candidate has earned full-budget promotion."""
        target_fact_values = list(target_error_facts or [])
        if not self.config.enabled:
            return CandidatePromotionResult(
                candidate_id=candidate_id,
                candidate_full_allowed=True,
                warnings=["Candidate promotion gate is disabled."],
                target_actions=list(target_actions or []),
                target_error_facts=target_fact_values,
            )

        target_action_values = list(dict.fromkeys(target_actions or []))
        target_fact_keys = _target_fact_keys(target_fact_values)
        debug_nodes = _nodes_with_metric(evidence.metric_records, candidate_id, self.config.required_debug_metric)
        pilot_nodes = _nodes_with_metric(evidence.metric_records, candidate_id, self.config.required_pilot_metric)
        baseline_nodes = _baseline_nodes(evidence.metric_records, self.config.baseline_candidate_patterns)
        reasons: list[str] = []
        warnings: list[str] = []

        if not debug_nodes:
            reasons.append("missing_candidate_debug_passed")
        if not pilot_nodes:
            reasons.append("missing_candidate_pilot_passed")
        if not baseline_nodes:
            reasons.append("missing_baseline_reference_nodes")

        improved = _improved_error_facts(
            error_facts=error_facts,
            candidate_id=candidate_id,
            baseline_nodes=baseline_nodes,
            candidate_nodes=pilot_nodes,
            target_actions=target_action_values,
            target_error_facts=target_fact_values,
        )
        required_improved = max(self.config.minimum_improved_error_facts, len(target_fact_keys))
        if target_fact_values and not target_fact_keys:
            reasons.append("invalid_target_error_facts")
        if len(improved) < required_improved:
            reasons.append(
                f"insufficient_target_error_fact_improvement:{len(improved)}/{required_improved}"
            )

        runtime_comparisons, runtime_reasons, runtime_warnings = _runtime_regression_checks(
            evidence,
            candidate_id=candidate_id,
            baseline_nodes=baseline_nodes,
            candidate_nodes=pilot_nodes,
            config=self.config,
        )
        reasons.extend(runtime_reasons)
        warnings.extend(runtime_warnings)

        return CandidatePromotionResult(
            candidate_id=candidate_id,
            candidate_full_allowed=not reasons,
            candidate_promotion_rejection_reason=list(dict.fromkeys(reasons)),
            warnings=list(dict.fromkeys(warnings)),
            debug_nodes=sorted(debug_nodes),
            pilot_nodes=sorted(pilot_nodes),
            baseline_nodes=sorted(baseline_nodes),
            improved_error_facts=improved,
            runtime_comparisons=runtime_comparisons,
            target_actions=target_action_values,
            target_error_facts=target_fact_values,
        )

    def persist_decisions(
        self,
        store: EvidenceStore,
        run_id: str,
        results: list[CandidatePromotionResult],
        dataset_version: str = "unversioned",
    ) -> Path:
        """Persist candidate promotion decisions as artifact and node-level evidence."""
        artifact_path = store.create_run(run_id) / "artifacts" / "candidate_promotion.json"
        artifact_path.write_text(
            json.dumps([result.model_dump(mode="json") for result in results], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        store.log_artifact_manifest(
            run_id=run_id,
            name="candidate_promotion",
            artifact_path=artifact_path,
            producer_stage="candidate_promotion_gate",
        )
        for result in results:
            store.log_candidate_metrics(
                run_id=run_id,
                candidate_id=result.candidate_id,
                node_id=f"candidate_promotion_{result.candidate_id}",
                metrics={
                    "candidate_full_allowed": result.candidate_full_allowed,
                    "candidate_promotion_rejection_reason": ";".join(result.candidate_promotion_rejection_reason),
                    "candidate_promotion_improved_error_fact_count": len(result.improved_error_facts),
                    "candidate_promotion_target_error_fact_count": len(result.target_error_facts),
                },
                dataset_version=dataset_version,
                split="runtime",
                source="candidate_promotion_gate",
                verified=True,
                validator="candidate_promotion_gate",
                source_artifact=artifact_path,
            )
        return artifact_path


def _nodes_with_metric(records: list[MetricEvidence], candidate_id: str, metric_name: str) -> set[str]:
    return {
        record.node_id
        for record in records
        if record.candidate_id == candidate_id
        and record.metric_name == metric_name
        and record.value is True
        and record.verified
    }


def _baseline_nodes(records: list[MetricEvidence], patterns: list[str]) -> set[str]:
    lowered_patterns = [pattern.lower() for pattern in patterns]
    nodes: set[str] = set()
    for record in records:
        candidate = record.candidate_id.lower()
        node = record.node_id.lower()
        if any(pattern in candidate or pattern in node for pattern in lowered_patterns):
            nodes.add(record.node_id)
    return nodes


def _improved_error_facts(
    error_facts: list[ErrorFact],
    candidate_id: str,
    baseline_nodes: set[str],
    candidate_nodes: set[str],
    target_actions: list[str],
    target_error_facts: list[dict[str, Any]],
) -> list[ImprovedErrorFact]:
    candidate_facts = [
        fact
        for fact in error_facts
        if fact.candidate_id == candidate_id and (not candidate_nodes or fact.node_id in candidate_nodes)
    ]
    if not candidate_facts:
        return []
    target_keys = _target_fact_keys(target_error_facts)
    baseline = {}
    for fact in error_facts:
        key = _fact_key(fact)
        if fact.node_id not in baseline_nodes:
            continue
        if target_keys:
            if key not in target_keys:
                continue
        elif not _targeted(fact, target_actions):
            continue
        baseline[key] = fact
    candidate = {
        _fact_key(fact): fact
        for fact in candidate_facts
    }
    improved: list[ImprovedErrorFact] = []
    for key, baseline_fact in baseline.items():
        candidate_fact = candidate.get(key)
        trend = _trend(baseline_fact, candidate_fact)
        if trend not in {"improved", "resolved"}:
            continue
        improved.append(
            ImprovedErrorFact(
                fact_key="|".join(key),
                trend=trend,
                baseline_value=_compare_value(baseline_fact),
                candidate_value=_compare_value(candidate_fact) if candidate_fact is not None else None,
                baseline_severity=baseline_fact.severity,
                candidate_severity=candidate_fact.severity if candidate_fact is not None else None,
                action_candidates=list(baseline_fact.action_candidates),
            )
        )
    return sorted(improved, key=lambda item: item.fact_key)


def _targeted(fact: ErrorFact, target_actions: list[str]) -> bool:
    if not target_actions:
        return fact.severity in {"high", "medium"}
    return bool(set(fact.action_candidates) & set(target_actions))


def _target_fact_keys(target_error_facts: list[dict[str, Any]]) -> set[tuple[str, str, str, str, str, str]]:
    """Return stable keys for explicitly bound target error facts."""
    keys: set[tuple[str, str, str, str, str, str]] = set()
    for item in target_error_facts:
        key = (
            str(item.get("fact_type") or ""),
            str(item.get("subject") or ""),
            str(item.get("class_name") or ""),
            str(item.get("class_pair") or ""),
            str(item.get("area") or ""),
            str(item.get("metric_name") or ""),
        )
        if key[0] and key[1]:
            keys.add(key)
    return keys


def _runtime_regression_checks(
    evidence: Evidence,
    candidate_id: str,
    baseline_nodes: set[str],
    candidate_nodes: set[str],
    config: CandidatePromotionConfig,
) -> tuple[dict[str, dict[str, float]], list[str], list[str]]:
    index = EvidenceIndex(evidence.metric_records)
    comparisons: dict[str, dict[str, float]] = {}
    reasons: list[str] = []
    warnings: list[str] = []
    checks = [
        (config.latency_metric, False, config.max_latency_regression_ratio, "latency_regression"),
        (config.runtime_throughput_metric, True, config.max_runtime_throughput_drop_ratio, "runtime_throughput_regression"),
        (config.epoch_time_metric, False, config.max_epoch_time_regression_ratio, "epoch_time_regression"),
    ]
    comparable_count = 0
    for metric_name, higher_is_better, max_regression, reason_name in checks:
        baseline_value = _selected_value(index, baseline_nodes, metric_name)
        candidate_value = _selected_value(index, candidate_nodes, metric_name, candidate_id=candidate_id)
        if baseline_value is None or candidate_value is None:
            continue
        comparable_count += 1
        comparisons[metric_name] = {"baseline": baseline_value, "candidate": candidate_value}
        if higher_is_better:
            floor = baseline_value * (1.0 - max_regression)
            if candidate_value < floor:
                reasons.append(f"{reason_name}:{candidate_value:.6g}<{floor:.6g}")
        else:
            ceiling = baseline_value * (1.0 + max_regression)
            if candidate_value > ceiling:
                reasons.append(f"{reason_name}:{candidate_value:.6g}>{ceiling:.6g}")
    if comparable_count == 0:
        message = "missing_runtime_or_latency_comparison"
        if config.require_runtime_comparison:
            reasons.append(message)
        else:
            warnings.append(message)
    return comparisons, reasons, warnings


def _selected_value(
    index: EvidenceIndex,
    node_ids: set[str],
    metric_name: str,
    candidate_id: str | None = None,
) -> float | None:
    values: list[float] = []
    for node_id in node_ids:
        record = index.select_one(
            candidate_id=candidate_id,
            node_id=node_id,
            metric_name=metric_name,
            verified=True,
        )
        numeric = _numeric(record.value) if record is not None else None
        if numeric is not None:
            values.append(numeric)
    if not values:
        return None
    return sum(values) / len(values)


def _fact_key(fact: ErrorFact) -> tuple[str, str, str, str, str, str]:
    return (
        fact.fact_type,
        fact.subject,
        fact.class_name or "",
        fact.class_pair or "",
        fact.area or "",
        fact.metric_name or "",
    )


def _trend(baseline: ErrorFact, candidate: ErrorFact | None) -> str:
    if candidate is None:
        return "resolved"
    baseline_value = _compare_value(baseline)
    candidate_value = _compare_value(candidate)
    if baseline_value is None or candidate_value is None:
        baseline_severity = _severity_score(baseline.severity)
        candidate_severity = _severity_score(candidate.severity)
        if candidate_severity < baseline_severity:
            return "improved"
        if candidate_severity > baseline_severity:
            return "regressed"
        return "unchanged"
    delta = candidate_value - baseline_value
    if abs(delta) <= 1e-9:
        return "unchanged"
    if _higher_is_better(baseline):
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def _compare_value(fact: ErrorFact | None) -> float | None:
    if fact is None:
        return None
    value = _numeric(fact.value)
    if value is not None:
        return value
    return float(fact.count) if fact.count is not None else None


def _higher_is_better(fact: ErrorFact) -> bool:
    if fact.count is not None and fact.value is None:
        return False
    if fact.fact_type in {
        "false_negative_heavy_class",
        "localization_heavy_class",
        "class_confusion_pair",
        "background_false_positive_class",
    }:
        return False
    return True


def _severity_score(severity: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(severity, 1)


def _numeric(value: MetricValue) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None
