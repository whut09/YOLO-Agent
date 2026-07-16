"""Diagnosis-bound promotion checks for guarded candidate budgets."""

from __future__ import annotations

import math
from statistics import stdev
from typing import Any, Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.loop_evidence import error_fact_delta
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.matched_baseline import PairedMetricDelta, paired_metric_delta


DiagnosisCheckStatus = Literal["passed", "failed", "missing", "not_applicable"]


class DiagnosisPromotionPolicy(BaseModel):
    """Noise, regression, and evidence guards for diagnosis-bound promotion."""

    minimum_metric_noise_floor: float = Field(default=0.0005, ge=0.0)
    minimum_class_noise_floor: float = Field(default=0.0005, ge=0.0)
    noise_confidence_multiplier: float = Field(default=1.96, ge=0.0)
    minimum_fn_reduction: float = Field(default=1.0, ge=0.0)
    max_overall_map_regression: float = Field(default=0.005, ge=0.0)
    max_latency_regression: float = Field(default=0.05, ge=0.0)
    max_model_size_regression: float = Field(default=0.10, ge=0.0)
    require_related_classes_for_small_object: bool = True
    require_fn_reduction_for_small_object: bool = True


class DiagnosisPromotionCheck(BaseModel):
    """One traceable promotion condition."""

    check_id: str
    status: DiagnosisCheckStatus
    metric_name: str | None = None
    subject: str | None = None
    observed_delta: float | None = None
    required_delta: float | None = None
    baseline_value: float | None = None
    candidate_value: float | None = None
    reason: str


class DiagnosisPromotionResult(BaseModel):
    """Complete diagnosis-bound promotion decision."""

    candidate_id: str
    node_id: str
    allowed: bool
    target_metric: str | None = None
    related_classes: list[str] = Field(default_factory=list)
    checks: list[DiagnosisPromotionCheck] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    measurement_noise: dict[str, float] = Field(default_factory=dict)


class DiagnosisPromotionGate:
    """Require the candidate to improve the diagnosis it was created to solve."""

    def __init__(self, policy: DiagnosisPromotionPolicy | None = None) -> None:
        self.policy = policy or DiagnosisPromotionPolicy()

    def evaluate(
        self,
        *,
        candidate_id: str,
        node_id: str,
        target_error_facts: list[dict[str, Any]],
        metric_records: list[MetricEvidence],
        error_facts: list[ErrorFact],
    ) -> DiagnosisPromotionResult:
        """Evaluate target, class, FN, overall, latency, size, and noise guards."""
        target_metric = _target_metric(target_error_facts)
        related_classes = _related_classes(target_error_facts)
        checks: list[DiagnosisPromotionCheck] = []
        noise: dict[str, float] = {}

        if target_metric is None:
            checks.append(_missing("target_metric", "target diagnosis does not bind a measurable metric"))
        else:
            threshold = _measurement_noise(
                metric_records,
                metric_name=target_metric,
                policy=self.policy,
                class_metric=target_metric.startswith("per_class_"),
            )
            noise[target_metric] = threshold
            checks.append(
                _metric_improvement_check(
                    metric_records,
                    candidate_id=candidate_id,
                    node_id=node_id,
                    metric_name=target_metric,
                    threshold=threshold,
                    check_id="target_metric_improvement",
                )
            )

        small_object_target = target_metric == "ap_small" or _targets_small_objects(target_error_facts)
        if small_object_target:
            checks.extend(
                self._small_object_class_checks(
                    candidate_id=candidate_id,
                    node_id=node_id,
                    related_classes=related_classes,
                    metric_records=metric_records,
                    error_facts=error_facts,
                    noise=noise,
                )
            )
            checks.append(
                self._false_negative_check(
                    candidate_id=candidate_id,
                    node_id=node_id,
                    related_classes=related_classes,
                    error_facts=error_facts,
                )
            )

        checks.append(
            _non_regression_check(
                metric_records,
                candidate_id=candidate_id,
                node_id=node_id,
                metric_name="map50_95",
                maximum_regression=self.policy.max_overall_map_regression,
                check_id="overall_map_guard",
            )
        )
        checks.append(
            _ratio_guard_check(
                metric_records,
                candidate_id=candidate_id,
                node_id=node_id,
                metric_name="latency_ms",
                maximum_ratio=self.policy.max_latency_regression,
                check_id="latency_guard",
            )
        )
        checks.append(
            _ratio_guard_check(
                metric_records,
                candidate_id=candidate_id,
                node_id=node_id,
                metric_name="model_size_mb",
                maximum_ratio=self.policy.max_model_size_regression,
                check_id="model_size_guard",
            )
        )
        rejection_reasons = [
            f"{check.check_id}:{check.reason}"
            for check in checks
            if check.status in {"failed", "missing"}
        ]
        return DiagnosisPromotionResult(
            candidate_id=candidate_id,
            node_id=node_id,
            allowed=not rejection_reasons,
            target_metric=target_metric,
            related_classes=related_classes,
            checks=checks,
            rejection_reasons=rejection_reasons,
            measurement_noise=noise,
        )

    def _small_object_class_checks(
        self,
        *,
        candidate_id: str,
        node_id: str,
        related_classes: list[str],
        metric_records: list[MetricEvidence],
        error_facts: list[ErrorFact],
        noise: dict[str, float],
    ) -> list[DiagnosisPromotionCheck]:
        if not related_classes:
            if self.policy.require_related_classes_for_small_object:
                return [_missing("related_class_ap", "AP_small diagnosis has no bound small-object classes")]
            return [
                DiagnosisPromotionCheck(
                    check_id="related_class_ap",
                    status="not_applicable",
                    reason="related class guard disabled",
                )
            ]
        checks: list[DiagnosisPromotionCheck] = []
        for class_name in related_classes:
            metric_name = f"per_class_ap/{class_name}"
            threshold = _measurement_noise(
                metric_records,
                metric_name=metric_name,
                policy=self.policy,
                class_metric=True,
            )
            noise[metric_name] = threshold
            metric_check = _metric_improvement_check(
                metric_records,
                candidate_id=candidate_id,
                node_id=node_id,
                metric_name=metric_name,
                threshold=threshold,
                check_id="related_class_ap",
                subject=class_name,
            )
            if metric_check.status == "missing":
                metric_check = _fact_improvement_check(
                    error_facts,
                    candidate_id=candidate_id,
                    node_id=node_id,
                    fact_type="per_class_metric",
                    metric_name="per_class_ap",
                    class_name=class_name,
                    threshold=threshold,
                    check_id="related_class_ap",
                )
            checks.append(metric_check)
        return checks

    def _false_negative_check(
        self,
        *,
        candidate_id: str,
        node_id: str,
        related_classes: list[str],
        error_facts: list[ErrorFact],
    ) -> DiagnosisPromotionCheck:
        if not related_classes:
            return _missing("false_negative_reduction", "small-object FN guard has no bound classes")
        deltas = _candidate_error_deltas(error_facts, candidate_id=candidate_id, node_id=node_id)
        matching = [
            item
            for item in deltas.get("improved_errors", [])
            if item.get("fact_type") == "false_negative_heavy_class"
            and str(item.get("class_name") or item.get("subject") or "") in related_classes
            and isinstance(item.get("delta"), (int, float))
            and float(item["delta"]) <= -self.policy.minimum_fn_reduction
        ]
        if matching:
            best = min(matching, key=lambda item: float(item["delta"]))
            return DiagnosisPromotionCheck(
                check_id="false_negative_reduction",
                status="passed",
                metric_name="false_negative_count",
                subject=str(best.get("class_name") or best.get("subject") or ""),
                observed_delta=float(best["delta"]),
                required_delta=-self.policy.minimum_fn_reduction,
                reason="bound class false negatives decreased",
            )
        available = [
            fact
            for fact in error_facts
            if fact.fact_type == "false_negative_heavy_class"
            and fact.class_name in related_classes
            and fact.candidate_id == candidate_id
            and fact.node_id == node_id
        ]
        if not available:
            status: DiagnosisCheckStatus = (
                "missing" if self.policy.require_fn_reduction_for_small_object else "not_applicable"
            )
            return DiagnosisPromotionCheck(
                check_id="false_negative_reduction",
                status=status,
                metric_name="false_negative_count",
                reason="missing matched false-negative evidence for bound classes",
            )
        return DiagnosisPromotionCheck(
            check_id="false_negative_reduction",
            status="failed",
            metric_name="false_negative_count",
            required_delta=-self.policy.minimum_fn_reduction,
            reason="bound class false negatives did not decrease beyond the minimum",
        )


def _target_metric(targets: list[dict[str, Any]]) -> str | None:
    for target in targets:
        metric = str(target.get("metric_name") or "").strip()
        if metric in {"ap_small", "ap_medium", "ap_large"}:
            return metric
        area = str(target.get("area") or target.get("object_size") or target.get("subject") or "").lower()
        fact_type = str(target.get("fact_type") or "")
        if area == "small" and fact_type in {"area_metric", "false_negative", "false_negative_heavy_class"}:
            return "ap_small"
    for target in targets:
        metric = str(target.get("metric_name") or "").strip()
        class_name = str(target.get("class_name") or "").strip()
        if metric in {"per_class_ap", "class_ap"} and class_name:
            return f"per_class_ap/{class_name}"
        if metric:
            return metric
    return None


def _related_classes(targets: list[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(target.get("class_name") or "").strip()
            for target in targets
            if str(target.get("class_name") or "").strip()
        )
    )


def _targets_small_objects(targets: list[dict[str, Any]]) -> bool:
    return any(
        str(target.get("area") or target.get("object_size") or target.get("subject") or "").lower() == "small"
        for target in targets
    )


def _metric_improvement_check(
    records: list[MetricEvidence],
    *,
    candidate_id: str,
    node_id: str,
    metric_name: str,
    threshold: float,
    check_id: str,
    subject: str | None = None,
) -> DiagnosisPromotionCheck:
    delta = _paired_metric(records, candidate_id=candidate_id, node_id=node_id, metric_name=metric_name)
    if delta is None:
        return _missing(check_id, f"missing matched {metric_name}", metric_name=metric_name, subject=subject)
    passed = delta.effect_delta > threshold
    return DiagnosisPromotionCheck(
        check_id=check_id,
        status="passed" if passed else "failed",
        metric_name=metric_name,
        subject=subject,
        observed_delta=delta.effect_delta,
        required_delta=threshold,
        baseline_value=delta.baseline_value,
        candidate_value=delta.candidate_value,
        reason=(
            "improvement exceeds measurement noise"
            if passed
            else "improvement does not exceed measurement noise"
        ),
    )


def _fact_improvement_check(
    facts: list[ErrorFact],
    *,
    candidate_id: str,
    node_id: str,
    fact_type: str,
    metric_name: str,
    class_name: str,
    threshold: float,
    check_id: str,
) -> DiagnosisPromotionCheck:
    deltas = _candidate_error_deltas(facts, candidate_id=candidate_id, node_id=node_id)
    matching = [
        item
        for item in deltas.get("improved_errors", [])
        if item.get("fact_type") in {fact_type, "class_low_ap"}
        and item.get("metric_name") == metric_name
        and str(item.get("class_name") or item.get("subject") or "") == class_name
        and isinstance(item.get("delta"), (int, float))
    ]
    if not matching:
        return _missing(check_id, f"missing matched class AP for {class_name}", metric_name=metric_name, subject=class_name)
    best = max(matching, key=lambda item: float(item["delta"]))
    observed = float(best["delta"])
    return DiagnosisPromotionCheck(
        check_id=check_id,
        status="passed" if observed > threshold else "failed",
        metric_name=metric_name,
        subject=class_name,
        observed_delta=observed,
        required_delta=threshold,
        reason=(
            "class AP improvement exceeds measurement noise"
            if observed > threshold
            else "class AP improvement does not exceed measurement noise"
        ),
    )


def _non_regression_check(
    records: list[MetricEvidence],
    *,
    candidate_id: str,
    node_id: str,
    metric_name: str,
    maximum_regression: float,
    check_id: str,
) -> DiagnosisPromotionCheck:
    delta = _paired_metric(records, candidate_id=candidate_id, node_id=node_id, metric_name=metric_name)
    if delta is None:
        return _missing(check_id, f"missing matched {metric_name}", metric_name=metric_name)
    passed = delta.effect_delta >= -maximum_regression
    return DiagnosisPromotionCheck(
        check_id=check_id,
        status="passed" if passed else "failed",
        metric_name=metric_name,
        observed_delta=delta.effect_delta,
        required_delta=-maximum_regression,
        baseline_value=delta.baseline_value,
        candidate_value=delta.candidate_value,
        reason="overall metric remains within guard" if passed else "overall metric regressed beyond guard",
    )


def _ratio_guard_check(
    records: list[MetricEvidence],
    *,
    candidate_id: str,
    node_id: str,
    metric_name: str,
    maximum_ratio: float,
    check_id: str,
) -> DiagnosisPromotionCheck:
    delta = _paired_metric(records, candidate_id=candidate_id, node_id=node_id, metric_name=metric_name)
    if delta is None or delta.baseline_value == 0:
        return _missing(check_id, f"missing matched {metric_name}", metric_name=metric_name)
    regression = (delta.candidate_value - delta.baseline_value) / abs(delta.baseline_value)
    passed = regression <= maximum_ratio
    return DiagnosisPromotionCheck(
        check_id=check_id,
        status="passed" if passed else "failed",
        metric_name=metric_name,
        observed_delta=regression,
        required_delta=maximum_ratio,
        baseline_value=delta.baseline_value,
        candidate_value=delta.candidate_value,
        reason="resource regression remains within guard" if passed else "resource regression exceeds guard",
    )


def _paired_metric(
    records: list[MetricEvidence],
    *,
    candidate_id: str,
    node_id: str,
    metric_name: str,
) -> PairedMetricDelta | None:
    candidates = [
        record
        for record in records
        if record.candidate_id == candidate_id
        and record.node_id == node_id
        and record.metric_name == metric_name
        and record.evidence_role == "current_observation"
        and record.inheritance_depth == 0
        and record.verified
        and isinstance(record.value, (int, float))
    ]
    baselines = [
        record
        for record in records
        if record.metric_name == metric_name
        and record.evidence_role == "baseline_reference"
        and record.verified
        and isinstance(record.value, (int, float))
    ]
    if not candidates:
        return None
    candidate = max(candidates, key=lambda record: record.created_at)
    _, delta = paired_metric_delta(candidate, baselines)
    return delta


def _measurement_noise(
    records: list[MetricEvidence],
    *,
    metric_name: str,
    policy: DiagnosisPromotionPolicy,
    class_metric: bool,
) -> float:
    floor = policy.minimum_class_noise_floor if class_metric else policy.minimum_metric_noise_floor
    explicit_names = {f"measurement_noise/{metric_name}", f"{metric_name}_measurement_noise"}
    explicit = [
        abs(float(record.value))
        for record in records
        if record.metric_name in explicit_names
        and record.verified
        and isinstance(record.value, (int, float))
    ]
    if explicit:
        return max(floor, max(explicit))
    baseline_values = [
        float(record.value)
        for record in records
        if record.metric_name == metric_name
        and record.evidence_role == "baseline_reference"
        and record.verified
        and isinstance(record.value, (int, float))
    ]
    if len(baseline_values) < 3:
        return floor
    standard_error = stdev(baseline_values) / math.sqrt(len(baseline_values))
    return max(floor, policy.noise_confidence_multiplier * standard_error)


def _candidate_error_deltas(
    facts: list[ErrorFact],
    *,
    candidate_id: str,
    node_id: str,
) -> dict[str, Any]:
    baseline = [fact for fact in facts if fact.evidence_role == "baseline_reference"]
    current = [
        fact
        for fact in facts
        if fact.evidence_role == "current_observation"
        and fact.candidate_id == candidate_id
        and fact.node_id == node_id
    ]
    return error_fact_delta(baseline, current)


def _missing(
    check_id: str,
    reason: str,
    *,
    metric_name: str | None = None,
    subject: str | None = None,
) -> DiagnosisPromotionCheck:
    return DiagnosisPromotionCheck(
        check_id=check_id,
        status="missing",
        metric_name=metric_name,
        subject=subject,
        reason=reason,
    )
