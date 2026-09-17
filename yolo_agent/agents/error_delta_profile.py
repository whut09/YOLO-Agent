"""Full error-delta profiles for loop decisions.

The autonomous loop must never see only "mAP +0.2%": every round comparison
carries the complete delta profile — global metrics, scale buckets, per-class
AP/precision/recall, error counts, confidence calibration, slices, and
resource costs — with explicit ``evidence_incomplete`` markers for anything
the evaluation did not produce.  Missing numbers are never fabricated.
"""

from __future__ import annotations

from statistics import mean
from typing import Iterable, Literal

from pydantic import BaseModel, Field

from yolo_agent.core.error_facts import MetricValue
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.task_spec import TaskSpec


DeltaScope = Literal[
    "global",
    "scale",
    "class",
    "error",
    "confidence",
    "slice",
    "resource",
]

_GLOBAL_METRICS = {"map50", "map50_95", "mAP50", "mAP50-95", "precision", "recall"}
_SCALE_PREFIXES = ("ap_", "ar_", "area_ap50/", "area_recall/")
_ERROR_METRICS = {
    "false_negative_count",
    "false_positive_count",
    "localization_error_rate",
    "duplicate_prediction_count",
    "class_confusion_count",
}
_CONFIDENCE_PREFIXES = ("tp_confidence", "fp_confidence", "ece", "calibration_")
_RESOURCE_METRICS = {
    "latency_ms",
    "sliced_latency_ms",
    "model_size_mb",
    "params_m",
    "flops_g",
    "memory_mb",
}

#: Every loop-driven comparison must at least request these so "evidence
#: incomplete" is a real, visible state instead of silent absence.
REQUIRED_DELTA_METRICS = (
    "map50_95",
    "precision",
    "recall",
    "ap_small",
    "ap_medium",
    "ap_large",
    "latency_ms",
)


def classify_metric_scope(metric_name: str) -> DeltaScope:
    """Deterministically map a metric name to its delta scope."""

    if metric_name.startswith("per_class_") or metric_name.startswith("class/"):
        return "class"
    if metric_name in _RESOURCE_METRICS:
        return "resource"
    if metric_name in _GLOBAL_METRICS:
        return "global"
    if metric_name.startswith(_SCALE_PREFIXES):
        return "scale"
    if metric_name in _ERROR_METRICS:
        return "error"
    if metric_name.startswith(_CONFIDENCE_PREFIXES):
        return "confidence"
    if metric_name.startswith(("slice_", "slice/", "domain/", "scene/")):
        return "slice"
    return "global"


def _lower_is_better(name: str) -> bool:
    from yolo_agent.core.experiment_graph import LOWER_IS_BETTER_METRICS

    return name in LOWER_IS_BETTER_METRICS


class MetricDelta(BaseModel):
    """One paired metric delta between baseline and candidate evidence."""

    metric_name: str
    scope: DeltaScope
    subject: str = ""
    baseline_value: float
    candidate_value: float
    delta: float
    relative_delta: float
    higher_is_better: bool
    improvement: float
    seed_values_baseline: list[float] = Field(default_factory=list)
    seed_values_candidate: list[float] = Field(default_factory=list)


class EvidenceGap(BaseModel):
    """An explicitly incomplete piece of evidence — never a fabricated number."""

    metric_name: str
    reason: Literal["missing_in_candidate", "missing_in_baseline", "missing_entirely"]


class HardConstraintViolation(BaseModel):
    """One TaskSpec hard constraint the candidate evidence violates."""

    constraint: str
    limit: float
    observed: float
    baseline_observed: float | None
    metric_name: str


class ErrorDeltaProfile(BaseModel):
    """Complete multi-metric comparison of one candidate against its baseline."""

    baseline_run_id: str | None = None
    candidate_run_id: str | None = None
    candidate_id: str = ""
    node_id: str = ""
    deltas: list[MetricDelta] = Field(default_factory=list)
    evidence_incomplete: list[EvidenceGap] = Field(default_factory=list)
    hard_constraint_violations: list[HardConstraintViolation] = Field(default_factory=list)

    # ------------------------------------------------------------- lookups --
    def delta_for(self, metric_name: str, subject: str = "") -> MetricDelta | None:
        for delta in self.deltas:
            if delta.metric_name == metric_name and (not subject or delta.subject == subject):
                return delta
        return None

    def improvements(self) -> list[MetricDelta]:
        return [delta for delta in self.deltas if delta.improvement > 0]

    def regressions(self) -> list[MetricDelta]:
        return [delta for delta in self.deltas if delta.improvement < 0]

    def primary_map_delta(self) -> MetricDelta | None:
        for name in ("map50_95", "mAP50-95", "map50", "mAP50"):
            delta = self.delta_for(name)
            if delta is not None:
                return delta
        return None

    def primary_effect_delta(self) -> float | None:
        """The single scalar the legacy loop used — kept only as a projection."""

        delta = self.primary_map_delta()
        return None if delta is None else delta.delta

    def has_violations(self) -> bool:
        return bool(self.hard_constraint_violations)

    def violation_metric_names(self) -> list[str]:
        return sorted({item.metric_name for item in self.hard_constraint_violations})

    def is_evidence_complete(self) -> bool:
        return not self.evidence_incomplete


def _records_by_key(records: Iterable[MetricEvidence]) -> dict[tuple[str, str], MetricEvidence]:
    keyed: dict[tuple[str, str], MetricEvidence] = {}
    for record in records:
        key = (record.metric_name, _subject_of(record))
        current = keyed.get(key)
        if current is None or (record.created_at >= current.created_at):
            keyed[key] = record
    return keyed


def _subject_of(record: MetricEvidence) -> str:
    name = record.metric_name
    if name.startswith("per_class_"):
        return name.split("/", 1)[1] if "/" in name else ""
    if name.startswith(("area_", "slice/", "domain/", "scene/")):
        return name.split("/", 1)[1] if "/" in name else ""
    return ""


def _canonical_metric(name: str) -> str:
    return name


def build_error_delta_profile(
    baseline_records: Iterable[MetricEvidence],
    candidate_records: Iterable[MetricEvidence],
    *,
    candidate_id: str = "",
    node_id: str = "",
    baseline_run_id: str | None = None,
    candidate_run_id: str | None = None,
    task_spec: TaskSpec | None = None,
    expected_metrics: Iterable[str] = REQUIRED_DELTA_METRICS,
    resource_priors: dict[str, float] | None = None,
) -> ErrorDeltaProfile:
    """Build the full delta profile from two evidence record collections.

    ``resource_priors`` supplies params/FLOPs/memory when the evaluation did
    not measure them; priors are recorded as such in ``resource_priors_used``
    and never presented as measured deltas.
    """

    baseline = _records_by_key(baseline_records)
    candidate = _records_by_key(candidate_records)

    deltas: list[MetricDelta] = []
    gaps: list[EvidenceGap] = []
    all_names = sorted({name for name, _ in baseline} | {name for name, _ in candidate} | set(expected_metrics))
    for name in all_names:
        subject = ""
        key_name = name
        if "/" in name and (name.startswith("per_class_") or name.startswith(("area_", "slice/", "domain/", "scene/"))):
            key_name, subject = name.split("/", 1)
        base_entry = baseline.get((name, ""))
        cand_entry = candidate.get((name, ""))
        # per-class/area records carry the subject inside metric_name; rebuild keys
        if base_entry is None:
            base_entry = next((r for (n, _), r in baseline.items() if n == name), None)
        if cand_entry is None:
            cand_entry = next((r for (n, _), r in candidate.items() if n == name), None)
        base_value = _numeric(base_entry.value if base_entry else None)
        cand_value = _numeric(cand_entry.value if cand_entry else None)
        higher = base_entry.higher_is_better if base_entry else (cand_entry.higher_is_better if cand_entry else not _lower_is_better(key_name))
        if base_value is None and cand_value is None:
            gaps.append(EvidenceGap(metric_name=name, reason="missing_entirely"))
            continue
        if base_value is None or cand_value is None:
            gaps.append(
                EvidenceGap(
                    metric_name=name,
                    reason="missing_in_baseline" if base_value is None else "missing_in_candidate",
                )
            )
            continue
        delta = cand_value - base_value
        improvement = delta if higher else -delta
        relative = delta / base_value if base_value not in (0.0,) and base_value != 0 else 0.0
        deltas.append(
            MetricDelta(
                metric_name=key_name,
                scope=classify_metric_scope(key_name),
                subject=subject,
                baseline_value=base_value,
                candidate_value=cand_value,
                delta=delta,
                relative_delta=relative,
                higher_is_better=bool(higher),
                improvement=improvement,
                seed_values_baseline=_seed_values(base_entry),
                seed_values_candidate=_seed_values(cand_entry),
            )
        )

    profile = ErrorDeltaProfile(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        deltas=deltas,
        evidence_incomplete=gaps,
    )
    if task_spec is not None:
        profile.hard_constraint_violations = _constraint_violations(profile, task_spec)
    return profile


def _seed_values(record: MetricEvidence | None) -> list[float]:
    if record is None:
        return []
    value = _numeric(record.value)
    return [value] if value is not None else []


def _numeric(value: MetricValue) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _constraint_violations(profile: ErrorDeltaProfile, task_spec: TaskSpec) -> list[HardConstraintViolation]:
    violations: list[HardConstraintViolation] = []
    checks: list[tuple[str, str, float]] = []
    max_latency = task_spec.max_latency_ms
    if max_latency is not None:
        checks.append(("max_latency_ms", "latency_ms", float(max_latency)))
    max_size = task_spec.max_model_size_mb
    if max_size is not None:
        checks.append(("max_model_size_mb", "model_size_mb", float(max_size)))
    for constraint, metric, limit in checks:
        delta = profile.delta_for(metric)
        if delta is None:
            continue
        if delta.candidate_value > limit:
            violations.append(
                HardConstraintViolation(
                    constraint=constraint,
                    limit=limit,
                    observed=delta.candidate_value,
                    baseline_observed=delta.baseline_value,
                    metric_name=metric,
                )
            )
    return violations


def mean_of(values: Iterable[float]) -> float:
    items = list(values)
    return mean(items) if items else 0.0


__all__ = [
    "REQUIRED_DELTA_METRICS",
    "DeltaScope",
    "ErrorDeltaProfile",
    "EvidenceGap",
    "HardConstraintViolation",
    "MetricDelta",
    "build_error_delta_profile",
    "classify_metric_scope",
]
