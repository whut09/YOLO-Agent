"""Select COCO baseline error facts that should drive the next experiment round."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import MetricValue


CocoDiagnosisKind = Literal[
    "small_object_ap",
    "medium_object_ap",
    "large_object_ap",
    "class_recall",
    "class_low_ap",
    "false_negative_class",
    "localization_class",
    "background_fp_class",
    "class_confusion",
    "subset_performance",
    "generic_error_fact",
]


class CocoErrorFocus(BaseModel):
    """One selected unresolved COCO diagnosis for the next round."""

    diagnosis_id: str
    diagnosis_kind: CocoDiagnosisKind
    fact_type: str
    subject: str
    class_name: str | None = None
    class_pair: str | None = None
    area: str | None = None
    metric_name: str | None = None
    value: MetricValue = None
    count: int | None = None
    severity: str
    priority: float
    action_candidates: list[str] = Field(default_factory=list)
    target_error_key: str
    candidate_id: str
    node_id: str
    reason: str = ""


class CocoErrorSelectionResult(BaseModel):
    """Selected COCO error facts and this-round optimization focus."""

    baseline_node_ids: list[str] = Field(default_factory=list)
    top_unresolved_diagnoses: list[CocoErrorFocus] = Field(default_factory=list)
    current_round_focus: list[CocoErrorFocus] = Field(default_factory=list)
    focus_action_candidates: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CocoErrorFactSelector:
    """Pick high-signal baseline COCO errors for pilot-only next-round proposals."""

    def __init__(
        self,
        max_diagnoses: int = 8,
        max_focus: int = 4,
    ) -> None:
        self.max_diagnoses = max_diagnoses
        self.max_focus = max_focus

    def select(
        self,
        facts: list[ErrorFact],
        baseline_node_ids: list[str] | None = None,
    ) -> CocoErrorSelectionResult:
        """Select top unresolved baseline error facts."""
        baseline_nodes = list(dict.fromkeys(baseline_node_ids or _infer_baseline_node_ids(facts)))
        warnings: list[str] = []
        if baseline_nodes:
            baseline_facts = [fact for fact in facts if fact.node_id in set(baseline_nodes)]
        else:
            baseline_facts = []
            warnings.append("No baseline COCO error facts found; next-round proposals must not claim targeted learning.")

        candidates = [
            _focus_from_fact(fact)
            for fact in baseline_facts
            if fact.severity in {"high", "medium"}
        ]
        deduped = _dedupe_focus(candidates)
        ranked = sorted(deduped, key=lambda item: (-item.priority, item.diagnosis_id))
        top = ranked[: self.max_diagnoses]
        focus = _diverse_focus(top, self.max_focus)
        return CocoErrorSelectionResult(
            baseline_node_ids=baseline_nodes,
            top_unresolved_diagnoses=top,
            current_round_focus=focus,
            focus_action_candidates=_actions(focus),
            warnings=warnings,
        )


def select_coco_error_facts(
    facts: list[ErrorFact],
    baseline_node_ids: list[str] | None = None,
    max_diagnoses: int = 8,
    max_focus: int = 4,
) -> CocoErrorSelectionResult:
    """Convenience wrapper for selecting next-round COCO focus facts."""
    return CocoErrorFactSelector(max_diagnoses=max_diagnoses, max_focus=max_focus).select(
        facts,
        baseline_node_ids=baseline_node_ids,
    )


def _focus_from_fact(fact: ErrorFact) -> CocoErrorFocus:
    kind = _diagnosis_kind(fact)
    priority = _priority(fact, kind)
    key = _target_key(fact, kind)
    return CocoErrorFocus(
        diagnosis_id=key,
        diagnosis_kind=kind,
        fact_type=fact.fact_type,
        subject=fact.subject,
        class_name=fact.class_name,
        class_pair=fact.class_pair,
        area=fact.area,
        metric_name=fact.metric_name,
        value=fact.value,
        count=fact.count,
        severity=fact.severity,
        priority=priority,
        action_candidates=list(fact.action_candidates),
        target_error_key=key,
        candidate_id=fact.candidate_id,
        node_id=fact.node_id,
        reason=_reason(fact, kind),
    )


def _diagnosis_kind(fact: ErrorFact) -> CocoDiagnosisKind:
    if fact.fact_type in {"area_metric", "subset_performance"} and fact.area == "small":
        return "small_object_ap"
    if fact.fact_type in {"area_metric", "subset_performance"} and fact.area == "medium":
        return "medium_object_ap"
    if fact.fact_type in {"area_metric", "subset_performance"} and fact.area == "large":
        return "large_object_ap"
    if fact.fact_type == "per_class_metric" and fact.metric_name == "per_class_ar":
        return "class_recall"
    if fact.fact_type == "class_low_ap":
        return "class_low_ap"
    if fact.fact_type == "false_negative_heavy_class":
        return "false_negative_class"
    if fact.fact_type == "localization_heavy_class":
        return "localization_class"
    if fact.fact_type == "background_false_positive_class":
        return "background_fp_class"
    if fact.fact_type == "class_confusion_pair":
        return "class_confusion"
    if fact.fact_type == "subset_performance":
        return "subset_performance"
    return "generic_error_fact"


def _priority(fact: ErrorFact, kind: CocoDiagnosisKind) -> float:
    severity = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(fact.severity, 1.0)
    kind_bonus = {
        "small_object_ap": 1.0,
        "class_recall": 0.9,
        "false_negative_class": 0.85,
        "localization_class": 0.8,
        "class_low_ap": 0.7,
        "background_fp_class": 0.55,
        "class_confusion": 0.5,
        "medium_object_ap": 0.35,
        "large_object_ap": 0.2,
        "subset_performance": 0.25,
        "generic_error_fact": 0.0,
    }[kind]
    value_penalty = _numeric(fact.value)
    low_metric_bonus = 0.0 if value_penalty is None else max(0.0, 1.0 - value_penalty)
    count_bonus = min(0.5, (fact.count or 0) / 100.0)
    rank_bonus = 0.0 if fact.rank is None else max(0.0, 0.5 - fact.rank * 0.03)
    return round(severity + kind_bonus + low_metric_bonus + count_bonus + rank_bonus, 6)


def _reason(fact: ErrorFact, kind: CocoDiagnosisKind) -> str:
    if kind == "small_object_ap":
        return "Small-object AP is an unresolved baseline weakness."
    if kind == "class_recall":
        return f"{fact.class_name or fact.subject} recall is low in baseline COCO eval."
    if kind == "localization_class":
        return f"{fact.class_name or fact.subject} has localization-heavy errors."
    if kind == "false_negative_class":
        return f"{fact.class_name or fact.subject} has many false negatives."
    if kind == "class_low_ap":
        return f"{fact.class_name or fact.subject} has low per-class AP."
    if kind == "background_fp_class":
        return f"{fact.class_name or fact.subject} has background false positives."
    if kind == "class_confusion":
        return f"{fact.class_pair or fact.subject} is a class-confusion pair."
    return "Baseline COCO eval produced an unresolved error fact."


def _target_key(fact: ErrorFact, kind: CocoDiagnosisKind) -> str:
    parts = [
        kind,
        fact.class_name or fact.class_pair or fact.area or fact.subject,
        fact.metric_name or "",
    ]
    return ":".join(part for part in parts if part)


def _infer_baseline_node_ids(facts: list[ErrorFact]) -> list[str]:
    nodes: list[str] = []
    for fact in facts:
        text = f"{fact.candidate_id} {fact.node_id}".lower()
        if "baseline" in text:
            nodes.append(fact.node_id)
    return list(dict.fromkeys(nodes))


def _dedupe_focus(items: list[CocoErrorFocus]) -> list[CocoErrorFocus]:
    by_key: dict[str, CocoErrorFocus] = {}
    for item in items:
        previous = by_key.get(item.target_error_key)
        if previous is None or item.priority > previous.priority:
            by_key[item.target_error_key] = item
    return list(by_key.values())


def _diverse_focus(items: list[CocoErrorFocus], limit: int) -> list[CocoErrorFocus]:
    selected: list[CocoErrorFocus] = []
    seen_kinds: set[str] = set()
    for item in items:
        if len(selected) >= limit:
            break
        if item.diagnosis_kind in seen_kinds and len(seen_kinds) < min(limit, 3):
            continue
        selected.append(item)
        seen_kinds.add(item.diagnosis_kind)
    for item in items:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    return selected


def _actions(items: list[CocoErrorFocus]) -> list[str]:
    actions: list[str] = []
    for item in items:
        actions.extend(item.action_candidates)
    return list(dict.fromkeys(actions))


def _numeric(value: MetricValue) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None
