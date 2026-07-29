"""Match current local error facts to canonical paper components."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.research.component_aliases import ResolvedComponentAlias, normalize_component_id


class AdapterDiagnosisContext(BaseModel):
    signals: list[str] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)
    source_fact_count: int = 0
    small_object_false_negative: bool = False


class AdapterDiagnosisMatch(BaseModel):
    matched: bool
    targets: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def build_adapter_diagnosis_context(facts: Iterable[ErrorFact]) -> AdapterDiagnosisContext:
    signals: set[str] = set()
    metrics: set[str] = set()
    current = [item for item in facts if item.evidence_role == "current_observation"]
    for fact in current:
        values = [
            fact.fact_type,
            fact.subject,
            fact.class_name,
            fact.area,
            fact.metric_name,
            *fact.action_candidates,
        ]
        normalized = {normalize_component_id(str(value)) for value in values if value}
        signals.update(normalized)
        if fact.metric_name:
            metrics.add(normalize_component_id(fact.metric_name))
        if fact.area == "small" or "ap_small" in normalized or any("small_object" in item for item in normalized):
            signals.update({"small_object", "small_object_false_negative"})
        if fact.fact_type == "false_negative" or any(item in {"fn", "false_negative"} or "false_negative" in item for item in normalized):
            signals.update({"false_negative", "small_object_false_negative"} if "small_object" in signals else {"false_negative"})
    return AdapterDiagnosisContext(
        signals=sorted(signals),
        target_metrics=sorted(metrics),
        source_fact_count=len(current),
        small_object_false_negative=(
            "small_object" in signals
            and ("false_negative" in signals or "small_object_false_negative" in signals)
        ),
    )


def match_component_to_diagnosis(
    mapping: ResolvedComponentAlias,
    context: AdapterDiagnosisContext,
) -> AdapterDiagnosisMatch:
    component_targets = {
        normalize_component_id(item)
        for item in [*mapping.target_error_types, *mapping.target_metrics]
    }
    matched = sorted(component_targets.intersection({*context.signals, *context.target_metrics}))
    component_id = mapping.canonical_component_id
    if context.small_object_false_negative and component_id in {
        "sampling.small_object",
        "head.p2_small_object",
        "distillation.yolo26_teacher_student",
        "inference.sahi_slicing",
    }:
        matched.append("small_object_false_negative_priority")
    return AdapterDiagnosisMatch(
        matched=bool(matched),
        targets=sorted(set(matched)),
        reasons=[f"diagnosis_match:{item}" for item in sorted(set(matched))],
    )


__all__ = [
    "AdapterDiagnosisContext",
    "AdapterDiagnosisMatch",
    "build_adapter_diagnosis_context",
    "match_component_to_diagnosis",
]
