"""Causal diagnosis graph for error-fact driven YOLO optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.core.error_facts import ErrorFact, ErrorFactType, ErrorSeverity
from yolo_agent.core.experiment_graph import MetricValue
from yolo_agent.resources import ResourcePaths
from yolo_agent.utils import dedupe_list


CauseConfidence = Literal["low", "medium", "high"]


class DiagnosisGraphMatch(BaseModel):
    """Rule predicates for matching error facts."""

    fact_types: list[ErrorFactType] = Field(default_factory=list)
    severities: list[ErrorSeverity] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    class_names: list[str] = Field(default_factory=list)
    class_pairs: list[str] = Field(default_factory=list)
    areas: list[str] = Field(default_factory=list)
    metric_names: list[str] = Field(default_factory=list)
    action_candidates_any: list[str] = Field(default_factory=list)


class DiagnosisCause(BaseModel):
    """One possible cause behind an observed detection symptom."""

    cause_id: str
    description: str
    evidence_needed: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    confidence: CauseConfidence = "medium"
    risks: list[str] = Field(default_factory=list)


class DiagnosisRule(BaseModel):
    """Configured mapping from symptoms to causal hypotheses."""

    rule_id: str
    symptom: str
    match: DiagnosisGraphMatch = Field(default_factory=DiagnosisGraphMatch)
    possible_causes: list[DiagnosisCause] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    rationale: str = ""


class MatchedErrorFact(BaseModel):
    """Compact error fact reference stored in diagnosis output."""

    fact_type: str
    subject: str
    class_name: str | None = None
    class_pair: str | None = None
    area: str | None = None
    metric_name: str | None = None
    value: MetricValue = None
    count: int | None = None
    severity: str
    action_candidates: list[str] = Field(default_factory=list)
    candidate_id: str
    node_id: str


class DiagnosisFinding(BaseModel):
    """One causal diagnosis produced from matched error facts."""

    diagnosis_id: str
    symptom: str
    priority: float
    matched_facts: list[MatchedErrorFact] = Field(default_factory=list)
    possible_causes: list[DiagnosisCause] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    rationale: str = ""


class DiagnosisGraphReport(BaseModel):
    """Causal graph diagnosis report for a set of error facts."""

    findings: list[DiagnosisFinding] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    action_candidates: list[str] = Field(default_factory=list)
    unmatched_error_facts: list[MatchedErrorFact] = Field(default_factory=list)


class DiagnosisGraph(BaseModel):
    """Configured symptom-to-cause diagnosis graph."""

    rules: list[DiagnosisRule] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "DiagnosisGraph":
        """Load a diagnosis graph from YAML."""
        graph_path = Path(path) if path is not None else ResourcePaths.DIAGNOSIS_GRAPH
        with graph_path.open("r", encoding="utf-8-sig") as file:
            raw = yaml.safe_load(file) or {}
        rules = raw.get("rules", raw) if isinstance(raw, dict) else raw
        if not isinstance(rules, list):
            raise ValueError(f"Diagnosis graph YAML must contain a rules list: {graph_path}")
        normalized = []
        for item in rules:
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("rule_id") or item.get("id") or "")
            normalized.append({**item, "rule_id": rule_id})
        return cls(rules=[DiagnosisRule.model_validate(item) for item in normalized])

    def diagnose(self, facts: list[ErrorFact], limit: int | None = None) -> DiagnosisGraphReport:
        """Diagnose likely causes from error facts."""
        findings: list[DiagnosisFinding] = []
        matched_fact_ids: set[int] = set()
        for rule in self.rules:
            matched = [fact for fact in facts if _matches_rule(fact, rule.match)]
            if not matched:
                continue
            matched_fact_ids.update(id(fact) for fact in matched)
            finding = _finding(rule, matched)
            findings.append(finding)

        ranked = sorted(findings, key=lambda item: (-item.priority, item.diagnosis_id))
        if limit is not None:
            ranked = ranked[:limit]
        evidence_needed: list[str] = []
        action_candidates: list[str] = []
        for finding in ranked:
            evidence_needed.extend(finding.evidence_needed)
            action_candidates.extend(finding.actions)
        unmatched = [_fact_ref(fact) for fact in facts if id(fact) not in matched_fact_ids]
        return DiagnosisGraphReport(
            findings=ranked,
            evidence_needed=dedupe_list(evidence_needed),
            action_candidates=dedupe_list(action_candidates),
            unmatched_error_facts=unmatched,
        )


def _finding(rule: DiagnosisRule, facts: list[ErrorFact]) -> DiagnosisFinding:
    evidence_needed = list(rule.evidence_needed)
    actions = list(rule.actions)
    for cause in rule.possible_causes:
        evidence_needed.extend(cause.evidence_needed)
        actions.extend(cause.actions)
    for fact in facts:
        actions.extend(fact.action_candidates)
    return DiagnosisFinding(
        diagnosis_id=rule.rule_id,
        symptom=rule.symptom,
        priority=_priority(facts),
        matched_facts=[_fact_ref(fact) for fact in sorted(facts, key=_fact_sort_key)],
        possible_causes=rule.possible_causes,
        evidence_needed=dedupe_list(evidence_needed),
        actions=dedupe_list(actions),
        rationale=rule.rationale,
    )


def _matches_rule(fact: ErrorFact, match: DiagnosisGraphMatch) -> bool:
    if match.fact_types and fact.fact_type not in set(match.fact_types):
        return False
    if match.severities and fact.severity not in set(match.severities):
        return False
    if match.subjects and fact.subject not in set(match.subjects):
        return False
    if match.class_names and fact.class_name not in set(match.class_names):
        return False
    if match.class_pairs and fact.class_pair not in set(match.class_pairs):
        return False
    if match.areas and fact.area not in set(match.areas):
        return False
    if match.metric_names and fact.metric_name not in set(match.metric_names):
        return False
    if match.action_candidates_any and not (set(match.action_candidates_any) & set(fact.action_candidates)):
        return False
    return True


def _priority(facts: list[ErrorFact]) -> float:
    severity_score = {"high": 3.0, "medium": 2.0, "low": 1.0}
    best = max((severity_score.get(fact.severity, 1.0) for fact in facts), default=1.0)
    count_bonus = min(0.75, len(facts) * 0.15)
    rank_bonus = max((max(0.0, 0.5 - (fact.rank or 999) * 0.03) for fact in facts), default=0.0)
    return round(best + count_bonus + rank_bonus, 6)


def _fact_ref(fact: ErrorFact) -> MatchedErrorFact:
    return MatchedErrorFact(
        fact_type=fact.fact_type,
        subject=fact.subject,
        class_name=fact.class_name,
        class_pair=fact.class_pair,
        area=fact.area,
        metric_name=fact.metric_name,
        value=fact.value,
        count=fact.count,
        severity=fact.severity,
        action_candidates=list(fact.action_candidates),
        candidate_id=fact.candidate_id,
        node_id=fact.node_id,
    )


def _fact_sort_key(fact: ErrorFact) -> tuple[str, str, str, str]:
    return (
        fact.fact_type,
        fact.class_name or "",
        fact.area or "",
        fact.subject,
    )


def diagnosis_graph_from_error_facts(
    facts: list[ErrorFact],
    path: Path | str | None = None,
    limit: int | None = None,
) -> DiagnosisGraphReport:
    """Convenience wrapper for one-shot error-fact diagnosis."""
    return DiagnosisGraph.from_yaml(path).diagnose(facts, limit=limit)

