"""Link trusted local error facts to frozen paper diagnostic priors."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.error_facts import ErrorFact, ErrorFactType
from yolo_agent.research.component_aliases import ComponentAliasResolver, normalize_component_id
from yolo_agent.research.harness_hint_parser import PaperDiagnosticHint
from yolo_agent.research.schemas import ComponentCategory, ComponentTaxonomy, PaperComponentClaim
from yolo_agent.research.snapshot import ResearchSnapshot
from yolo_agent.resources import ResourcePaths
from yolo_agent.tools.dataset_stats import DatasetReport


FamilyCompatibility = Literal["compatible", "adapter_required", "incompatible"]


class PaperDiagnosisFactPattern(BaseModel):
    """All populated predicates must match one current error fact."""

    model_config = ConfigDict(extra="forbid")

    fact_types: list[ErrorFactType] = Field(default_factory=list)
    metric_names: list[str] = Field(default_factory=list)
    areas: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    action_candidates: list[str] = Field(default_factory=list)


class PaperDiagnosisCauseRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause_id: str
    description: str
    evidence_needed: list[str] = Field(default_factory=list)


class PaperDiagnosisFamilyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_id: str
    component_ids: list[str] = Field(default_factory=list)
    categories: list[ComponentCategory] = Field(default_factory=list)
    detector_families: list[str] = Field(default_factory=lambda: ["generic"])
    compatibility: FamilyCompatibility = "adapter_required"
    requires_nms: bool = False
    requires_dfl: bool = False
    reason: str


class PaperDiagnosisRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    symptom: str
    patterns: list[PaperDiagnosisFactPattern]
    hint_metrics: list[str] = Field(default_factory=list)
    hint_error_facts: list[str] = Field(default_factory=list)
    likely_causes: list[PaperDiagnosisCauseRule] = Field(default_factory=list)
    evidence_requests: list[str] = Field(default_factory=list)
    families: list[PaperDiagnosisFamilyRule] = Field(default_factory=list)


class PaperDiagnosisRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_diagnosis_rules.v1"
    rules: list[PaperDiagnosisRule]

    @classmethod
    def from_yaml(cls, path: Path | str = ResourcePaths.PAPER_DIAGNOSIS_RULES) -> "PaperDiagnosisRules":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        return cls.model_validate(payload)


class DiagnosisLinkedPaper(BaseModel):
    paper_id: str
    matched_rule_ids: list[str] = Field(default_factory=list)
    matched_component_ids: list[str] = Field(default_factory=list)
    source_locations: list[str] = Field(default_factory=list)
    evidence_level: Literal["paper_claim"] = "paper_claim"


class LinkedLikelyCause(BaseModel):
    cause_id: str
    description: str
    supporting_fact_ids: list[str] = Field(default_factory=list)
    linked_paper_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_level: Literal["paper_claim"] = "paper_claim"
    local_evidence_role: Literal["diagnostic_trigger_only"] = "diagnostic_trigger_only"


class PaperEvidenceRequest(BaseModel):
    evidence: str
    reason: str
    required_for: list[str] = Field(default_factory=list)
    priority: Literal["low", "medium", "high"] = "medium"


class CandidateComponentFamily(BaseModel):
    family_id: str
    component_ids: list[str] = Field(default_factory=list)
    categories: list[ComponentCategory] = Field(default_factory=list)
    compatibility: FamilyCompatibility
    implementation_status: str
    linked_paper_ids: list[str] = Field(default_factory=list)
    reason: str
    evidence_level: Literal["paper_claim"] = "paper_claim"
    executable: bool = False


class RejectedPaperFamily(BaseModel):
    family_id: str
    component_ids: list[str] = Field(default_factory=list)
    linked_paper_ids: list[str] = Field(default_factory=list)
    reason: str
    blocked_by: list[str] = Field(default_factory=list)
    evidence_level: Literal["paper_claim"] = "paper_claim"


class PaperPriorSummary(BaseModel):
    snapshot_hash: str | None = None
    paper_intelligence: str = "unavailable"
    linked_paper_count: int = 0
    diagnostic_hint_count: int = 0
    component_claim_count: int = 0
    evidence_level: Literal["paper_claim"] = "paper_claim"
    used_as_local_evidence: bool = False
    note: str = "Paper priors guide diagnosis only; promotion requires local verified evidence."


class PaperDiagnosisLinkResult(BaseModel):
    diagnosis_linked_papers: list[DiagnosisLinkedPaper] = Field(default_factory=list)
    likely_causes: list[LinkedLikelyCause] = Field(default_factory=list)
    evidence_requests: list[PaperEvidenceRequest] = Field(default_factory=list)
    candidate_component_families: list[CandidateComponentFamily] = Field(default_factory=list)
    rejected_paper_families: list[RejectedPaperFamily] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    paper_prior_summary: PaperPriorSummary = Field(default_factory=PaperPriorSummary)


class PaperDiagnosisLinker:
    """Deterministically connect local symptoms to non-executable paper priors."""

    def __init__(
        self,
        rules: PaperDiagnosisRules | None = None,
        *,
        alias_resolver: ComponentAliasResolver | None = None,
    ) -> None:
        self.rules = rules or PaperDiagnosisRules.from_yaml()
        self.alias_resolver = alias_resolver or ComponentAliasResolver.from_yaml()

    def link(
        self,
        *,
        error_facts: Iterable[ErrorFact],
        dataset_report: DatasetReport | None,
        coco_post_eval: BaseModel | dict[str, Any] | None,
        per_class_ap_ar: dict[str, Any] | None,
        confusion_summary: dict[str, Any] | str | None,
        diagnostic_hints: Iterable[PaperDiagnosticHint],
        component_claims: Iterable[PaperComponentClaim],
        taxonomy: ComponentTaxonomy,
        research_snapshot: ResearchSnapshot | None,
    ) -> PaperDiagnosisLinkResult:
        del taxonomy  # Config categories are validated against ComponentCategory.
        facts = list(error_facts)
        hints = list(diagnostic_hints)
        claims = list(component_claims)
        prior_summary = _paper_prior_summary(research_snapshot, hints, claims)

        if not facts:
            return PaperDiagnosisLinkResult(
                evidence_requests=_dedupe_requests([
                    PaperEvidenceRequest(
                        evidence="current_error_facts",
                        reason="No current error facts are available; paper priors cannot trigger a diagnosis.",
                        required_for=["paper_diagnosis_linking"],
                        priority="high",
                    ),
                    *_base_evidence_requests(per_class_ap_ar, coco_post_eval, confusion_summary),
                ]),
                reasons=["Evidence-first guard: no candidate family or likely cause was produced without error facts."],
                paper_prior_summary=prior_summary,
            )

        if not _snapshot_available(research_snapshot):
            return PaperDiagnosisLinkResult(
                evidence_requests=[PaperEvidenceRequest(
                    evidence="frozen_research_snapshot",
                    reason="Paper Intelligence is unavailable or the snapshot is not frozen.",
                    required_for=["paper_prior_linking"],
                    priority="high",
                )],
                reasons=["Local error facts exist, but no frozen paper prior may be used."],
                paper_prior_summary=prior_summary,
            )

        inventory = _EvidenceInventory(
            facts=facts,
            dataset_report=dataset_report,
            coco_post_eval=_mapping(coco_post_eval),
            per_class_ap_ar=per_class_ap_ar or {},
            confusion_summary=_confusion_mapping(confusion_summary),
        )
        linked_papers: dict[str, DiagnosisLinkedPaper] = {}
        causes: list[LinkedLikelyCause] = []
        requests: list[PaperEvidenceRequest] = []
        candidates: list[CandidateComponentFamily] = []
        rejected: list[RejectedPaperFamily] = []
        reasons: list[str] = []
        confidences: list[float] = []

        for rule in self.rules.rules:
            matched_facts = [fact for fact in facts if any(_matches_pattern(fact, pattern) for pattern in rule.patterns)]
            if not matched_facts:
                continue
            relevant_hints = [
                hint for hint in hints if _hint_supports_rule(hint, rule, self.alias_resolver)
            ]
            relevant_claims = [
                claim for claim in claims if _claim_supports_rule(claim, rule, self.alias_resolver)
            ]
            rule_paper_ids = sorted({
                item.paper_id for item in [*relevant_hints, *relevant_claims]
                if item.paper_id != "unknown"
            })
            confidence = _rule_confidence(matched_facts, relevant_hints, relevant_claims, inventory, rule)
            confidences.append(confidence)
            fact_ids = [_fact_id(fact) for fact in matched_facts]
            for cause in rule.likely_causes:
                causes.append(LinkedLikelyCause(
                    cause_id=cause.cause_id,
                    description=cause.description,
                    supporting_fact_ids=fact_ids,
                    linked_paper_ids=rule_paper_ids,
                    confidence=confidence,
                ))
                requests.extend(_missing_requests(cause.evidence_needed, inventory, [rule.rule_id]))
            requests.extend(_missing_requests(rule.evidence_requests, inventory, [rule.rule_id]))
            for hint in relevant_hints:
                requests.extend(_missing_requests(hint.evidence_needed, inventory, [rule.rule_id]))

            for family in rule.families:
                family_hints = [
                    hint
                    for hint in relevant_hints
                    if _hint_supports_family(hint, family, self.alias_resolver)
                ]
                family_claims = [
                    claim
                    for claim in relevant_claims
                    if _claim_supports_family(claim, family, self.alias_resolver)
                ]
                if not family_hints and not family_claims:
                    continue
                family_papers = sorted({
                    item.paper_id for item in [*family_hints, *family_claims]
                    if item.paper_id != "unknown"
                })
                blocked = _family_blocks(family, self.alias_resolver)
                if blocked:
                    rejected.append(RejectedPaperFamily(
                        family_id=family.family_id,
                        component_ids=family.component_ids,
                        linked_paper_ids=family_papers,
                        reason=family.reason,
                        blocked_by=blocked,
                    ))
                else:
                    status, executable = _family_implementation(family, self.alias_resolver)
                    candidates.append(CandidateComponentFamily(
                        family_id=family.family_id,
                        component_ids=family.component_ids,
                        categories=family.categories,
                        compatibility=family.compatibility,
                        implementation_status=status,
                        linked_paper_ids=family_papers,
                        reason=family.reason,
                        executable=executable,
                    ))
                for paper_id in family_papers:
                    linked = linked_papers.setdefault(paper_id, DiagnosisLinkedPaper(paper_id=paper_id))
                    linked.matched_rule_ids = sorted(set([*linked.matched_rule_ids, rule.rule_id]))
                    linked.matched_component_ids = sorted(set([*linked.matched_component_ids, *family.component_ids]))
                    locations = [hint.source_location for hint in family_hints if hint.paper_id == paper_id]
                    linked.source_locations = sorted(set([*linked.source_locations, *locations]))
            reasons.append(
                f"Matched {rule.rule_id} from {len(matched_facts)} current error fact(s); "
                f"paper hints={len(relevant_hints)} claims={len(relevant_claims)}."
            )

        if not causes:
            requests.append(PaperEvidenceRequest(
                evidence="diagnosable_error_facts",
                reason="Current facts did not match a configured paper diagnosis symptom.",
                required_for=["paper_diagnosis_linking"],
                priority="medium",
            ))
            reasons.append("No paper diagnosis rule matched the current error facts.")
        prior_summary.linked_paper_count = len(linked_papers)
        return PaperDiagnosisLinkResult(
            diagnosis_linked_papers=sorted(linked_papers.values(), key=lambda item: item.paper_id),
            likely_causes=_dedupe_causes(causes),
            evidence_requests=_dedupe_requests(requests),
            candidate_component_families=_dedupe_candidates(candidates),
            rejected_paper_families=_dedupe_rejected(rejected),
            reasons=reasons,
            confidence=max(confidences, default=0.0),
            paper_prior_summary=prior_summary,
        )


class _EvidenceInventory:
    def __init__(
        self,
        *,
        facts: list[ErrorFact],
        dataset_report: DatasetReport | None,
        coco_post_eval: dict[str, Any],
        per_class_ap_ar: dict[str, Any],
        confusion_summary: dict[str, Any],
    ) -> None:
        self.facts = facts
        self.dataset_report = dataset_report
        self.coco_post_eval = coco_post_eval
        self.per_class_ap_ar = per_class_ap_ar
        self.confusion_summary = confusion_summary

    def has(self, evidence: str) -> bool:
        key = normalize_component_id(evidence)
        fact_text = " ".join(_fact_text(fact) for fact in self.facts)
        if key in {"current_error_facts", "diagnosable_error_facts"}:
            return bool(self.facts)
        if key in {"per_class_ap_ar", "ap_small_by_class"}:
            return bool(self.per_class_ap_ar)
        if key in {"confusion_summary", "duplicate_pairs_by_class"}:
            return bool(self.confusion_summary or self.coco_post_eval.get("class_confusion_pairs"))
        if key in {"class_distribution"}:
            return bool(self.dataset_report and self.dataset_report.class_distribution)
        if key in {"bbox_area_histogram", "dataset_small_object_ratio"}:
            return bool(self.dataset_report and self.dataset_report.object_size_ratio)
        if key in {"ap_small", "ar_small"}:
            return key in normalize_component_id(json.dumps(self.coco_post_eval, sort_keys=True)) or key in fact_text
        if key in {"localization_error_breakdown", "iou_error_distribution"}:
            return "localization" in fact_text or bool(self.coco_post_eval.get("localization_error_top_classes"))
        if key in {"false_negative_gallery", "small_object_false_negative_classes"}:
            return "false_negative" in fact_text or bool(self.coco_post_eval.get("false_negative_top_classes"))
        return key in fact_text or key in normalize_component_id(json.dumps(self.coco_post_eval, sort_keys=True))


def _matches_pattern(fact: ErrorFact, pattern: PaperDiagnosisFactPattern) -> bool:
    if pattern.fact_types and fact.fact_type not in pattern.fact_types:
        return False
    if pattern.metric_names and normalize_component_id(fact.metric_name or "") not in {
        normalize_component_id(item) for item in pattern.metric_names
    }:
        return False
    if pattern.areas and normalize_component_id(fact.area or "") not in {
        normalize_component_id(item) for item in pattern.areas
    }:
        return False
    text = _fact_text(fact)
    if pattern.keywords and not any(normalize_component_id(item) in text for item in pattern.keywords):
        return False
    if pattern.action_candidates and not set(pattern.action_candidates).intersection(fact.action_candidates):
        return False
    return True


def _fact_text(fact: ErrorFact) -> str:
    payload = [
        fact.fact_type,
        fact.subject,
        fact.class_name or "",
        fact.class_pair or "",
        fact.area or "",
        fact.metric_name or "",
        *fact.action_candidates,
        *fact.evidence.keys(),
    ]
    return normalize_component_id(" ".join(payload))


def _hint_supports_rule(
    hint: PaperDiagnosticHint,
    rule: PaperDiagnosisRule,
    resolver: ComponentAliasResolver,
) -> bool:
    return bool(
        set(hint.target_metrics).intersection(rule.hint_metrics)
        or set(hint.target_error_facts).intersection(rule.hint_error_facts)
        or any(_component_in_rule(item, rule, resolver) for item in hint.candidate_component_ids)
    )


def _claim_supports_rule(
    claim: PaperComponentClaim,
    rule: PaperDiagnosisRule,
    resolver: ComponentAliasResolver,
) -> bool:
    return bool(
        set(claim.target_metrics).intersection(rule.hint_metrics)
        or set(claim.target_error_types).intersection(rule.hint_error_facts)
        or any(_claim_supports_family(claim, family, resolver) for family in rule.families)
    )


def _hint_supports_family(
    hint: PaperDiagnosticHint,
    family: PaperDiagnosisFamilyRule,
    resolver: ComponentAliasResolver,
) -> bool:
    if not hint.candidate_component_ids:
        return True
    return any(_component_in_family(item, family, resolver) for item in hint.candidate_component_ids)


def _claim_supports_family(
    claim: PaperComponentClaim,
    family: PaperDiagnosisFamilyRule,
    resolver: ComponentAliasResolver,
) -> bool:
    if claim.component_category and claim.component_category in family.categories:
        return True
    return _component_in_family(claim.component_id, family, resolver)


def _component_in_rule(
    component_id: str,
    rule: PaperDiagnosisRule,
    resolver: ComponentAliasResolver,
) -> bool:
    return any(_component_in_family(component_id, family, resolver) for family in rule.families)


def _component_in_family(
    component_id: str,
    family: PaperDiagnosisFamilyRule,
    resolver: ComponentAliasResolver,
) -> bool:
    normalized = normalize_component_id(component_id)
    if normalized in {normalize_component_id(item) for item in family.component_ids}:
        return True
    resolution = resolver.resolve(component_id)
    return any(mapping.canonical_component_id in family.component_ids for mapping in resolution.mappings)


def _family_blocks(
    family: PaperDiagnosisFamilyRule,
    resolver: ComponentAliasResolver,
) -> list[str]:
    blocked: list[str] = []
    if family.requires_nms:
        blocked.append("yolo26_one_to_one_nms_free_blocks_nms_recipe")
    if family.requires_dfl:
        blocked.append("yolo26_dfl_free_blocks_dfl_dependent_component")
    if family.compatibility == "incompatible":
        blocked.append("paper_family_marked_yolo26_incompatible")
    detector_families = set(family.detector_families)
    if "generic" not in detector_families and "yolo26" not in detector_families:
        blocked.append("detector_family_incompatible_with_yolo26")
    for component_id in family.component_ids:
        resolution = resolver.resolve(component_id)
        if any(mapping.yolo26_compatibility == "incompatible" for mapping in resolution.mappings):
            blocked.append(f"component_incompatible_with_yolo26:{component_id}")
    return sorted(set(blocked))


def _family_implementation(
    family: PaperDiagnosisFamilyRule,
    resolver: ComponentAliasResolver,
) -> tuple[str, bool]:
    statuses: list[str] = []
    executable = False
    for component_id in family.component_ids:
        resolution = resolver.resolve(component_id)
        statuses.extend(mapping.implementation_status for mapping in resolution.mappings)
        executable = executable or any(mapping.executable for mapping in resolution.mappings)
    if executable:
        return "smoke_passed", True
    if statuses:
        rank = {
            "metadata_only": 0,
            "recipe_idea_only": 1,
            "adapter_required": 2,
            "adapter_implemented": 3,
            "smoke_passed": 4,
            "pilot_reproduced": 5,
            "full_reproduced": 6,
        }
        return max(statuses, key=lambda item: rank.get(item, -1)), False
    return "metadata_only", False


def _rule_confidence(
    facts: list[ErrorFact],
    hints: list[PaperDiagnosticHint],
    claims: list[PaperComponentClaim],
    inventory: _EvidenceInventory,
    rule: PaperDiagnosisRule,
) -> float:
    severity = {"low": 0.48, "medium": 0.62, "high": 0.76}
    score = max(severity[fact.severity] for fact in facts)
    if hints:
        score += min(0.08, max(hint.confidence for hint in hints) * 0.1)
    if claims:
        score += 0.05
    if any(inventory.has(item) for item in rule.evidence_requests):
        score += 0.05
    return min(score, 0.95)


def _missing_requests(
    evidence_names: Iterable[str],
    inventory: _EvidenceInventory,
    required_for: list[str],
) -> list[PaperEvidenceRequest]:
    return [
        PaperEvidenceRequest(
            evidence=name,
            reason="Required diagnostic evidence is not present in current local artifacts.",
            required_for=required_for,
            priority="high" if name in {"per_class_ap_ar", "localization_error_breakdown", "confusion_summary"} else "medium",
        )
        for name in evidence_names
        if name != "unknown" and not inventory.has(name)
    ]


def _base_evidence_requests(
    per_class_ap_ar: dict[str, Any] | None,
    coco_post_eval: BaseModel | dict[str, Any] | None,
    confusion_summary: dict[str, Any] | str | None,
) -> list[PaperEvidenceRequest]:
    requests: list[PaperEvidenceRequest] = []
    if not per_class_ap_ar:
        requests.append(PaperEvidenceRequest(
            evidence="per_class_ap_ar",
            reason="Per-class AP/AR is missing.",
            required_for=["class_and_size_diagnosis"],
            priority="high",
        ))
    if not _mapping(coco_post_eval):
        requests.append(PaperEvidenceRequest(
            evidence="coco_post_eval",
            reason="Candidate COCO post-eval is missing.",
            required_for=["error_delta_diagnosis"],
            priority="high",
        ))
    if not _confusion_mapping(confusion_summary):
        requests.append(PaperEvidenceRequest(
            evidence="confusion_summary",
            reason="Confusion evidence is missing.",
            required_for=["duplicate_and_long_tail_diagnosis"],
            priority="medium",
        ))
    return requests


def _paper_prior_summary(
    snapshot: ResearchSnapshot | None,
    hints: list[PaperDiagnosticHint],
    claims: list[PaperComponentClaim],
) -> PaperPriorSummary:
    return PaperPriorSummary(
        snapshot_hash=snapshot.snapshot_hash if snapshot else None,
        paper_intelligence=snapshot.paper_intelligence if snapshot else "unavailable",
        diagnostic_hint_count=len(hints),
        component_claim_count=len(claims),
    )


def _snapshot_available(snapshot: ResearchSnapshot | None) -> bool:
    return bool(snapshot and snapshot.frozen and snapshot.paper_intelligence == "available")


def _mapping(value: BaseModel | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value or {})


def _confusion_mapping(value: dict[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {"summary": value}


def _fact_id(fact: ErrorFact) -> str:
    return f"{fact.node_id}:{fact.fact_type}:{fact.subject}"


def _dedupe_requests(items: list[PaperEvidenceRequest]) -> list[PaperEvidenceRequest]:
    output: dict[str, PaperEvidenceRequest] = {}
    for item in items:
        existing = output.get(item.evidence)
        if existing is None:
            output[item.evidence] = item
        else:
            existing.required_for = sorted(set([*existing.required_for, *item.required_for]))
            if item.priority == "high":
                existing.priority = "high"
    return [output[key] for key in sorted(output)]


def _dedupe_causes(items: list[LinkedLikelyCause]) -> list[LinkedLikelyCause]:
    output: dict[str, LinkedLikelyCause] = {}
    for item in items:
        existing = output.get(item.cause_id)
        if existing is None:
            output[item.cause_id] = item
        else:
            existing.supporting_fact_ids = sorted(set([*existing.supporting_fact_ids, *item.supporting_fact_ids]))
            existing.linked_paper_ids = sorted(set([*existing.linked_paper_ids, *item.linked_paper_ids]))
            existing.confidence = max(existing.confidence, item.confidence)
    return [output[key] for key in sorted(output)]


def _dedupe_candidates(items: list[CandidateComponentFamily]) -> list[CandidateComponentFamily]:
    return _dedupe_models(items, "family_id")


def _dedupe_rejected(items: list[RejectedPaperFamily]) -> list[RejectedPaperFamily]:
    return _dedupe_models(items, "family_id")


def _dedupe_models(items: list[Any], key: str) -> list[Any]:
    output: dict[str, Any] = {}
    for item in items:
        identity = str(getattr(item, key))
        if identity not in output:
            output[identity] = item
    return [output[name] for name in sorted(output)]


__all__ = [
    "CandidateComponentFamily",
    "DiagnosisLinkedPaper",
    "LinkedLikelyCause",
    "PaperDiagnosisLinkResult",
    "PaperDiagnosisLinker",
    "PaperDiagnosisRules",
    "PaperEvidenceRequest",
    "PaperPriorSummary",
    "RejectedPaperFamily",
]
