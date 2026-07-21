"""Build non-executable recipe priors from frozen paper evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yolo_agent.components.compatibility import CompatibilityResult
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import maturity_rank
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.research.component_aliases import normalize_component_id
from yolo_agent.research.harness_hint_parser import PaperDiagnosticHint
from yolo_agent.research.note_parser import PaperMethodClaim
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.research.snapshot import ResearchSnapshot


PriorImplementationStatus = Literal[
    "metadata_only",
    "adapter_required",
    "adapter_implemented",
    "smoke_passed",
    "pilot_reproduced",
    "full_reproduced",
]
PriorCompatibility = Literal["compatible", "adapter_required", "incompatible", "unknown"]


class RecipePriorBuildError(ValueError):
    """Raised when paper context cannot form an evidence-bound recipe prior."""


class RecipePriorEvidence(BaseModel):
    """One paper-only evidence reference retained without local promotion authority."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    claim: str
    source_location: str
    evidence_level: Literal["paper_claim", "paper_prior"]


class RecipePrior(BaseModel):
    """A paper-derived recipe idea that cannot itself enter an execution queue."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_recipe_prior.v1"
    prior_id: str
    research_snapshot_hash: str
    paper_ids: list[str]
    component_ids: list[str]
    target_error_facts: list[dict[str, Any]]
    target_metrics: list[str]
    suggested_changed_variables: list[str]
    baseline_protocol: dict[str, Any]
    evidence_prior: list[RecipePriorEvidence]
    expected_paper_effect: dict[str, float | str]
    implementation_status: PriorImplementationStatus
    yolo26_compatibility: PriorCompatibility
    required_adapter: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source_locations: list[str]
    coupling_reason: str | None = None
    internal_ablation_plan: list[dict[str, Any]] = Field(default_factory=list)
    executable: Literal[False] = False

    @field_validator("prior_id", "research_snapshot_hash")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prior_id and research_snapshot_hash must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def _guard_prior(self) -> "RecipePrior":
        if not self.paper_ids or not self.component_ids:
            raise ValueError("RecipePrior requires paper_ids and component_ids")
        if not self.target_error_facts:
            raise ValueError("RecipePrior must bind current target error facts")
        if not self.suggested_changed_variables:
            raise ValueError("RecipePrior must declare a changed variable")
        if any(normalize_component_id(item) in {"imgsz", "image_size"} for item in self.suggested_changed_variables):
            raise ValueError("imgsz=640 is fixed and cannot be a suggested changed variable")
        if self.baseline_protocol.get("imgsz") != 640:
            raise ValueError("RecipePrior baseline protocol must fix imgsz=640")
        is_coupled = len(self.component_ids) > 1 or len(self.suggested_changed_variables) > 1
        if is_coupled and (not self.coupling_reason or not self.internal_ablation_plan):
            raise ValueError("Coupled paper priors require coupling_reason and internal_ablation_plan")
        if not is_coupled and (self.coupling_reason or self.internal_ablation_plan):
            raise ValueError("Atomic paper priors cannot declare coupled-recipe fields")
        return self


class PaperRecipePriorBuilder:
    """Convert explicit paper claims into replayable, non-executable priors."""

    def build(
        self,
        *,
        paper: PaperRecord,
        method_claim: PaperMethodClaim,
        diagnostic_hints: Iterable[PaperDiagnosticHint],
        component_contracts: Mapping[str, ComponentContract] | Iterable[ComponentContract],
        compatibility: CompatibilityResult,
        research_snapshot: ResearchSnapshot,
        current_error_facts: Iterable[ErrorFact],
        coupling_reason: str | None = None,
        internal_ablation_plan: list[dict[str, Any]] | None = None,
    ) -> RecipePrior:
        contracts = _contract_mapping(component_contracts)
        hints = [hint for hint in diagnostic_hints if hint.paper_id in {"unknown", paper.paper_id}]
        facts = [fact for fact in current_error_facts if fact.evidence_role == "current_observation"]
        _validate_snapshot(research_snapshot)

        component_ids = _selected_components(method_claim, hints, contracts)
        changed_variables = _changed_variables(method_claim)
        target_facts = _target_facts(facts, hints, method_claim)
        if not target_facts:
            raise RecipePriorBuildError(
                "No current error fact matches the paper diagnostic target; collect local evidence before recipe planning"
            )
        if not changed_variables:
            raise RecipePriorBuildError("Paper method does not explicitly declare a changed variable")

        is_coupled = len(component_ids) > 1 or len(changed_variables) > 1
        if len(component_ids) > 1 and len(changed_variables) < 2:
            raise RecipePriorBuildError(
                "A multi-component paper prior must explicitly identify each coupled changed variable"
            )
        if is_coupled and (not coupling_reason or not internal_ablation_plan):
            raise RecipePriorBuildError(
                "Paper method contains multiple components or variables; explicit coupling reason and ablation plan are required"
            )

        status = _implementation_status([contracts[item] for item in component_ids])
        yolo26_compatibility = _compatibility_status(compatibility, status)
        required_adapter = _required_adapters([contracts[item] for item in component_ids], compatibility)
        evidence_prior = _evidence_prior(paper, method_claim, hints)
        source_locations = sorted({item.source_location for item in evidence_prior})
        baseline_protocol = _baseline_protocol(target_facts, research_snapshot)
        confidence = _confidence(hints, compatibility, target_facts)
        payload = {
            "snapshot": research_snapshot.snapshot_hash,
            "paper_ids": [paper.paper_id],
            "component_ids": component_ids,
            "target_error_facts": target_facts,
            "changed_variables": changed_variables,
            "source_locations": source_locations,
        }
        prior_id = "paper-prior-" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:16]

        return RecipePrior(
            prior_id=prior_id,
            research_snapshot_hash=research_snapshot.snapshot_hash,
            paper_ids=[paper.paper_id],
            component_ids=component_ids,
            target_error_facts=target_facts,
            target_metrics=_target_metrics(method_claim, hints),
            suggested_changed_variables=changed_variables,
            baseline_protocol=baseline_protocol,
            evidence_prior=evidence_prior,
            expected_paper_effect=dict(method_claim.reported_delta),
            implementation_status=status,
            yolo26_compatibility=yolo26_compatibility,
            required_adapter=required_adapter,
            confidence=confidence,
            source_locations=source_locations,
            coupling_reason=coupling_reason,
            internal_ablation_plan=list(internal_ablation_plan or []),
        )


def _validate_snapshot(snapshot: ResearchSnapshot) -> None:
    if not snapshot.frozen:
        raise RecipePriorBuildError("ResearchSnapshot must be frozen")
    if snapshot.paper_intelligence != "available":
        raise RecipePriorBuildError("Paper Intelligence is unavailable in the bound ResearchSnapshot")


def _contract_mapping(
    contracts: Mapping[str, ComponentContract] | Iterable[ComponentContract],
) -> dict[str, ComponentContract]:
    if isinstance(contracts, Mapping):
        return dict(contracts)
    return {item.component_id: item for item in contracts}


def _selected_components(
    claim: PaperMethodClaim,
    hints: list[PaperDiagnosticHint],
    contracts: dict[str, ComponentContract],
) -> list[str]:
    explicit = list(claim.component_ids)
    if not explicit:
        explicit = [component for hint in hints for component in hint.candidate_component_ids]
    selected = list(dict.fromkeys(explicit))
    if not selected:
        raise RecipePriorBuildError("Paper method does not identify a component")
    missing = sorted(set(selected) - set(contracts))
    if missing:
        raise RecipePriorBuildError("Paper components have no ComponentContract: " + ", ".join(missing))
    return selected


def _changed_variables(claim: PaperMethodClaim) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in claim.changed_variables if item.strip()))


def _target_facts(
    facts: list[ErrorFact],
    hints: list[PaperDiagnosticHint],
    claim: PaperMethodClaim,
) -> list[dict[str, Any]]:
    terms = {
        normalize_component_id(item)
        for hint in hints
        for item in [*hint.target_error_facts, *hint.target_metrics]
        if item and item != "unknown"
    }
    terms.update(normalize_component_id(item) for item in claim.reported_delta)
    matched: list[ErrorFact] = []
    for fact in facts:
        text = normalize_component_id(" ".join([
            fact.fact_type,
            fact.subject,
            fact.class_name or "",
            fact.area or "",
            fact.metric_name or "",
            *fact.action_candidates,
        ]))
        if not terms or any(term in text or text in term for term in terms):
            matched.append(fact)
    return [
        {
            "run_id": fact.run_id,
            "candidate_id": fact.candidate_id,
            "node_id": fact.node_id,
            "fact_type": fact.fact_type,
            "subject": fact.subject,
            "class_name": fact.class_name,
            "area": fact.area,
            "metric_name": fact.metric_name,
            "severity": fact.severity,
            "protocol_hash": fact.protocol_hash,
        }
        for fact in matched
    ]


def _target_metrics(claim: PaperMethodClaim, hints: list[PaperDiagnosticHint]) -> list[str]:
    values = [*claim.reported_delta, *(metric for hint in hints for metric in hint.target_metrics)]
    return list(dict.fromkeys(item for item in values if item and item != "unknown"))


def _baseline_protocol(facts: list[dict[str, Any]], snapshot: ResearchSnapshot) -> dict[str, Any]:
    protocol_hashes = sorted({str(item["protocol_hash"]) for item in facts if item.get("protocol_hash")})
    return {
        "imgsz": 640,
        "protocol_hashes": protocol_hashes,
        "research_snapshot_hash": snapshot.snapshot_hash,
        "evidence_role": "current_observation",
    }


def _implementation_status(contracts: list[ComponentContract]) -> PriorImplementationStatus:
    def normalized(contract: ComponentContract) -> PriorImplementationStatus:
        if contract.maturity == "metadata_only":
            return "metadata_only"
        if contract.maturity == "reference_code_available":
            return "adapter_required"
        if contract.maturity in {"adapter_implemented", "unit_tested"}:
            return "adapter_implemented"
        if contract.maturity == "smoke_passed":
            return "smoke_passed"
        if contract.maturity == "pilot_reproduced":
            return "pilot_reproduced"
        return "full_reproduced"

    order = {
        "metadata_only": 0,
        "adapter_required": 1,
        "adapter_implemented": 2,
        "smoke_passed": 3,
        "pilot_reproduced": 4,
        "full_reproduced": 5,
    }
    return min((normalized(item) for item in contracts), key=order.__getitem__)


def _compatibility_status(
    result: CompatibilityResult,
    implementation_status: PriorImplementationStatus,
) -> PriorCompatibility:
    yolo26 = result.yolo26 or {}
    if not result.ok or bool(yolo26.get("incompatible")) or bool(yolo26.get("blocked_by")):
        return "incompatible"
    if bool(yolo26.get("research_adapter_required")) or implementation_status in {
        "metadata_only",
        "adapter_required",
    }:
        return "adapter_required"
    if result.ok:
        return "compatible"
    return "unknown"


def _required_adapters(
    contracts: list[ComponentContract],
    compatibility: CompatibilityResult,
) -> list[str]:
    yolo26 = compatibility.yolo26 or {}
    required = [str(item) for item in yolo26.get("required_adapters", [])]
    for contract in contracts:
        if maturity_rank(contract.maturity) < maturity_rank("adapter_implemented") or not contract.adapter_class:
            required.append(contract.adapter_class or contract.component_id)
    return sorted(set(required))


def _evidence_prior(
    paper: PaperRecord,
    claim: PaperMethodClaim,
    hints: list[PaperDiagnosticHint],
) -> list[RecipePriorEvidence]:
    evidence = [
        RecipePriorEvidence(
            paper_id=paper.paper_id,
            claim=claim.method_name,
            source_location=claim.source_location,
            evidence_level="paper_claim",
        )
    ]
    evidence.extend(
        RecipePriorEvidence(
            paper_id=paper.paper_id,
            claim=hint.symptom,
            source_location=hint.source_location,
            evidence_level="paper_claim",
        )
        for hint in hints
    )
    return evidence


def _confidence(
    hints: list[PaperDiagnosticHint],
    compatibility: CompatibilityResult,
    target_facts: list[dict[str, Any]],
) -> float:
    hint_confidence = max((item.confidence for item in hints), default=0.4)
    score = 0.45 + hint_confidence * 0.25 + min(len(target_facts), 3) * 0.05
    if compatibility.ok:
        score += 0.1
    return round(min(score, 0.9), 6)


__all__ = [
    "PaperRecipePriorBuilder",
    "PriorCompatibility",
    "PriorImplementationStatus",
    "RecipePrior",
    "RecipePriorBuildError",
    "RecipePriorEvidence",
]
