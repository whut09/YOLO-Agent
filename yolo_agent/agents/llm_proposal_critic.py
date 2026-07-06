"""Lightweight guard for LLM policy proposals before harness evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from yolo_agent.agents.loop_policy_evaluator import infer_changed_variables
from yolo_agent.agents.strategy_policy import CandidatePolicy


class LLMProposalCritique(BaseModel):
    """Quality verdict for one LLM proposal."""

    policy_id: str
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    changed_variables: dict[str, Any] = Field(default_factory=dict)


class LLMProposalQualityReport(BaseModel):
    """Batch quality report for LLM proposals."""

    accepted: int = 0
    rejected: int = 0
    rejection_reasons: list[str] = Field(default_factory=list)
    critiques: list[LLMProposalCritique] = Field(default_factory=list)

    @property
    def accepted_policy_ids(self) -> set[str]:
        """Return accepted proposal IDs."""
        return {critique.policy_id for critique in self.critiques if critique.accepted}


class LLMProposalCritic:
    """Reject unsafe or underspecified LLM proposals before full evaluation."""

    def critique(
        self,
        proposals: list[CandidatePolicy],
        *,
        fixed_imgsz: int | None = None,
        missing_diagnostic_evidence: list[str] | None = None,
        require_target_error_facts: bool = True,
        require_expected_improvement: bool = True,
    ) -> LLMProposalQualityReport:
        """Critique LLM proposals using deterministic harness rules."""
        critiques = [
            self.critique_one(
                proposal,
                fixed_imgsz=fixed_imgsz,
                missing_diagnostic_evidence=missing_diagnostic_evidence,
                require_target_error_facts=require_target_error_facts,
                require_expected_improvement=require_expected_improvement,
            )
            for proposal in proposals
        ]
        rejection_reasons: list[str] = []
        for critique in critiques:
            rejection_reasons.extend(critique.rejection_reasons)
        return LLMProposalQualityReport(
            accepted=sum(1 for critique in critiques if critique.accepted),
            rejected=sum(1 for critique in critiques if not critique.accepted),
            rejection_reasons=list(dict.fromkeys(rejection_reasons)),
            critiques=critiques,
        )

    def critique_one(
        self,
        proposal: CandidatePolicy,
        *,
        fixed_imgsz: int | None = None,
        missing_diagnostic_evidence: list[str] | None = None,
        require_target_error_facts: bool = True,
        require_expected_improvement: bool = True,
    ) -> LLMProposalCritique:
        """Critique one LLM proposal."""
        reasons: list[str] = []
        warnings: list[str] = []
        changed_variables = infer_changed_variables(proposal)

        if require_target_error_facts and proposal.action_domain != "evidence" and not proposal.target_error_facts:
            reasons.append("missing_target_error_facts")
        if require_expected_improvement and proposal.action_domain != "evidence" and not proposal.expected_improvement:
            reasons.append("missing_expected_improvement")
        if len(changed_variables) > 1:
            reasons.append("multi_variable_proposal")
        reasons.extend(_fixed_imgsz_reasons(proposal, fixed_imgsz))
        reasons.extend(_yolo26_incompatibility_reasons(proposal))
        reasons.extend(_diagnostic_evidence_first_reasons(proposal, missing_diagnostic_evidence or []))
        if proposal.execution_action == "run_training" and proposal.evidence_required:
            reasons.append("pushes_training_before_required_evidence")

        return LLMProposalCritique(
            policy_id=proposal.policy_id,
            accepted=not reasons,
            rejection_reasons=list(dict.fromkeys(reasons)),
            warnings=warnings,
            changed_variables=changed_variables,
        )


def _fixed_imgsz_reasons(proposal: CandidatePolicy, fixed_imgsz: int | None) -> list[str]:
    if fixed_imgsz is None or "imgsz" not in proposal.train_overrides:
        return []
    try:
        requested = int(proposal.train_overrides["imgsz"])
    except (TypeError, ValueError):
        return ["violates_fixed_imgsz"]
    if requested > fixed_imgsz:
        return ["violates_fixed_imgsz"]
    return []


def _yolo26_incompatibility_reasons(proposal: CandidatePolicy) -> list[str]:
    if "yolo26" not in proposal.base_model.lower():
        return []
    tokens = [
        *proposal.components,
        str(proposal.action_id or ""),
        *[str(value) for value in proposal.train_overrides.values()],
    ]
    lowered = " ".join(tokens).lower()
    if "nms" in lowered:
        return ["yolo26_nms_incompatible_action"]
    if "dfl" in lowered:
        return ["yolo26_dfl_incompatible_action"]
    if "head.p2" in lowered or "p2_head" in lowered or "p2_small_object" in lowered:
        return ["yolo26_p2_head_requires_verified_recipe"]
    if any(component.startswith("loss.bbox.") for component in proposal.components):
        return ["yolo26_loss_patch_requires_verified_recipe"]
    return []


def _diagnostic_evidence_first_reasons(
    proposal: CandidatePolicy,
    missing_diagnostic_evidence: list[str],
) -> list[str]:
    if not missing_diagnostic_evidence:
        return []
    reasons: list[str] = []
    if proposal.action_domain != "evidence":
        reasons.append("diagnostic_evidence_missing_requires_evidence_action")
    if proposal.execution_action == "run_training":
        reasons.append("diagnostic_evidence_missing_blocks_run_training")
    return reasons


__all__ = [
    "LLMProposalCritic",
    "LLMProposalCritique",
    "LLMProposalQualityReport",
]
