"""Prompt-18I §8: evidence-driven capability promotion for error facts.

Two README capabilities — ``candidate_coco_error_facts`` and
``error_delta_next_round`` — may only be recorded as *executable* when a
real run proves the full loop:

    1. baseline profile facts complete
    2. candidate profile facts complete
    3. the error delta was computed from the two complete profiles
    4. a next-round decision trace consumed the delta

The promotion is decided from round artifacts (profiles, delta, trace,
gate verdicts) — never by editing README text.  The result is a
machine-readable promotion record the capability matrix can cite.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.detection_error_delta import DetectionErrorDelta
from yolo_agent.core.detection_error_profile import DetectionErrorProfile
from yolo_agent.core.error_decision_trace import ErrorDecisionTrace
from yolo_agent.core.error_profile_evidence_gate import (
    evaluate_error_profile_evidence,
)

PROMOTION_SCHEMA_VERSION = "error_capability_promotion.v1"

#: The two capabilities governed by this module.
FACTS_CAPABILITY = "candidate_coco_error_facts"
DECISION_CAPABILITY = "error_delta_next_round"

PromotionVerdict = Literal["executable", "not_promoted"]


class ErrorLoopPromotionRecord(BaseModel):
    """The machine-readable promotion decision for the error loop."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PROMOTION_SCHEMA_VERSION
    run_id: str
    facts_capability: PromotionVerdict = "not_promoted"
    decision_capability: PromotionVerdict = "not_promoted"
    baseline_facts_complete: bool = False
    candidate_facts_complete: bool = False
    error_delta_consumed: bool = False
    next_round_decision_consumed_delta: bool = False
    reasons: list[str] = Field(default_factory=list)


def _facts_complete(profile: DetectionErrorProfile | None) -> bool:
    """Judge one profile's section completeness in isolation."""
    if profile is None:
        return False
    # Evaluate the profile as both sides so section gaps are the only
    # missing entries (delta/profile-pair absence is judged separately).
    gate = evaluate_error_profile_evidence(
        baseline_profile=profile, candidate_profile=profile, delta=None
    )
    return all(
        not item.startswith(("baseline.", "candidate."))
        for item in gate.missing_sections
    )


def evaluate_error_loop_promotion(
    *,
    run_id: str,
    baseline_profile: DetectionErrorProfile | None,
    candidate_profile: DetectionErrorProfile | None,
    delta: DetectionErrorDelta | None,
    decision_trace: ErrorDecisionTrace | None,
) -> ErrorLoopPromotionRecord:
    """Judge whether a real run proved both capabilities end to end."""
    reasons: list[str] = []

    baseline_ok = _facts_complete(baseline_profile)
    candidate_ok = _facts_complete(candidate_profile)
    if not baseline_ok:
        reasons.append("baseline profile facts are not complete")
    if not candidate_ok:
        reasons.append("candidate profile facts are not complete")

    delta_ok = (
        delta is not None
        and delta.matched_evaluation
        and baseline_ok
        and candidate_ok
    )
    if not delta_ok:
        reasons.append("error delta was not computed from two complete matched profiles")

    decision_ok = (
        decision_trace is not None
        and delta_ok
        and decision_trace.evidence_delta_id == delta.candidate_profile_id
        and decision_trace.evidence_profile_id == candidate_profile.profile_id
        and bool(decision_trace.candidate_actions)
    )
    if not decision_ok:
        reasons.append("next-round decision trace did not consume the candidate-vs-parent delta")

    record = ErrorLoopPromotionRecord(
        run_id=run_id,
        baseline_facts_complete=baseline_ok,
        candidate_facts_complete=candidate_ok,
        error_delta_consumed=delta_ok,
        next_round_decision_consumed_delta=decision_ok,
        reasons=reasons,
    )
    if baseline_ok and candidate_ok and delta_ok:
        record = record.model_copy(update={"facts_capability": "executable"})
    if decision_ok:
        record = record.model_copy(update={"decision_capability": "executable"})
    return record


def manifest_upgrade_allowed(record: ErrorLoopPromotionRecord) -> bool:
    """Whether the capability manifest may promote either capability.

    The manifest itself is only edited through the normal generation path;
    this predicate decides whether the promotion record authorizes it.
    """
    return record.facts_capability == "executable" and record.decision_capability == "executable"


__all__ = [
    "DECISION_CAPABILITY",
    "ErrorLoopPromotionRecord",
    "FACTS_CAPABILITY",
    "PROMOTION_SCHEMA_VERSION",
    "evaluate_error_loop_promotion",
    "manifest_upgrade_allowed",
]
