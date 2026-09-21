"""Evidence completeness gate for error-profile rounds (Prompt-18I step 4).

After a candidate evaluation, the round's evidence is judged
``complete``, ``partial``, or ``insufficient``.  Only a *complete* profile
pair (baseline facts complete, candidate facts complete, delta computed)
may proceed to the next round's budget request; partial evidence restricts
the next round to evidence-only actions (``collect_evidence``,
``rerun_eval``, ``repair_dataset_metadata``), and insufficient evidence
permits nothing but evidence collection.

The gate never fabricates missing evidence and never blocks debug/pilot
runs — it governs the *next training round's* budget request only.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.detection_error_delta import DetectionErrorDelta
from yolo_agent.core.detection_error_profile import DetectionErrorProfile

GATE_SCHEMA_VERSION = "error_profile_evidence_gate.v1"

EvidenceVerdict = Literal["complete", "partial", "insufficient"]

#: Actions that never consume training budget and are always allowed.
EVIDENCE_ONLY_ACTIONS: tuple[str, ...] = (
    "collect_evidence",
    "rerun_eval",
    "repair_dataset_metadata",
    "repair_evaluation",
)

#: Profile sections whose content makes the facts decision-grade.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "global",
    "scale",
    "per_class",
    "false_negative",
    "false_positive",
    "localization",
    "classification",
    "confidence",
)


class ErrorProfileEvidenceGateResult(BaseModel):
    """The gate verdict for one round's profile evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = GATE_SCHEMA_VERSION
    verdict: EvidenceVerdict
    missing_sections: list[str] = Field(default_factory=list)
    missing_profiles: list[str] = Field(default_factory=list)
    delta_matched: bool = True
    reasons: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    budget_request_allowed: bool = False

    def allows(self, action: str) -> bool:
        return action in self.allowed_actions


def _profile_gaps(profile: DetectionErrorProfile | None) -> list[str]:
    """Sections missing decision-grade content in one profile."""
    if profile is None:
        return list(REQUIRED_SECTIONS)
    missing: list[str] = []
    if profile.global_.map50 is None and profile.global_.precision is None:
        missing.append("global")
    scale_aps = (profile.scale.ap_small, profile.scale.ap_medium, profile.scale.ap_large)
    scale_recalls = (profile.scale.recall_small, profile.scale.recall_medium, profile.scale.recall_large)
    if not any(value is not None for value in scale_aps) or not any(
        value is not None for value in scale_recalls
    ):
        missing.append("scale")
    if not profile.per_class:
        missing.append("per_class")
    elif not any(item.ap50 is not None for item in profile.per_class):
        # Per-class rows exist but AP@0.5 was never recorded in any of them.
        missing.append("per_class")
    if profile.false_negative.total <= 0 and profile.global_.recall is not None and profile.global_.recall < 1.0:
        # Recall below 1 with zero FNs means the FN section was never filled.
        missing.append("false_negative")
    if profile.false_positive.total <= 0 and profile.global_.precision is not None and profile.global_.precision < 1.0:
        missing.append("false_positive")
    if (
        not profile.localization.matched_iou_distribution
        and profile.localization.mean_matched_iou is None
    ):
        missing.append("localization")
    if (
        not profile.confidence.tp_confidence_histogram
        and not profile.confidence.fp_confidence_histogram
    ):
        missing.append("confidence")
    return missing


def evaluate_error_profile_evidence(
    *,
    baseline_profile: DetectionErrorProfile | None,
    candidate_profile: DetectionErrorProfile | None,
    delta: DetectionErrorDelta | None,
) -> ErrorProfileEvidenceGateResult:
    """Judge one round's evidence and constrain the next round's actions."""
    missing_profiles: list[str] = []
    if baseline_profile is None:
        missing_profiles.append("baseline_profile")
    if candidate_profile is None:
        missing_profiles.append("candidate_profile")

    missing_sections: list[str] = []
    missing_sections += [f"baseline.{name}" for name in _profile_gaps(baseline_profile)]
    missing_sections += [f"candidate.{name}" for name in _profile_gaps(candidate_profile)]

    reasons: list[str] = []
    delta_matched = True
    if delta is None:
        delta_matched = False
        reasons.append("error_delta_missing")
    else:
        delta_matched = delta.matched_evaluation
        if not delta_matched:
            reasons.append("error_delta_not_matched_evaluation")

    hard_gaps = bool(missing_profiles) or len(missing_sections) > 4
    if missing_profiles:
        reasons.append(f"missing_profiles:{','.join(missing_profiles)}")
    if missing_sections:
        reasons.append(f"incomplete_sections:{','.join(missing_sections)}")

    if hard_gaps or not delta_matched:
        verdict: EvidenceVerdict = "insufficient"
    elif missing_sections or reasons:
        verdict = "partial"
    else:
        verdict = "complete"

    if verdict == "complete":
        allowed = ["next_round_experiment"]
        budget_allowed = True
    elif verdict == "partial":
        allowed = list(EVIDENCE_ONLY_ACTIONS)
        budget_allowed = False
        reasons.append("budget_request_blocked_until_evidence_complete")
    else:
        allowed = ["collect_evidence"]
        budget_allowed = False
        reasons.append("budget_request_blocked_evidence_insufficient")

    return ErrorProfileEvidenceGateResult(
        verdict=verdict,
        missing_sections=missing_sections,
        missing_profiles=missing_profiles,
        delta_matched=delta_matched,
        reasons=reasons,
        allowed_actions=allowed,
        budget_request_allowed=budget_allowed,
    )
