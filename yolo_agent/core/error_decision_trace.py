"""Prompt-18I §7: round decision traces over measured error evidence.

A decision trace is the machine-readable record of one optimization round:

    problem -> evidence -> hypotheses -> candidate actions
            -> rejected actions (with reasons) -> selected experiment
            -> expected effect -> observed effect

Every element is derived from real artifacts (profiles, delta, gate
verdict) — the trace never invents evidence and never upgrades its status
without the observed effect being recorded from a real evaluation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.detection_error_delta import DetectionErrorDelta
from yolo_agent.core.detection_error_profile import DetectionErrorProfile
from yolo_agent.core.error_profile_evidence_gate import (
    ErrorProfileEvidenceGateResult,
)
from yolo_agent.core.error_root_cause import EvidenceLinkedHypothesis

TRACE_SCHEMA_VERSION = "error_decision_trace.v1"

TraceStatus = Literal["pending_observation", "observed", "blocked_by_evidence"]


class RejectedAction(BaseModel):
    """One action family that was considered and not selected, with why."""

    model_config = ConfigDict(extra="forbid")

    family: str
    reason: str

    @model_validator(mode="after")
    def validate_rejection(self) -> "RejectedAction":
        if not self.family.strip():
            raise ValueError("rejected action family must not be empty")
        if not self.reason.strip():
            raise ValueError("a rejected action requires a recorded reason")
        return self


class SelectedExperiment(BaseModel):
    """The action actually chosen for the round and its budget status."""

    model_config = ConfigDict(extra="forbid")

    family: str
    expected_effect: str
    round_budget_requested: bool = False
    gate_allows_budget: bool = False

    @model_validator(mode="after")
    def validate_selection(self) -> "SelectedExperiment":
        if not self.family.strip():
            raise ValueError("selected family must not be empty")
        if not self.expected_effect.strip():
            raise ValueError("selected experiment requires an expected effect")
        if self.round_budget_requested and not self.gate_allows_budget:
            raise ValueError(
                "budget cannot be requested while the evidence gate withholds it"
            )
        return self


class ObservedEffect(BaseModel):
    """Post-run measured movement, filled only from a real evaluation."""

    model_config = ConfigDict(extra="forbid")

    delta_source: str
    improvements: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_observation(self) -> "ObservedEffect":
        if not self.delta_source.strip():
            raise ValueError("observed effect must cite its delta source")
        return self


class ErrorDecisionTrace(BaseModel):
    """The full §7 record for one round."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRACE_SCHEMA_VERSION
    run_id: str
    round_id: str
    status: TraceStatus = "pending_observation"
    problem: str
    evidence_profile_id: str
    evidence_delta_id: str | None = None
    gate_verdict: str
    hypotheses: list[EvidenceLinkedHypothesis] = Field(default_factory=list)
    candidate_actions: list[str] = Field(default_factory=list)
    rejected_actions: list[RejectedAction] = Field(default_factory=list)
    selected: SelectedExperiment | None = None
    observed: ObservedEffect | None = None

    @model_validator(mode="after")
    def validate_trace(self) -> "ErrorDecisionTrace":
        if not self.problem.strip():
            raise ValueError("a trace requires a problem statement")
        if not self.evidence_profile_id.strip():
            raise ValueError("a trace requires its evidence profile id")
        if self.status == "observed" and self.observed is None:
            raise ValueError("an observed trace must carry the observed effect")
        if self.observed is not None and self.status == "blocked_by_evidence":
            raise ValueError("an evidence-blocked round cannot have observations")
        return self


def _problem_from(profile: DetectionErrorProfile, delta: DetectionErrorDelta | None) -> str:
    """Name the round's problem from real numbers, not a guess."""
    regressions = delta.regressions() if delta is not None else []
    if regressions:
        return "regressions after candidate change: " + ", ".join(regressions[:4])
    if profile.false_negative.total > 0 and profile.false_positive.total > 0:
        return (
            f"open error budget: {profile.false_negative.total} FNs and "
            f"{profile.false_positive.total} FPs on split '{profile.split}'"
        )
    return f"error profile round on split '{profile.split}'"


def _reject_unavailable(
    hypotheses: list[EvidenceLinkedHypothesis],
    gate: ErrorProfileEvidenceGateResult,
) -> list[RejectedAction]:
    """Families dropped because the round's evidence cannot support them.

    Training-budget-consuming families are rejected whenever the gate
    withholds the budget; evidence-only families are always retained.
    """
    from yolo_agent.core.error_profile_evidence_gate import EVIDENCE_ONLY_ACTIONS

    rejected: list[RejectedAction] = []
    if gate.budget_request_allowed:
        return rejected
    for hypothesis in hypotheses:
        for family in hypothesis.candidate_action_families:
            if family not in EVIDENCE_ONLY_ACTIONS:
                rejected.append(
                    RejectedAction(
                        family=family,
                        reason=(
                            f"evidence gate verdict '{gate.verdict}' withholds "
                            "the next-round training budget"
                        ),
                    )
                )
    return rejected


def build_error_decision_trace(
    profile: DetectionErrorProfile,
    gate: ErrorProfileEvidenceGateResult,
    delta: DetectionErrorDelta | None = None,
    hypotheses: list[EvidenceLinkedHypothesis] | None = None,
    selected_family: str | None = None,
    expected_effect: str = "",
) -> ErrorDecisionTrace:
    """Assemble the round trace from measured artifacts.

    When the gate withholds budget, every non-evidence-only family is moved
    to ``rejected_actions`` and no experiment is selected — the round can
    only collect evidence, rerun evaluation, or repair dataset metadata.
    """
    linked = hypotheses if hypotheses is not None else []
    candidate_actions: list[str] = []
    seen: set[str] = set()
    for hypothesis in linked:
        for family in hypothesis.candidate_action_families:
            if family not in seen:
                seen.add(family)
                candidate_actions.append(family)

    rejected = _reject_unavailable(linked, gate)

    selected: SelectedExperiment | None = None
    if selected_family is not None and gate.budget_request_allowed:
        selected = SelectedExperiment(
            family=selected_family,
            expected_effect=expected_effect or "no explicit expectation recorded",
            round_budget_requested=True,
            gate_allows_budget=True,
        )
    elif selected_family is not None and not gate.budget_request_allowed:
        raise ValueError(
            "cannot select a budget-consuming experiment while the gate "
            f"verdict is '{gate.verdict}'"
        )

    trace = ErrorDecisionTrace(
        run_id=profile.run_id,
        round_id=f"round:{profile.candidate_id}",
        status="blocked_by_evidence" if not gate.budget_request_allowed else "pending_observation",
        problem=_problem_from(profile, delta),
        evidence_profile_id=profile.profile_id,
        evidence_delta_id=delta.candidate_profile_id if delta is not None else None,
        gate_verdict=gate.verdict,
        hypotheses=linked,
        candidate_actions=candidate_actions,
        rejected_actions=rejected,
        selected=selected,
    )
    return trace


def record_observation(
    trace: ErrorDecisionTrace,
    delta: DetectionErrorDelta,
) -> ErrorDecisionTrace:
    """Attach the post-run observed effect from a real candidate-vs-parent delta."""
    if trace.status == "blocked_by_evidence":
        raise ValueError("an evidence-blocked round has no experiment to observe")
    observed = ObservedEffect(
        delta_source=delta.candidate_profile_id,
        improvements=delta.improvements(),
        regressions=delta.regressions(),
    )
    return trace.model_copy(update={"observed": observed, "status": "observed"})


__all__ = [
    "ErrorDecisionTrace",
    "ObservedEffect",
    "RejectedAction",
    "SelectedExperiment",
    "build_error_decision_trace",
    "record_observation",
]
