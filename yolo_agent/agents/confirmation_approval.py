"""Explicit full-run approval gate for three-seed confirmation (Prompt-18K §5).

A pilot winner never turns into GPU spend by itself.  The agent may prepare a
``FullRunConfirmationRequest`` (planned runs + estimated cost) after a pilot
wins, but launching the six-run matrix requires a user-issued
``FullRunConfirmationApproval`` that:

- is bound to this exact request / confirmation / pilot winner (a stale
  approval from an earlier pilot round does not carry over),
- is explicitly marked as such and issued by the user, never self-issued,
- budgets at least the estimated GPU hours of the planned runs.

Approvals are single-use: consuming one marks the request so a second launch
cannot silently ride the same authorization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ApprovalRequestState = Literal["pending", "approved", "rejected", "consumed"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FullRunConfirmationRequest(BaseModel):
    """Agent-prepared request for the six-run full confirmation."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    confirmation_id: str = Field(min_length=1)
    pilot_winner_id: str = Field(min_length=1)
    planned_runs: int = Field(default=6, ge=1)
    estimated_gpu_hours: float = Field(gt=0.0)
    state: ApprovalRequestState = "pending"
    created_at: datetime = Field(default_factory=_utc_now)

    @classmethod
    def from_pilot_winner(
        cls,
        *,
        request_id: str,
        confirmation_id: str,
        pilot_winner_id: str,
        estimated_gpu_hours: float,
    ) -> FullRunConfirmationRequest:
        """Prepare (but never approve) the full-run request after a pilot win."""
        return cls(
            request_id=request_id,
            confirmation_id=confirmation_id,
            pilot_winner_id=pilot_winner_id,
            estimated_gpu_hours=estimated_gpu_hours,
        )


class FullRunConfirmationApproval(BaseModel):
    """User-issued, single-use authorization to spend the full-run budget."""

    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    confirmation_id: str = Field(min_length=1)
    pilot_winner_id: str = Field(min_length=1)
    #: Only the human user can issue this; the agent has no construction path
    #: that satisfies the gate without one.
    approved_by: Literal["user"]
    #: Explicitness is a required, declared property of the approval.
    explicit: bool
    approved_gpu_hours: float = Field(ge=0.0)
    approved_at: datetime = Field(default_factory=_utc_now)


class ApprovalDecision(BaseModel):
    """Whether the six-run launch may proceed, and why not when denied."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_full_run_approval(
    request: FullRunConfirmationRequest,
    approval: FullRunConfirmationApproval | None = None,
) -> ApprovalDecision:
    """Decide whether the planned six runs may consume GPU budget."""
    reasons: list[str] = []

    if request.state == "rejected":
        reasons.append("request_rejected")
    if request.state == "consumed":
        reasons.append("approval_already_consumed")

    if approval is None:
        reasons.append("explicit_user_approval_required")
        return ApprovalDecision(allowed=False, reasons=reasons)

    if approval.request_id != request.request_id:
        reasons.append("approval_request_mismatch")
    if approval.confirmation_id != request.confirmation_id:
        reasons.append(
            f"stale_approval_binding:confirmation={approval.confirmation_id}"
        )
    if approval.pilot_winner_id != request.pilot_winner_id:
        reasons.append(
            f"stale_approval_binding:pilot_winner={approval.pilot_winner_id}"
        )
    if not approval.explicit:
        reasons.append("approval_not_explicit")
    if approval.approved_by != "user":  # pragma: no cover - Literal-typed
        reasons.append("approval_not_user_issued")
    if approval.approved_gpu_hours < request.estimated_gpu_hours:
        reasons.append(
            "approved_budget_below_estimate:"
            f"{approval.approved_gpu_hours}<{request.estimated_gpu_hours}"
        )

    return ApprovalDecision(allowed=not reasons, reasons=reasons)


def grant_approval(
    request: FullRunConfirmationRequest,
    approval: FullRunConfirmationApproval,
) -> FullRunConfirmationRequest:
    """Validated approval flips the request into ``approved`` (still spendable)."""
    decision = evaluate_full_run_approval(request, approval)
    if not decision.allowed:
        raise ValueError(
            "cannot grant an approval the gate denies: " + "; ".join(decision.reasons)
        )
    return request.model_copy(update={"state": "approved"})


def consume_approval(
    request: FullRunConfirmationRequest,
) -> FullRunConfirmationRequest:
    """Mark the request as spent once the six-run launch begins (single-use)."""
    if request.state != "approved":
        raise ValueError(
            f"only an approved request can be consumed, not '{request.state}'"
        )
    return request.model_copy(update={"state": "consumed"})


__all__ = [
    "ApprovalDecision",
    "FullRunConfirmationApproval",
    "FullRunConfirmationRequest",
    "consume_approval",
    "evaluate_full_run_approval",
    "grant_approval",
]
