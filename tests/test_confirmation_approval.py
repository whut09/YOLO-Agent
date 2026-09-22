"""Approval gate: no full-run GPU budget without explicit user sign-off (§18K §5)."""

from __future__ import annotations

import pytest

from yolo_agent.agents.confirmation_approval import (
    FullRunConfirmationApproval,
    FullRunConfirmationRequest,
    consume_approval,
    evaluate_full_run_approval,
    grant_approval,
)


def _request(**overrides: object) -> FullRunConfirmationRequest:
    values: dict[str, object] = {
        "request_id": "req-1",
        "confirmation_id": "conf-1",
        "pilot_winner_id": "trial-abc",
        "estimated_gpu_hours": 6.0,
    }
    values.update(overrides)
    return FullRunConfirmationRequest.from_pilot_winner(  # type: ignore[arg-type]
        request_id=str(values["request_id"]),
        confirmation_id=str(values["confirmation_id"]),
        pilot_winner_id=str(values["pilot_winner_id"]),
        estimated_gpu_hours=float(values["estimated_gpu_hours"]),  # type: ignore[arg-type]
    )


def _approval(**overrides: object) -> FullRunConfirmationApproval:
    values: dict[str, object] = {
        "approval_id": "appr-1",
        "request_id": "req-1",
        "confirmation_id": "conf-1",
        "pilot_winner_id": "trial-abc",
        "approved_by": "user",
        "explicit": True,
        "approved_gpu_hours": 8.0,
    }
    values.update(overrides)
    return FullRunConfirmationApproval.model_validate(values)  # type: ignore[arg-type]


def test_pilot_winner_only_prepares_request_without_approval() -> None:
    """The agent may prepare the request; the gate stays closed by default."""
    request = _request()

    assert request.state == "pending"

    decision = evaluate_full_run_approval(request)

    assert decision.allowed is False
    assert "explicit_user_approval_required" in decision.reasons


def test_explicit_user_approval_with_budget_opens_gate() -> None:
    request = grant_approval(_request(), _approval())

    decision = evaluate_full_run_approval(request, _approval())

    assert decision.allowed is True
    assert decision.reasons == []
    assert request.state == "approved"


def test_stale_approval_from_other_pilot_winner_is_denied() -> None:
    stale = _approval(pilot_winner_id="trial-old")

    decision = evaluate_full_run_approval(_request(), stale)

    assert decision.allowed is False
    assert any(r.startswith("stale_approval_binding") for r in decision.reasons)


def test_approval_for_other_request_is_denied() -> None:
    decision = evaluate_full_run_approval(_request(), _approval(request_id="req-2"))

    assert decision.allowed is False
    assert "approval_request_mismatch" in decision.reasons


def test_non_explicit_approval_is_denied() -> None:
    decision = evaluate_full_run_approval(_request(), _approval(explicit=False))

    assert decision.allowed is False
    assert "approval_not_explicit" in decision.reasons


def test_budget_below_estimate_is_denied() -> None:
    decision = evaluate_full_run_approval(
        _request(), _approval(approved_gpu_hours=3.0)
    )

    assert decision.allowed is False
    assert any(r.startswith("approved_budget_below_estimate") for r in decision.reasons)


def test_consumed_approval_is_single_use() -> None:
    approved = grant_approval(_request(), _approval())
    spent = consume_approval(approved)

    assert spent.state == "consumed"

    decision = evaluate_full_run_approval(spent, _approval())

    assert decision.allowed is False
    assert "approval_already_consumed" in decision.reasons


def test_rejected_request_stays_closed_even_with_approval() -> None:
    rejected = _request().model_copy(update={"state": "rejected"})

    decision = evaluate_full_run_approval(rejected, _approval())

    assert decision.allowed is False
    assert "request_rejected" in decision.reasons


def test_grant_refuses_a_denied_approval() -> None:
    with pytest.raises(ValueError, match="gate denies"):
        grant_approval(_request(), _approval(approved_gpu_hours=1.0))


def test_consume_requires_an_approved_request() -> None:
    with pytest.raises(ValueError, match="approved request"):
        consume_approval(_request())
