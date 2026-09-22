"""The five §18K §8 scenarios, each exercised through the full workflow.

Consistent improvement, high variance, one bad seed, latency violation and an
interrupted-seed resume — all with a monkeypatched executor, no GPU, no real
training.  Each scenario asserts the four-way verdict the contract prescribes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.confirmation_approval import FullRunConfirmationApproval
from yolo_agent.agents.confirmation_workflow import (
    ConfirmationRunError,
    execute_confirmation,
    prepare_full_run_request,
)
from yolo_agent.agents.promotion_rule import PromotionRule
from yolo_agent.agents.seed_confirmation import ConfirmationRun

PRIMARY = "map50_95"
SEED_ORDER = [42, 43, 44]


def _rule() -> PromotionRule:
    return PromotionRule.model_validate(
        {
            "target_delta": 0.01,
            "max_latency_regression": 0.05,
            "max_model_size_regression": None,
        }
    )


def _approval(request_id: str, confirmation_id: str) -> FullRunConfirmationApproval:
    return FullRunConfirmationApproval.model_validate(
        {
            "approval_id": f"appr-{confirmation_id}",
            "request_id": request_id,
            "confirmation_id": confirmation_id,
            "pilot_winner_id": "trial-abc",
            "approved_by": "user",
            "explicit": True,
            "approved_gpu_hours": 12.0,
        }
    )


def _executor(
    lifts: dict[int, float], *, latency_factor: float = 1.04
) -> tuple[list[str], object]:
    """Candidate lifts per seed; baseline stays at 0.30."""
    calls: list[str] = []

    def executor(run: ConfirmationRun) -> dict[str, float]:
        calls.append(run.key)
        lift = lifts.get(run.seed, 0.0) if run.role == "candidate" else 0.0
        latency = 10.0 * (latency_factor if run.role == "candidate" else 1.0)
        return {
            PRIMARY: 0.30 + lift,
            "precision": 0.70,
            "recall": 0.60,
            "ap_small": 0.20,
            "latency_ms": latency,
        }

    return calls, executor


def _prepare(confirmation_id: str):
    return prepare_full_run_request(
        request_id=f"req-{confirmation_id}",
        confirmation_id=confirmation_id,
        pilot_winner_id="trial-abc",
        estimated_gpu_hours=6.0,
    )


def test_scenario_consistent_improvement_confirms() -> None:
    state, request = _prepare("conf-consistent")
    calls, executor = _executor({42: 0.04, 43: 0.05, 44: 0.03})

    outcome = execute_confirmation(
        state,
        request=request,
        approval=_approval(request.request_id, request.confirmation_id),
        executor=executor,  # type: ignore[arg-type]
        rule=_rule(),
    )

    assert outcome.decision.verdict == "CONFIRMED"
    assert len(calls) == 6


def test_scenario_high_variance_is_possible_not_confirmed() -> None:
    state, request = _prepare("conf-variance")
    # Large mean gain but wildly uneven across seeds -> CI straddles zero.
    _, executor = _executor({42: 0.00, 43: 0.30, 44: -0.05})

    outcome = execute_confirmation(
        state,
        request=request,
        approval=_approval(request.request_id, request.confirmation_id),
        executor=executor,  # type: ignore[arg-type]
        rule=_rule(),
    )

    assert outcome.decision.verdict == "POSSIBLE"
    assert outcome.decision.verdict != "CONFIRMED"
    primary = outcome.decision.statistics.primary  # type: ignore[union-attr]
    assert primary.mean_delta is not None and primary.mean_delta >= 0.01
    assert primary.ci95_low is not None and primary.ci95_low < 0


def test_scenario_one_bad_seed_is_inconclusive() -> None:
    state, request = _prepare("conf-bad-seed")
    # Two good seeds, one collapsed seed: the average falls below the bar.
    _, executor = _executor({42: 0.05, 43: 0.04, 44: -0.09})

    outcome = execute_confirmation(
        state,
        request=request,
        approval=_approval(request.request_id, request.confirmation_id),
        executor=executor,  # type: ignore[arg-type]
        rule=_rule(),
    )

    assert outcome.decision.verdict == "INCONCLUSIVE"
    assert outcome.decision.verdict != "CONFIRMED"
    assert any(
        r.startswith("effect_below_target") for r in outcome.decision.reasons
    )


def test_scenario_latency_violation_rejects(tmp_path: Path) -> None:
    state, request = _prepare("conf-latency")
    # Accuracy improves consistently, but candidate latency regresses 24%.
    _, executor = _executor({42: 0.04, 43: 0.04, 44: 0.04}, latency_factor=1.24)

    outcome = execute_confirmation(
        state,
        request=request,
        approval=_approval(request.request_id, request.confirmation_id),
        executor=executor,  # type: ignore[arg-type]
        rule=_rule(),
        report_paths=(
            tmp_path / "confirmation_report.yaml",
            tmp_path / "confirmation_report.md",
        ),
    )

    assert outcome.decision.verdict == "REJECTED"
    assert outcome.decision.is_confirmed is False
    assert outcome.decision.hard_constraint_violations
    # The rejection reasons are visible in both renderings.
    assert "Hard-constraint violations:" in (
        tmp_path / "confirmation_report.md"
    ).read_text(encoding="utf-8")


def test_scenario_interrupted_seed_resumes_to_a_verdict() -> None:
    state, request = _prepare("conf-interrupt")
    approval = _approval(request.request_id, request.confirmation_id)

    healthy_calls: list[str]
    healthy_calls, healthy_executor = _executor({42: 0.04, 43: 0.05, 44: 0.03})

    def dies_after_three(run: ConfirmationRun) -> dict[str, float]:
        if len(healthy_calls) >= 3:
            raise RuntimeError("process killed")
        return healthy_executor(run)  # type: ignore[operator, misc]

    with pytest.raises(ConfirmationRunError, match="process killed"):
        execute_confirmation(
            state,
            request=request,
            approval=approval,
            executor=dies_after_three,  # type: ignore[arg-type]
            rule=_rule(),
        )

    completed_before = [run.key for run in state.runs if run.is_completed]
    assert len(completed_before) == 3

    resume_calls: list[str]
    resume_calls, resume_executor = _executor({42: 0.04, 43: 0.05, 44: 0.03})
    outcome = execute_confirmation(
        state,
        request=request,
        approval=approval,
        executor=resume_executor,  # type: ignore[arg-type]
        rule=_rule(),
    )

    # Only the three unfinished seeds ran; the completed ones were reused.
    assert len(resume_calls) == 3
    assert not set(resume_calls) & set(completed_before)
    assert len(outcome.skipped_completed) == 3
    assert outcome.decision.verdict == "CONFIRMED"
