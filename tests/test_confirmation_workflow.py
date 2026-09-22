"""End-to-end confirmation workflow: approval, matched runs, resume, verdict (§18K §1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.confirmation_approval import (
    FullRunConfirmationApproval,
    FullRunConfirmationRequest,
)
from yolo_agent.agents.confirmation_report import ConfirmationReport
from yolo_agent.agents.confirmation_workflow import (
    ConfirmationApprovalDenied,
    ConfirmationRunError,
    execute_confirmation,
    outcome_summary,
    prepare_full_run_request,
)
from yolo_agent.agents.promotion_rule import PromotionRule
from yolo_agent.agents.seed_confirmation import ConfirmationRun, SeedConfirmationState

PRIMARY = "map50_95"


def _approval(request: FullRunConfirmationRequest, **overrides: object) -> FullRunConfirmationApproval:
    values: dict[str, object] = {
        "approval_id": "appr-e2e",
        "request_id": request.request_id,
        "confirmation_id": request.confirmation_id,
        "pilot_winner_id": request.pilot_winner_id,
        "approved_by": "user",
        "explicit": True,
        "approved_gpu_hours": 12.0,
    }
    values.update(overrides)
    return FullRunConfirmationApproval.model_validate(values)  # type: ignore[arg-type]


def _rule() -> PromotionRule:
    return PromotionRule.model_validate(
        {
            "target_delta": 0.01,
            "max_latency_regression": 0.05,
            "max_model_size_regression": None,
        }
    )


def _good_executor(calls: list[str]) -> ConfirmationRun:
    def executor(run: ConfirmationRun) -> dict[str, float]:
        calls.append(run.key)
        lift = 0.04 if run.role == "candidate" else 0.0
        return {
            PRIMARY: 0.30 + lift,
            "precision": 0.70,
            "recall": 0.60,
            "latency_ms": 10.4 if run.role == "candidate" else 10.0,
        }

    return executor  # type: ignore[return-value]


def test_happy_path_executes_matched_seeds_and_confirms(tmp_path: Path) -> None:
    state, request = prepare_full_run_request(
        request_id="req-1",
        confirmation_id="conf-1",
        pilot_winner_id="trial-abc",
        estimated_gpu_hours=6.0,
    )
    calls: list[str] = []

    outcome = execute_confirmation(
        state,
        request=request,
        approval=_approval(request),
        executor=_good_executor(calls),
        rule=_rule(),
        report_paths=(
            tmp_path / "confirmation_report.yaml",
            tmp_path / "confirmation_report.md",
        ),
    )

    # §1 pipeline output: all six runs, four-way decision, reports.
    assert sorted(calls) == sorted(outcome.executed)
    assert len(outcome.executed) == 6
    assert outcome.decision.verdict == "CONFIRMED"
    assert outcome.skipped_completed == []
    assert outcome.consumed_request.state == "consumed"
    assert outcome_summary(outcome)["verdict"] == "CONFIRMED"

    # §2: matched seed pairing observed by the executor itself.
    baseline = {k.split(":")[1] for k in calls if k.startswith("baseline")}
    candidate = {k.split(":")[1] for k in calls if k.startswith("candidate")}
    assert baseline == candidate == {"seed42", "seed43", "seed44"}

    # §7 artifacts written.
    assert (tmp_path / "confirmation_report.yaml").exists()
    markdown = (tmp_path / "confirmation_report.md").read_text(encoding="utf-8")
    assert "## Decision: CONFIRMED" in markdown
    assert ConfirmationReport.from_yaml(
        tmp_path / "confirmation_report.yaml"
    ).decision.verdict == "CONFIRMED"


def test_gate_denies_before_the_executor_runs_at_all() -> None:
    state, request = prepare_full_run_request(
        request_id="req-2",
        confirmation_id="conf-2",
        pilot_winner_id="trial-abc",
        estimated_gpu_hours=6.0,
    )
    calls: list[str] = []

    with pytest.raises(ConfirmationApprovalDenied, match="explicit_user_approval"):
        execute_confirmation(
            state,
            request=request,
            approval=None,  # type: ignore[arg-type]
            executor=_good_executor(calls),
            rule=_rule(),
        )

    assert calls == []
    assert all(run.status == "pending" for run in state.runs)


def test_interrupted_run_fails_closed_and_resumes_without_rerunning() -> None:
    state, request = prepare_full_run_request(
        request_id="req-3",
        confirmation_id="conf-3",
        pilot_winner_id="trial-abc",
        estimated_gpu_hours=6.0,
    )
    approval = _approval(request)
    persisted: list[SeedConfirmationState] = []

    # Executor dies after three runs (simulated interruption on run four).
    calls: list[str] = []
    healthy = _good_executor(calls)

    def flaky(run: ConfirmationRun) -> dict[str, float]:
        if len(calls) >= 3:
            raise RuntimeError("cuda lost")
        return healthy(run)

    with pytest.raises(ConfirmationRunError, match="failed: cuda lost") as excinfo:
        execute_confirmation(
            state,
            request=request,
            approval=approval,
            executor=flaky,
            rule=_rule(),
            persist=persisted.append,
        )

    assert excinfo.value.run_key in {"baseline:seed44", "candidate:seed42"}
    # State was persisted at every transition and the failed run recorded.
    assert persisted
    statuses = {run.key: run.status for run in state.runs}
    assert statuses[excinfo.value.run_key] == "failed"
    assert sum(1 for s in statuses.values() if s == "completed") == 3
    # The approval was NOT spent by the interrupted launch (only granted):
    # the resume continues under the same authorization.
    assert request.state == "approved"

    # Resume: same approval, healthy executor, completed seeds untouched.
    resume_calls: list[str] = []
    outcome = execute_confirmation(
        state,
        request=request,
        approval=approval,
        executor=_good_executor(resume_calls),
        rule=_rule(),
    )

    assert len(resume_calls) == 3
    assert not set(resume_calls) & set(calls), "completed seeds were re-run"
    assert outcome.skipped_completed and len(outcome.skipped_completed) == 3
    assert outcome.decision.verdict in {
        "CONFIRMED", "POSSIBLE", "REJECTED", "INCONCLUSIVE",
    }
    assert outcome.consumed_request.state == "consumed"


def test_consumed_confirmation_cannot_be_launched_again() -> None:
    state, request = prepare_full_run_request(
        request_id="req-4",
        confirmation_id="conf-4",
        pilot_winner_id="trial-abc",
        estimated_gpu_hours=6.0,
    )
    approval = _approval(request)
    execute_confirmation(
        state,
        request=request,
        approval=approval,
        executor=_good_executor([]),
        rule=_rule(),
    )

    # Relaunching with the same (now consumed) request must be denied.
    with pytest.raises(ConfirmationApprovalDenied, match="approval_already_consumed"):
        execute_confirmation(
            state,
            request=request,
            approval=approval,
            executor=_good_executor([]),
            rule=_rule(),
        )


def test_missing_primary_metric_fails_the_run_not_silently(tmp_path: Path) -> None:
    state, request = prepare_full_run_request(
        request_id="req-5",
        confirmation_id="conf-5",
        pilot_winner_id="trial-abc",
        estimated_gpu_hours=6.0,
    )

    def bad(run: ConfirmationRun) -> dict[str, float]:
        return {"precision": 0.7}

    with pytest.raises(ConfirmationRunError, match=PRIMARY):
        execute_confirmation(
            state,
            request=request,
            approval=_approval(request),
            executor=bad,
            rule=_rule(),
        )

    failed = [run for run in state.runs if run.status == "failed"]
    assert len(failed) == 1
