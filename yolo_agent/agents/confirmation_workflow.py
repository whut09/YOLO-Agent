"""End-to-end three-seed confirmation workflow (Prompt-18K §1/§7).

The full promotion pipeline, orchestrated in one place:

    pilot winner
        -> explicit full-run approval (cost gate, §5)
        -> baseline seed 1/2/3 + candidate seed 1/2/3 (matched, §2;
           resume-safe per §6)
        -> paired statistics + 95% CI (§3)
        -> four-way promotion decision (§4)
        -> confirmation_report.yaml + Markdown (§7)

The workflow never touches a GPU itself: run execution is an injected
``executor`` callable, so every orchestration path (including interruption and
resume) is testable without real training.  The approval gate is evaluated
*before* the executor is invoked a single time — without a user approval the
pipeline stops at the cost gate.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.agents.confirmation_approval import (
    FullRunConfirmationApproval,
    FullRunConfirmationRequest,
    consume_approval,
    evaluate_full_run_approval,
    grant_approval,
)
from yolo_agent.agents.confirmation_report import (
    ConfirmationReport,
    build_confirmation_report,
    write_confirmation_report,
)
from yolo_agent.agents.confirmation_resume import ResumePlan, plan_resume
from yolo_agent.agents.promotion_rule import (
    PromotionDecision,
    PromotionRule,
    apply_promotion_rule,
)
from yolo_agent.agents.seed_confirmation import ConfirmationRun, SeedConfirmationState
from yolo_agent.core.task_spec import TaskSpec

#: Executes one confirmation run and returns its recorded metrics.
RunExecutor = Callable[[ConfirmationRun], dict[str, float]]
#: Persists the state after every transition (crash-safe resume).
PersistHook = Callable[[SeedConfirmationState], None]


class ConfirmationApprovalDenied(ValueError):
    """Raised before any run executes when the cost gate stays closed."""


class ConfirmationRunError(RuntimeError):
    """A run failed or was interrupted; the state is persisted for resume."""

    def __init__(self, run_key: str, cause: str) -> None:
        super().__init__(f"confirmation run {run_key} failed: {cause}")
        self.run_key = run_key
        self.cause = cause


class ConfirmationOutcome(BaseModel):
    """Everything one workflow invocation produced."""

    model_config = ConfigDict(extra="forbid")

    decision: PromotionDecision
    report: ConfirmationReport
    #: Keys executed during this invocation, in matched-seed order.
    executed: list[str] = Field(default_factory=list)
    #: Completed seeds reused without re-running (§6).
    skipped_completed: list[str] = Field(default_factory=list)
    resume_plan: ResumePlan
    #: The request after being spent (single-use approval).
    consumed_request: FullRunConfirmationRequest


def _find_run(state: SeedConfirmationState, key: str) -> ConfirmationRun:
    for run in state.runs:
        if run.key == key:
            return run
    raise KeyError(f"no confirmation run with key {key}")


def execute_confirmation(
    state: SeedConfirmationState,
    *,
    request: FullRunConfirmationRequest,
    approval: FullRunConfirmationApproval,
    executor: RunExecutor,
    rule: PromotionRule,
    task_spec: TaskSpec | None = None,
    persist: PersistHook | None = None,
    report_paths: tuple[Path | str, Path | str] | None = None,
) -> ConfirmationOutcome:
    """Run (or resume) the six-run matrix and produce the four-way decision.

    Order is deliberate: the cost gate is evaluated before the executor is
    invoked even once, finished seeds are skipped, and each transition is
    persisted so an interrupted confirmation resumes exactly where it stopped.
    """
    # §5: explicit user approval before any GPU-facing execution.
    gate = evaluate_full_run_approval(request, approval)
    if not gate.allowed:
        raise ConfirmationApprovalDenied(
            "full-run confirmation denied: " + "; ".join(gate.reasons)
        )
    approved_request = grant_approval(request, approval)
    # Sync the caller's request in place: a stale copy must never be able to
    # relaunch a spent approval (fail-closed lifecycle).
    request.state = approved_request.state

    plan = plan_resume(state)
    executed: list[str] = []

    for key in plan.execute:
        run = _find_run(state, key)
        state.mark_running(run.role, run.seed)
        if persist is not None:
            persist(state)
        try:
            metrics = dict(executor(run))
            state.mark_completed(run.role, run.seed, metrics)
        except ValueError as exc:
            # Missing primary metric etc.: a failed run, not a live process.
            state.mark_failed(run.role, run.seed, str(exc))
            if persist is not None:
                persist(state)
            raise ConfirmationRunError(key, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - executor boundary
            state.mark_failed(run.role, run.seed, str(exc))
            if persist is not None:
                persist(state)
            raise ConfirmationRunError(key, str(exc)) from exc
        executed.append(key)
        if persist is not None:
            persist(state)

    final_plan = plan_resume(state)
    if not final_plan.ready_for_statistics:  # pragma: no cover - defensive
        raise RuntimeError(
            "confirmation matrix still incomplete after execution: "
            + ", ".join(final_plan.execute)
        )

    # The approval is spent only now: an interrupted confirmation keeps its
    # authorization so the resume can continue the same six runs, while a
    # finished one becomes single-use (the consumed request cannot relaunch).
    spent_request = consume_approval(approved_request)
    request.state = spent_request.state

    decision = apply_promotion_rule(state, rule, task_spec=task_spec)
    report = build_confirmation_report(state, decision)
    if report_paths is not None:
        write_confirmation_report(report, report_paths[0], report_paths[1])

    return ConfirmationOutcome(
        decision=decision,
        report=report,
        executed=executed,
        skipped_completed=plan.skip_completed,
        resume_plan=final_plan,
        consumed_request=spent_request,
    )


def prepare_full_run_request(
    *,
    request_id: str,
    confirmation_id: str,
    pilot_winner_id: str,
    estimated_gpu_hours: float,
) -> tuple[SeedConfirmationState, FullRunConfirmationRequest]:
    """Pilot win -> pending state + pending request (no approval, no spend)."""
    state = SeedConfirmationState.create(
        confirmation_id=confirmation_id, pilot_winner_id=pilot_winner_id
    )
    request = FullRunConfirmationRequest.from_pilot_winner(
        request_id=request_id,
        confirmation_id=confirmation_id,
        pilot_winner_id=pilot_winner_id,
        estimated_gpu_hours=estimated_gpu_hours,
    )
    return state, request


def resume_confirmation(
    state: SeedConfirmationState,
    *,
    request: FullRunConfirmationRequest,
    approval: FullRunConfirmationApproval | None,
    executor: RunExecutor,
    rule: PromotionRule,
    task_spec: TaskSpec | None = None,
    persist: PersistHook | None = None,
    report_paths: tuple[Path | str, Path | str] | None = None,
) -> ConfirmationOutcome:
    """Resume after a crash.

    The original approval is re-submitted here; because a consumed request is
    single-use, resuming *continues* a matrix whose approval was already spent
    only if the caller passes the approval bound to the still-pending request.
    The completed seeds are never re-executed either way.
    """
    if approval is None:
        raise ConfirmationApprovalDenied(
            "resume requires the original explicit user approval"
        )
    return execute_confirmation(
        state,
        request=request,
        approval=approval,
        executor=executor,
        rule=rule,
        task_spec=task_spec,
        persist=persist,
        report_paths=report_paths,
    )


def outcome_summary(outcome: ConfirmationOutcome) -> dict[str, Any]:
    """Compact machine summary of one workflow invocation."""
    return {
        "verdict": outcome.decision.verdict,
        "confirmation_id": outcome.report.confirmation_id,
        "executed": len(outcome.executed),
        "skipped_completed": len(outcome.skipped_completed),
        "reasons": list(outcome.decision.reasons),
        "violations": list(outcome.decision.hard_constraint_violations),
    }


__all__ = [
    "ConfirmationApprovalDenied",
    "ConfirmationOutcome",
    "ConfirmationRunError",
    "RunExecutor",
    "execute_confirmation",
    "outcome_summary",
    "prepare_full_run_request",
    "resume_confirmation",
]
