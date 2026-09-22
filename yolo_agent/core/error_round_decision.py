"""TaskSpec-driven round decisions from measured error deltas (Prompt-18J §2/§6).

The engine consumes a real :class:`DetectionErrorDelta` — never a headline
mAP number alone — plus the deployment constraints of the
:class:`~yolo_agent.core.task_spec.TaskSpec`, and emits one of the unified
decisions:

``promote`` / ``reject`` / ``refine`` / ``rollback`` /
``collect_evidence`` / ``collect_data`` / ``request_annotation`` /
``stop_goal_met`` / ``stop_budget`` / ``stop_exhausted``.

Accuracy gains never speak for themselves: a candidate that improves the
objective while violating a hard deployment constraint (latency, model
size) is ``rejected`` or ``rolled back``; a candidate that improves the
objective but regresses a non-objective error surface is ``refined``
instead of promoted; a candidate whose evidence is incomplete cannot ask
for a budget at all.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.detection_error_delta import DetectionErrorDelta
from yolo_agent.core.task_spec import TaskSpec

ROUND_DECISION_SCHEMA_VERSION = "round_decision.v1"

RoundDecisionAction = Literal[
    "promote",
    "reject",
    "refine",
    "rollback",
    "collect_evidence",
    "collect_data",
    "request_annotation",
    "stop_goal_met",
    "stop_budget",
    "stop_exhausted",
]

#: Actions that consume training budget.  Every other decision defers or
#: bookkeeps instead of training.
BUDGET_ACTIONS: frozenset[str] = frozenset({"promote", "refine"})


class DecisionObjection(BaseModel):
    """One measured reason a decision went the way it did."""

    model_config = ConfigDict(extra="forbid")

    code: str
    detail: str
    metric: str | None = None
    delta_value: float | None = None


class RoundDecision(BaseModel):
    """The verdict for one round, derived from the full delta surface."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = ROUND_DECISION_SCHEMA_VERSION
    run_id: str
    candidate_id: str
    parent_candidate_id: str
    decision: RoundDecisionAction
    objections: list[DecisionObjection] = Field(default_factory=list)
    acknowledgements: list[str] = Field(default_factory=list)
    goal_met: bool = False

    @property
    def consumes_budget(self) -> bool:
        return self.decision in BUDGET_ACTIONS


def _objection(code: str, detail: str, *, metric: str | None = None,
               delta_value: float | None = None) -> DecisionObjection:
    return DecisionObjection(
        code=code, detail=detail, metric=metric, delta_value=delta_value
    )


def _metric_movement(delta: DetectionErrorDelta, section: str, metric: str):
    found = delta.metric(section, metric)
    return None if found is None else found.delta


def decide_next_round(
    delta: DetectionErrorDelta,
    task_spec: TaskSpec,
    *,
    budget_remaining: int = 1,
    rounds_exhausted: bool = False,
) -> RoundDecision:
    """Judge one candidate round from its complete error delta.

    The decision order encodes the Prompt-18J priorities:

    1. evidence adequacy (matched evaluation) — else ``collect_evidence``
    2. budget bookkeeping — ``stop_budget`` / ``stop_exhausted``
    3. hard deployment constraints from the TaskSpec — reject / rollback
    4. non-objective error regressions — refine instead of promote
    5. the weighted TaskSpec objective — promote, or keep collecting data
       / annotations when the objective is data-limited
    """
    objections: list[DecisionObjection] = []
    acknowledgements: list[str] = list(delta.improvements())

    if not delta.matched_evaluation:
        objections.append(
            _objection(
                "evidence_not_matched",
                "candidate and parent were not evaluated on the same GT "
                "artifact and dataset manifest; the movement is not usable",
            )
        )
        return RoundDecision(
            run_id=delta.run_id,
            candidate_id=delta.candidate_id,
            parent_candidate_id=delta.parent_candidate_id,
            decision="collect_evidence",
            objections=objections,
        )

    if rounds_exhausted:
        return RoundDecision(
            run_id=delta.run_id,
            candidate_id=delta.candidate_id,
            parent_candidate_id=delta.parent_candidate_id,
            decision="stop_exhausted",
            acknowledgements=acknowledgements,
        )
    if budget_remaining <= 0:
        return RoundDecision(
            run_id=delta.run_id,
            candidate_id=delta.candidate_id,
            parent_candidate_id=delta.parent_candidate_id,
            decision="stop_budget",
            acknowledgements=acknowledgements,
        )

    # --- hard deployment constraints (TaskSpec is the authority) -------------
    constraint_objections: list[DecisionObjection] = []
    resources = delta.resources
    if resources is not None:
        if task_spec.max_latency_ms is not None:
            latency = next(
                (item for item in resources.metrics if item.metric == "latency_ms"),
                None,
            )
            if latency is not None and latency.candidate is not None:
                if latency.candidate > task_spec.max_latency_ms:
                    constraint_objections.append(
                        _objection(
                            "latency_constraint_violated",
                            f"candidate latency {latency.candidate:.1f}ms exceeds "
                            f"the TaskSpec ceiling {task_spec.max_latency_ms:.1f}ms",
                            metric="latency_ms",
                            delta_value=latency.delta,
                        )
                    )
        params = next(
            (item for item in resources.counts if item.metric == "params"),
            None,
        )
        if params is not None and task_spec.max_model_size_mb is not None:
            ratio = params.candidate / max(params.parent, 1)
            if ratio > 1.5:
                constraint_objections.append(
                    _objection(
                        "model_size_guard_tripped",
                        "candidate grew parameters by "
                        f"{(ratio - 1.0) * 100:.0f}% against the TaskSpec size ceiling",
                        metric="params",
                        delta_value=float(params.delta),
                    )
                )

    primary = delta.metric("global", task_spec.primary_metric.name)
    if primary is None or primary.delta is None:
        objections.append(
            _objection(
                "objective_not_measured",
                f"the TaskSpec objective '{task_spec.primary_metric.name}' "
                "is missing from the matched delta",
            )
        )
        return RoundDecision(
            run_id=delta.run_id,
            candidate_id=delta.candidate_id,
            parent_candidate_id=delta.parent_candidate_id,
            decision="collect_evidence",
            objections=objections,
        )

    if constraint_objections:
        # If the candidate still improved the primary objective, the change is
        # directionally right but unusable as-is: roll back to the parent and
        # record the constraint.  If it also failed the objective, plain reject.
        if primary.delta and primary.delta > 0:
            decision: RoundDecisionAction = "rollback"
        else:
            decision = "reject"
        return RoundDecision(
            run_id=delta.run_id,
            candidate_id=delta.candidate_id,
            parent_candidate_id=delta.parent_candidate_id,
            decision=decision,
            objections=objections + constraint_objections,
            acknowledgements=acknowledgements,
        )

    # --- objective movement decides the branch -------------------------------
    if primary.delta is not None and primary.delta > 0:
        if delta.has_regressions():
            objections.append(
                _objection(
                    "non_objective_regressions",
                    "the candidate regressed error surfaces outside the primary "
                    "objective: " + ", ".join(delta.regressions()[:6]),
                )
            )
            return RoundDecision(
                run_id=delta.run_id,
                candidate_id=delta.candidate_id,
                parent_candidate_id=delta.parent_candidate_id,
                decision="refine",
                objections=objections,
                acknowledgements=acknowledgements,
            )
        return RoundDecision(
            run_id=delta.run_id,
            candidate_id=delta.candidate_id,
            parent_candidate_id=delta.parent_candidate_id,
            decision="promote",
            acknowledgements=acknowledgements,
        )
    if primary.delta is not None and primary.delta < 0:
        # The objective itself moved backwards with no regression elsewhere:
        # the data side is the likely bottleneck.
        objections.append(
            _objection(
                "objective_regressed",
                f"the TaskSpec objective '{task_spec.primary_metric.name}' "
                f"moved {primary.delta:+.4f} with no other regression to blame",
                metric=task_spec.primary_metric.name,
                delta_value=primary.delta,
            )
        )
        if task_spec.dataset is not None and task_spec.dataset.imbalance in {
            "medium",
            "high",
        }:
            return RoundDecision(
                run_id=delta.run_id,
                candidate_id=delta.candidate_id,
                parent_candidate_id=delta.parent_candidate_id,
                decision="collect_data",
                objections=objections,
            )
        return RoundDecision(
            run_id=delta.run_id,
            candidate_id=delta.candidate_id,
            parent_candidate_id=delta.parent_candidate_id,
            decision="request_annotation",
            objections=objections,
        )

    # Objective flat: a regression elsewhere means the candidate traded one
    # surface for nothing — refine.  Only a flat objective with nothing
    # regressed has genuinely nothing left to learn this round.
    if delta.has_regressions():
        objections.append(
            _objection(
                "non_objective_regressions",
                "the objective is flat but these surfaces regressed: "
                + ", ".join(delta.regressions()[:6]),
            )
        )
        return RoundDecision(
            run_id=delta.run_id,
            candidate_id=delta.candidate_id,
            parent_candidate_id=delta.parent_candidate_id,
            decision="refine",
            objections=objections,
            acknowledgements=acknowledgements,
        )
    return RoundDecision(
        run_id=delta.run_id,
        candidate_id=delta.candidate_id,
        parent_candidate_id=delta.parent_candidate_id,
        decision="stop_goal_met",
        acknowledgements=acknowledgements,
    )


__all__ = [
    "BUDGET_ACTIONS",
    "DecisionObjection",
    "RoundDecision",
    "RoundDecisionAction",
    "decide_next_round",
]
