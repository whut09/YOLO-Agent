"""Close the loop: observed trace + TaskSpec verdict → next decision (§7).

``record_observation`` already attaches the real candidate-vs-parent delta
to an :class:`ErrorDecisionTrace`; this module adds the *next round's
decision* to the observed trace and writes the final artifact.

The trace therefore records the full causal chain the Prompt-18J contract
requires: observed problem → evidence → hypotheses → eligible actions →
rejected actions → selected experiment → expected effect → observed error
delta → next decision.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from yolo_agent.core.detection_error_delta import DetectionErrorDelta
from yolo_agent.core.detection_error_profile import (
    DetectionErrorProfile,
    FalseNegativeFacts,
    FalsePositiveFacts,
    GlobalMetrics,
    ScaleMetrics,
)
from yolo_agent.core.error_decision_trace import (
    ErrorDecisionTrace,
    record_observation,
)
from yolo_agent.core.error_round_artifacts import write_decision_trace_artifact
from yolo_agent.core.error_round_decision import (
    RoundDecision,
    decide_next_round,
)
from yolo_agent.core.experiment_memory import ExperimentMemory, Outcome
from yolo_agent.core.task_spec import TaskSpec

NEXT_TRACE_SCHEMA_VERSION = "decision_trace_next.v1"

_MEMORY_OUTCOME: dict[str, Outcome] = {
    "promote": "promoted",
    "refine": "refined",
    "reject": "rejected",
    "rollback": "rolled_back",
}


class ObservedTraceWithDecision(BaseModel):
    """The observed trace plus the verdict for the following round."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = NEXT_TRACE_SCHEMA_VERSION
    trace: ErrorDecisionTrace
    next_decision: RoundDecision


def close_round(
    trace: ErrorDecisionTrace,
    delta: DetectionErrorDelta,
    task_spec: TaskSpec,
    memory: ExperimentMemory,
    *,
    selected_action_id: str | None = None,
    selected_parameters: dict | None = None,
    parent_fingerprint: str = "",
    artifact_path: Path | str | None = None,
    budget_remaining: int = 1,
    rounds_exhausted: bool = False,
) -> ObservedTraceWithDecision:
    """Observe the round, remember its outcome, and decide the next move."""
    observed_trace = record_observation(trace, delta)
    decision = decide_next_round(
        delta,
        task_spec,
        budget_remaining=budget_remaining,
        rounds_exhausted=rounds_exhausted,
    )

    if (
        selected_action_id is not None
        and selected_parameters is not None
        and decision.decision in _MEMORY_OUTCOME
    ):
        # Memory fingerprints need the round's error structure; rebuild the
        # minimal profile from the observed delta (structure, not run ids).
        memory.record(
            profile=_movement_profile(trace, delta),
            action_id=selected_action_id,
            parameters=selected_parameters,
            parent_fingerprint=parent_fingerprint,
            outcome=_MEMORY_OUTCOME[decision.decision],
            candidate_id=delta.candidate_id,
        )

    if artifact_path is not None:
        write_decision_trace_artifact(observed_trace, Path(artifact_path).parent)

    return ObservedTraceWithDecision(trace=observed_trace, next_decision=decision)


def _movement_profile(
    trace: ErrorDecisionTrace, delta: DetectionErrorDelta
) -> DetectionErrorProfile:
    """Rebuild the minimal error structure the memory fingerprints need."""
    global_delta = delta.metric("global", "map50_95")
    fn_delta = next(
        (
            item
            for section in delta.sections
            if section.section == "false_negative"
            for item in section.counts
            if item.metric == "total"
        ),
        None,
    )
    fp_delta = next(
        (
            item
            for section in delta.sections
            if section.section == "false_positive"
            for item in section.counts
            if item.metric == "total"
        ),
        None,
    )
    return DetectionErrorProfile(
        profile_id=trace.evidence_profile_id,
        run_id=delta.run_id,
        candidate_id=delta.candidate_id,
        global_=GlobalMetrics(
            map50=global_delta.candidate if global_delta else 0.0,
            map50_95=global_delta.candidate if global_delta else 0.0,
            precision=0.0,
            recall=0.0,
        ),
        scale=ScaleMetrics(),
        false_negative=FalseNegativeFacts(total=fn_delta.candidate if fn_delta else 0),
        false_positive=FalsePositiveFacts(total=fp_delta.candidate if fp_delta else 0),
    )


__all__ = [
    "ObservedTraceWithDecision",
    "close_round",
]
