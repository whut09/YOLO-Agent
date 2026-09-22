"""Memory-aware bounded-HPO planning for a selected action (§5).

For the chosen ``ActionSpec`` the planner:

1. validates the requested parameter grid against the spec's declared
   ``search_space`` (delegating to ``request_hpo_points`` — out-of-scope
   parameters or values raise instead of being silently pruned),
2. drops grid points whose (problem, action, parameters, parent) tuple is
   already remembered as failed in :class:`ExperimentMemory`,
3. caps the number of actually-run points so HPO stays bounded,
4. keeps threshold/postprocess/calibration families on the validation
   split only (enforced downstream by ``request_hpo_points``).

No global random search exists in this module: every point comes from the
declared grid, in deterministic order.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.agents.action_space_schemas import ActionSpec
from yolo_agent.agents.bounded_hpo import HpoRequestRejected, request_hpo_points
from yolo_agent.core.detection_error_profile import DetectionErrorProfile
from yolo_agent.core.experiment_memory import ExperimentMemory

PLANNER_SCHEMA_VERSION = "hpo_candidate_plan.v1"


class HpoCandidatePlan(BaseModel):
    """The bounded, memory-filtered grid for one selected action."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PLANNER_SCHEMA_VERSION
    action_id: str
    requested_points: int
    skipped_previously_failed: int = 0
    runnable_points: list[dict[str, float | int | str]] = Field(default_factory=list)


def plan_hpo_points(
    action: ActionSpec,
    profile: DetectionErrorProfile,
    parent_fingerprint: str,
    requested_params: dict[str, list[float | int | str]],
    memory: ExperimentMemory,
    *,
    max_points: int = 8,
    split: str = "val",
) -> HpoCandidatePlan:
    """Build the runnable grid for one action under memory and budget caps.

    Raises :class:`HpoRequestRejected` when the request reaches outside the
    spec's declared search space — the caller sees contract drift instead of
    a silently smaller sweep.
    """
    try:
        grid = request_hpo_points(action, requested_params, split=split)
    except HpoRequestRejected:
        raise

    runnable: list[dict[str, float | int | str]] = []
    skipped = 0
    for point in grid:
        if len(runnable) >= max_points:
            break
        blocked = memory.check_repeat(
            profile=profile,
            action_id=action.action_id,
            parameters=point,
            parent_fingerprint=parent_fingerprint,
        )
        if blocked is not None:
            skipped += 1
            continue
        runnable.append(point)

    return HpoCandidatePlan(
        action_id=action.action_id,
        requested_points=len(grid),
        skipped_previously_failed=skipped,
        runnable_points=runnable,
    )


__all__ = [
    "HpoCandidatePlan",
    "plan_hpo_points",
]
