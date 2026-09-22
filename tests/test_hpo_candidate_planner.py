"""Memory-aware bounded-HPO planning (Prompt-18J §5)."""

from __future__ import annotations

import pytest

from yolo_agent.agents.action_space import ActionCatalog
from yolo_agent.agents.bounded_hpo import HpoRequestRejected
from yolo_agent.core.detection_error_profile import (
    DetectionErrorProfile,
    FalseNegativeFacts,
    FalsePositiveFacts,
    GlobalMetrics,
    ScaleMetrics,
)
from yolo_agent.core.experiment_memory import ExperimentMemory
from yolo_agent.core.hpo_candidate_planner import plan_hpo_points

_CATALOG = ActionCatalog.from_yaml("configs/actions/detection_action_catalog.yaml")
_ACTION = next(
    a
    for a in _CATALOG.actions
    if a.action_id == "data.sampling.hard_negative_mining"
)


def _profile() -> DetectionErrorProfile:
    return DetectionErrorProfile(
        profile_id="prof-1",
        run_id="run-18j",
        candidate_id="cand",
        global_=GlobalMetrics(map50=0.5, map50_95=0.3, precision=0.7, recall=0.6),
        scale=ScaleMetrics(ap_small=0.3, ap_medium=0.5, ap_large=0.6),
        false_negative=FalseNegativeFacts(total=10),
        false_positive=FalsePositiveFacts(total=8),
    )


def test_grid_points_come_from_the_declared_search_space() -> None:
    plan = plan_hpo_points(
        _ACTION,
        _profile(),
        "parent-1",
        {"topk_per_image": [3, 6, 12]},
        ExperimentMemory(),
    )

    assert plan.runnable_points == [
        {"topk_per_image": 3},
        {"topk_per_image": 6},
        {"topk_per_image": 12},
    ]
    assert plan.skipped_previously_failed == 0


def test_previously_failed_points_are_skipped() -> None:
    memory = ExperimentMemory()
    profile = _profile()
    memory.record(
        profile=profile,
        action_id=_ACTION.action_id,
        parameters={"topk_per_image": 3},
        parent_fingerprint="parent-1",
        outcome="failed",
        candidate_id="cand-1",
    )

    plan = plan_hpo_points(
        _ACTION, profile, "parent-1", {"topk_per_image": [3, 6, 12]}, memory
    )

    assert plan.skipped_previously_failed == 1
    assert {"topk_per_image": 3} not in plan.runnable_points


def test_max_points_caps_the_runnable_grid() -> None:
    memory = ExperimentMemory()
    plan = plan_hpo_points(
        _ACTION,
        _profile(),
        "parent-1",
        {"topk_per_image": [3, 6, 12]},
        memory,
        max_points=2,
    )

    assert len(plan.runnable_points) == 2


def test_out_of_scope_parameter_is_rejected_not_pruned() -> None:
    with pytest.raises(HpoRequestRejected):
        plan_hpo_points(
            _ACTION,
            _profile(),
            "parent-1",
            {"magic_gamma": [0.5]},
            ExperimentMemory(),
        )


def test_value_outside_declared_grid_is_rejected() -> None:
    with pytest.raises(HpoRequestRejected):
        plan_hpo_points(
            _ACTION,
            _profile(),
            "parent-1",
            {"topk_per_image": [7]},
            ExperimentMemory(),
        )
