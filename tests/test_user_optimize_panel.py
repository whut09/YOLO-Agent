"""User-facing optimize panel guidance tests.

The panel is the user's only decision surface.  When the bounded automatic
search has already reached a terminal stop (no-improvement patience or
exhausted candidates), re-invoking the same command runs zero candidates, so
the panel must not tell the user to "run the same command" again.
"""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.auto_optimization_loop import AutoOptimizationResult
from yolo_agent.agents.optimize_runner import OptimizeResult
from yolo_agent.cli import _user_baseline_panel


def _result(stopped_reason: str) -> OptimizeResult:
    return OptimizeResult(
        kind="coco",
        run_id="first-training",
        run_dir=Path("runs/first-training"),
        profile="pilot",
        executor="ultralytics-train",
        executed=True,
        task_path=Path("runs/first-training/task.yaml"),
        experiment_plan_path=Path("runs/first-training/artifacts/experiment_plan.yaml"),
        queue_path=Path("runs/first-training/execution_queue.yaml"),
        queue_counts={"completed": 1},
        auto_optimization=AutoOptimizationResult(
            base_run_id="first-training",
            base_run_dir=Path("runs/first-training"),
            requested_rounds=5,
            executed=True,
            stopped_reason=stopped_reason,
            summary_path=Path("runs/first-training/artifacts/auto_summary.yaml"),
            full_candidate_recommendations_path=Path(
                "runs/first-training/artifacts/full_candidate_recommendations.yaml"
            ),
        ),
    )


def test_patience_stop_tells_user_not_to_rerun_the_same_command() -> None:
    panel = _user_baseline_panel(_result("no_improvement_patience_reached"), [])

    assert "run the same command to start automatic candidate optimization" != panel["next"]
    assert "do not rerun this run-id" in panel["next"]
    assert "no_improvement_patience" in panel["next"]


def test_exhausted_search_tells_user_not_to_rerun_the_same_command() -> None:
    panel = _user_baseline_panel(_result("method_candidates_exhausted"), [])

    assert "do not rerun this search" in panel["next"]


def test_plain_baseline_completion_still_suggests_the_same_command() -> None:
    panel = _user_baseline_panel(_result(""), [])

    assert panel["next"] == "run the same command to start automatic candidate optimization"
