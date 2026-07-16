"""Automatic optimization budget tests."""

from pathlib import Path

from yolo_agent.agents.optimize_runner import _bounded_auto_rounds
from yolo_agent.core.optimization_budget import AutoOptimizationBudget


def test_auto_budget_loads_bounded_defaults() -> None:
    budget = AutoOptimizationBudget.from_training_config(
        Path("configs/training/yolo26_coco_goal.yaml")
    )

    assert budget.mode == "auto"
    assert budget.max_gpu_hours == 24
    assert budget.max_pilots == 12
    assert budget.no_improvement_patience == 4
    assert budget.max_concurrent_pilots == 1
    assert budget.max_rounds_safety == 60
    assert budget.effective_round_limit == 60
    assert budget.full_requires_confirmation is True


def test_explicit_round_override_remains_available() -> None:
    budget = AutoOptimizationBudget.from_training_config(
        Path("configs/training/yolo26_coco_goal.yaml"),
        explicit_rounds=3,
    )

    assert budget.mode == "fixed_rounds"
    assert budget.effective_round_limit == 3


def test_round_safety_cap_is_absolute_across_resumes(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    (run_root / "coco-yolo26n-r58").mkdir(parents=True)
    (run_root / "coco-yolo26n-r59").mkdir()

    assert _bounded_auto_rounds(
        run_root=run_root,
        run_id="coco-yolo26n",
        requested_rounds=60,
        safety_limit=60,
    ) == 1
    assert _bounded_auto_rounds(
        run_root=run_root,
        run_id="coco-yolo26n",
        requested_rounds=3,
        safety_limit=59,
    ) == 0

