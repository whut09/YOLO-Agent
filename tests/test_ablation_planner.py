"""Ablation planner tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.ablation_planner import AblationPlan, AblationPlanner
from yolo_agent.agents.candidate_generator import CandidateConfig, CandidatePlan
from yolo_agent.cli import main


def _candidate(
    candidate_id: str,
    scale: str = "n",
    components: list[str] | None = None,
    train_overrides: dict[str, object] | None = None,
) -> CandidateConfig:
    return CandidateConfig(
        candidate_id=candidate_id,
        base_model=f"yolo11{scale}",
        scale=scale,
        framework="ultralytics",
        components=components or [],
        train_overrides=train_overrides or {},
    )


def test_ablation_planner_accepts_single_variable_changes() -> None:
    """Single primary-variable candidates should become ablation nodes."""
    candidates = [
        _candidate("yolo11n_baseline_n"),
        _candidate("yolo11s_scale_s", scale="s"),
        _candidate("yolo11n_nwd", components=["loss.bbox.nwd"]),
        _candidate("yolo11n_imgsz", train_overrides={"imgsz": 960}),
    ]

    plan = AblationPlanner().plan(candidates)

    assert plan.baseline_id == "yolo11n_baseline_n"
    assert len(plan.nodes) == 3
    changed_keys = [set(node.changed_variables) for node in plan.nodes]
    assert {"model_scale"} in changed_keys
    assert {"bbox_loss"} in changed_keys
    assert {"imgsz"} in changed_keys


def test_ablation_planner_marks_multi_variable_candidate_invalid() -> None:
    """Candidates changing multiple primary variables should be rejected."""
    candidates = [
        _candidate("yolo11n_baseline_n"),
        _candidate("yolo11s_nwd_imgsz", scale="s", components=["loss.bbox.nwd"], train_overrides={"imgsz": 960}),
    ]

    plan = AblationPlanner().plan(candidates)

    assert plan.nodes == []
    assert len(plan.invalid_candidates) == 1
    invalid = plan.invalid_candidates[0]
    assert invalid.candidate_id == "yolo11s_nwd_imgsz"
    assert set(invalid.changed_variables) == {"model_scale", "bbox_loss", "imgsz"}


def test_ablation_planner_requires_baseline() -> None:
    """A baseline candidate is required for interpretable ablations."""
    with pytest.raises(ValueError, match="requires a baseline"):
        AblationPlanner().plan([_candidate("yolo11n_nwd", components=["loss.bbox.nwd"])])


def test_ablate_plan_cli_writes_yaml(tmp_path: Path) -> None:
    """The CLI should write an ablation plan YAML."""
    candidate_plan = CandidatePlan(
        task_scene="generic",
        candidates=[
            _candidate("yolo11n_baseline_n"),
            _candidate("yolo11n_p2", components=["head.p2_small_object"]),
            _candidate("yolo11n_fullpad", components=["neck.fullpad"]),
        ],
    )
    plan_path = tmp_path / "plan.yaml"
    out_path = tmp_path / "ablation_plan.yaml"
    candidate_plan.to_yaml(plan_path)

    assert main(["ablate-plan", "--plan", str(plan_path), "--out", str(out_path)]) == 0

    loaded = AblationPlan.from_yaml(out_path)
    assert loaded.baseline_id == "yolo11n_baseline_n"
    assert {next(iter(node.changed_variables)) for node in loaded.nodes} == {
        "head_component",
        "neck_component",
    }

