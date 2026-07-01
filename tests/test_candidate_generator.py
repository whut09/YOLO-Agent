"""Candidate generator tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateGenerator, CandidatePlan, default_search_space_path
from yolo_agent.cli import main
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.task_spec import TaskSpec


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_DIR = ROOT / "configs" / "components"
SCENARIO_DIR = ROOT / "configs" / "scenarios"


def test_generator_includes_baselines_and_infrared_priorities() -> None:
    """Infrared small-target plans should include baselines and targeted candidates."""
    task_spec = TaskSpec.from_yaml(SCENARIO_DIR / "infrared_small_target.yaml")
    registry = ComponentRegistry.from_path(COMPONENT_DIR)
    generator = CandidateGenerator.from_yaml(registry, default_search_space_path())

    plan = generator.generate(task_spec)
    candidate_ids = {candidate.candidate_id for candidate in plan.candidates}
    component_sets = {tuple(candidate.components) for candidate in plan.candidates}

    assert {"yolo11n_baseline_n", "yolo11s_baseline_s"} <= candidate_ids
    assert ("head.p2_small_object",) in component_sets
    assert ("loss.bbox.nwd",) in component_sets
    assert ("assigner.stal",) in component_sets


def test_generator_edge_realtime_prefers_lightweight_candidates() -> None:
    """Traffic edge plans should include nano/small FP16 and lightweight-block options."""
    task_spec = TaskSpec.from_yaml(SCENARIO_DIR / "edge_realtime.yaml")
    registry = ComponentRegistry.from_path(COMPONENT_DIR)
    generator = CandidateGenerator.from_yaml(registry, default_search_space_path())

    plan = generator.generate(task_spec)

    assert any(candidate.scale == "n" for candidate in plan.candidates)
    assert any(candidate.train_overrides.get("fp16") is True for candidate in plan.candidates)
    assert any("backbone.dsconv" in candidate.components for candidate in plan.candidates)


def test_generator_writes_plan_yaml_via_cli(tmp_path: Path) -> None:
    """The plan CLI should write runs/plan.yaml-compatible output."""
    task_path = tmp_path / "task.yaml"
    out_path = tmp_path / "runs" / "plan.yaml"
    TaskSpec.from_yaml(SCENARIO_DIR / "infrared_small_target.yaml").to_yaml(task_path)

    exit_code = main(
        [
            "plan",
            "--task",
            str(task_path),
            "--components",
            str(COMPONENT_DIR),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    plan = CandidatePlan.from_yaml(out_path)
    assert plan.candidates
    assert all(candidate.risk in {"low", "medium", "high"} for candidate in plan.candidates)

