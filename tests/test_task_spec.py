"""TaskSpec scenario loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.cli import main
from yolo_agent.core.task_spec import TaskSpec


SCENARIO_DIR = Path(__file__).resolve().parents[1] / "configs" / "scenarios"


@pytest.mark.parametrize(
    "scenario_name",
    ["infrared_small_target", "industrial_defect", "edge_realtime"],
)
def test_scenario_yaml_loads_as_task_spec(scenario_name: str) -> None:
    """Each bundled scenario YAML should validate as a TaskSpec."""
    task_spec = TaskSpec.from_yaml(SCENARIO_DIR / f"{scenario_name}.yaml")

    assert task_spec.class_names
    assert task_spec.primary_metric.name


def test_init_generates_task_yaml_from_scenario(tmp_path: Path) -> None:
    """The init command should materialize a validated task.yaml."""
    output_path = tmp_path / "task.yaml"

    assert main(["init", "--scenario", "edge_realtime", "--output", str(output_path)]) == 0

    generated = TaskSpec.from_yaml(output_path)
    assert generated.scene == "traffic_edge"
    assert generated.target_fps == 30

