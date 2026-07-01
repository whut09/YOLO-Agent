"""Experiment graph schema tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan


def _candidate() -> CandidateConfig:
    return CandidateConfig(
        candidate_id="yolo11n_baseline_n",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
    )


def test_experiment_plan_roundtrip(tmp_path: Path) -> None:
    """Experiment plans should serialize and reload with candidate details."""
    plan = ExperimentPlan(
        plan_id="plan-001",
        nodes=[
            ExperimentNode(
                node_id="run-001",
                candidate_config=_candidate(),
                data_version="dataset@sha256:abc",
                seed=7,
                command="yolo train model=generated_models/yolo11n.yaml seed=7",
                changed_variables={"scale": "n"},
            )
        ],
    )
    output_path = tmp_path / "experiment_plan.yaml"

    plan.to_yaml(output_path)
    loaded = ExperimentPlan.from_yaml(output_path)

    assert loaded.plan_id == "plan-001"
    assert loaded.nodes[0].candidate_config.candidate_id == "yolo11n_baseline_n"
    assert loaded.nodes[0].status == "planned"
    assert loaded.nodes[0].seed == 7

