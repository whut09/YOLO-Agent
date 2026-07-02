"""Component contribution planner tests."""

from __future__ import annotations

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.component_contribution import ComponentContributionPlanner


def _baseline() -> CandidateConfig:
    return CandidateConfig(
        candidate_id="baseline",
        base_model="yolo11s",
        scale="s",
        framework="ultralytics",
    )


def test_builds_cumulative_ablation_matrix() -> None:
    """Planner should create baseline, +A, +A+B, +A+B+C nodes."""
    planner = ComponentContributionPlanner()

    matrix = planner.build_matrix(
        _baseline(),
        ["loss.bbox.nwd", "head.p2_small_object", "neck.fullpad"],
    )

    assert [node.component_set for node in matrix.nodes] == [
        [],
        ["loss.bbox.nwd"],
        ["loss.bbox.nwd", "head.p2_small_object"],
        ["loss.bbox.nwd", "head.p2_small_object", "neck.fullpad"],
    ]
    assert matrix.nodes[1].parent_id == "baseline"
    assert matrix.nodes[2].added_component == "head.p2_small_object"


def test_evaluates_component_metric_contributions() -> None:
    """Contribution report should compute deltas relative to parent nodes."""
    planner = ComponentContributionPlanner()
    matrix = planner.build_matrix(
        _baseline(),
        ["loss.bbox.nwd", "head.p2_small_object", "neck.fullpad"],
    )
    ids = [node.candidate_config.candidate_id for node in matrix.nodes]
    metrics = {
        ids[0]: {"map": 0.78, "map_small": 0.40, "latency_ms": 10.0},
        ids[1]: {"map": 0.812, "map_small": 0.43, "latency_ms": 10.2},
        ids[2]: {"map": 0.83, "map_small": 0.481, "latency_ms": 12.0},
        ids[3]: {"map": 0.848, "map_small": 0.49, "latency_ms": 11.5},
    }

    report = planner.evaluate(matrix, metrics)
    by_component = {item.component: item for item in report.contributions}

    assert by_component["loss.bbox.nwd"].deltas["map"] == 0.03200000000000003
    assert by_component["head.p2_small_object"].deltas["map_small"] == 0.05099999999999999
    assert by_component["neck.fullpad"].deltas["latency_ms"] == -0.5
    assert by_component["neck.fullpad"].deltas["map"] == 0.018000000000000016
    assert report.missing_metrics == []


def test_marks_missing_metrics() -> None:
    """Contribution evaluation should report candidates missing metrics."""
    planner = ComponentContributionPlanner()
    matrix = planner.build_matrix(_baseline(), ["loss.bbox.nwd"])

    report = planner.evaluate(matrix, {"baseline": {"map": 0.7}})

    assert report.contributions == []
    assert report.missing_metrics == [matrix.nodes[1].candidate_config.candidate_id]

