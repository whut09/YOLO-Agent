"""Component contribution planner tests."""

from __future__ import annotations

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.component_contribution import ComponentContributionPlanner
from yolo_agent.core.experiment_graph import Evidence, MetricEvidence


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


def test_evidence_contribution_excludes_inherited_candidate_result() -> None:
    planner = ComponentContributionPlanner()
    matrix = planner.build_matrix(_baseline(), ["loss.bbox.nwd"])
    candidate_id = matrix.nodes[1].candidate_config.candidate_id
    common = {
        "run_id": "run-1",
        "protocol_hash": "protocol-1",
        "dataset_manifest_sha256": "manifest-1",
        "split": "val2017",
        "seed": 1,
        "metric_name": "map50_95",
        "verified": True,
    }
    evidence = Evidence(
        run_id="run-1",
        metric_records=[
            MetricEvidence(
                candidate_id="baseline",
                node_id="baseline",
                origin_run_id="run-1",
                value=0.40,
                **common,
            ),
            MetricEvidence(
                candidate_id=candidate_id,
                node_id=candidate_id,
                origin_run_id="parent",
                evidence_role="baseline_reference",
                inheritance_depth=1,
                source="inherited:parent:test",
                value=0.90,
                **common,
            ),
        ],
    )

    report = planner.evaluate_evidence(
        matrix,
        evidence,
        protocol_hash="protocol-1",
        dataset_manifest_sha256="manifest-1",
        split="val2017",
        seed_by_candidate={"baseline": 1, candidate_id: 1},
    )

    assert report.contributions == []
    assert report.missing_metrics == [candidate_id]


def test_three_seed_contribution_requires_three_paired_controls() -> None:
    planner = ComponentContributionPlanner()
    matrix = planner.build_matrix(_baseline(), ["loss.bbox.nwd"])
    candidate_id = matrix.nodes[1].candidate_config.candidate_id
    records: list[MetricEvidence] = []
    for seed, baseline, candidate in [(1, 0.40, 0.41), (2, 0.39, 0.405), (3, 0.41, 0.42)]:
        identity = {
            "run_id": "run-1",
            "protocol_hash": "protocol-1",
            "dataset_manifest_sha256": "manifest-1",
            "subset_manifest_sha256": f"subset-{seed}",
            "split": "val2017",
            "seed": seed,
            "epochs": 10,
            "fidelity": "pilot_10",
            "batch_policy_hash": "batch-policy",
            "ultralytics_version": "9.0.0",
            "imgsz": 640,
            "eval_protocol_hash": "eval-protocol",
            "metric_name": "map50_95",
            "verified": True,
        }
        records.append(
            MetricEvidence(
                candidate_id="baseline",
                node_id=f"baseline-{seed}",
                evidence_role="baseline_reference",
                value=baseline,
                **identity,
            )
        )
        records.append(
            MetricEvidence(
                candidate_id=candidate_id,
                node_id=candidate_id,
                value=candidate,
                **identity,
            )
        )

    report = planner.evaluate_evidence(
        matrix,
        Evidence(run_id="run-1", metric_records=records),
        protocol_hash="protocol-1",
        dataset_manifest_sha256="manifest-1",
        split="val2017",
        seed_by_candidate={},
    )

    assert report.contributions[0].confidence == "confirmed"
    assert report.contributions[0].paired_seed_count == 3
