from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.optimize_runner import OptimizeRunner
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint
from yolo_agent.agents.utility_scorer import UtilityScorer
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.core.optimization_objective import (
    OptimizationObjective,
    evaluate_optimization_objective,
    parse_optimization_goal,
)
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def _objective(**updates) -> OptimizationObjective:
    values = {
        "baseline_run_id": "coco-yolo26n",
        "baseline_candidate_id": "yolo26n_coco_baseline_full",
        "baseline_protocol_hash": "protocol-1",
    }
    values.update(updates)
    return OptimizationObjective.model_validate(values)


def test_parse_plus_two_map_as_absolute_map50_95_points() -> None:
    objective = parse_optimization_goal(
        "+2map",
        baseline_run_id="run",
        baseline_candidate_id="baseline",
        baseline_protocol_hash="protocol",
    )
    relative = parse_optimization_goal(
        "+2%map50",
        baseline_run_id="run",
        baseline_candidate_id="baseline",
        baseline_protocol_hash="protocol",
    )

    assert objective.primary_metric == "map50_95"
    assert objective.delta_mode == "absolute"
    assert objective.target_absolute_delta == 0.02
    assert objective.required_delta(0.4) == 0.02
    assert relative.primary_metric == "map50"
    assert relative.delta_mode == "relative"
    assert relative.target_relative_delta == 0.02
    assert relative.required_delta(0.5) == 0.01


def test_utility_rejects_candidate_that_breaks_objective_latency_guard() -> None:
    proposal = CandidatePolicy(
        policy_id="slow_candidate",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        expected_improvement={"map50_95": 0.01, "confidence": 0.8},
        constraints=[
            PolicyConstraint(name="estimated_latency_regression", value=0.12),
        ],
        target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
    )
    task = TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["person"],
        primary_metric=MetricPriority(name="map50_95"),
    )

    score = UtilityScorer().score(
        proposal,
        task,
        changed_variables={"box": 8.0},
        optimization_objective=_objective(),
    )

    assert score.objective_progress == 0.5
    assert score.decision == "reject"
    assert any("estimated_latency_regression_exceeds_objective" in item for item in score.reasons)


def test_objective_status_stops_at_pending_confirmation_for_single_seed(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    objective = _objective()
    _write_objective_evidence(root, objective, candidate_seeds=[1], candidate_values=[0.405])

    status = evaluate_optimization_objective(objective, run_root=root, base_run_id="coco-yolo26n")

    assert status.baseline_value == 0.38
    assert status.observed_delta == 0.025000000000000022
    assert status.target_reached is True
    assert status.confirmed is False
    assert status.success is False
    assert status.should_stop is True
    assert status.stop_reason == "target_reached_pending_full_confirmation"


def test_objective_status_requires_trusted_baseline_and_multi_seed_confidence(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    objective = _objective()
    _write_objective_evidence(
        root,
        objective,
        candidate_seeds=[1, 2, 3],
        candidate_values=[0.405, 0.405, 0.405],
    )

    status = evaluate_optimization_objective(objective, run_root=root, base_run_id="coco-yolo26n")

    assert status.baseline_trusted is True
    assert status.candidate_seed_count == 3
    assert status.confirmed is True
    assert status.success is True
    assert status.stop_reason == "objective_confirmed"


def test_objective_status_stops_at_configured_pilot_round_limit(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    objective = _objective(max_pilot_rounds=1)
    _write_objective_evidence(root, objective, candidate_seeds=[1], candidate_values=[0.385])

    status = evaluate_optimization_objective(objective, run_root=root, base_run_id="coco-yolo26n")

    assert status.target_reached is False
    assert status.completed_pilot_rounds == 1
    assert status.should_stop is True
    assert status.stop_reason == "max_pilot_rounds_reached"


def test_optimize_runner_persists_typed_objective(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "images" / "train" / "image.jpg").write_bytes(b"image")
    (dataset / "labels" / "train" / "image.txt").write_text("", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/train\nnames:\n  0: object\n", encoding="utf-8")

    result = OptimizeRunner().run(
        kind="custom",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="objective-run",
        run_root=tmp_path / "runs",
        goal="+2map",
        profile="debug",
        execute=False,
        auto_rounds=0,
    )
    objective_path = result.run_dir / "artifacts" / "optimization_objective.yaml"
    objective = OptimizationObjective.from_yaml(objective_path)
    plan = ExperimentPlan.from_yaml(result.experiment_plan_path)

    assert objective.target_absolute_delta == 0.02
    assert objective.primary_metric == "map50_95"
    assert plan.metadata["optimization_objective_hash"] == objective.objective_hash
    assert plan.nodes[0].command_spec is not None
    assert plan.nodes[0].command_spec.metadata["baseline_protocol_hash"] == objective.baseline_protocol_hash


def _write_objective_evidence(
    root: Path,
    objective: OptimizationObjective,
    *,
    candidate_seeds: list[int],
    candidate_values: list[float],
) -> None:
    store = EvidenceStore(root)
    baseline_node = ExperimentNode(
        node_id="node_baseline",
        candidate_config=CandidateConfig(
            candidate_id=objective.baseline_candidate_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        seed=1,
        command_spec=CommandSpec.ultralytics_train(
            model="yolo26n.pt",
            data="coco.yaml",
            project="runs/ultralytics",
            name="baseline",
            metadata={"training_budget_profile": "baseline_full"},
        ),
    )
    base_dir = store.create_run(objective.baseline_run_id)
    ExperimentPlan(plan_id="baseline", nodes=[baseline_node]).to_yaml(base_dir / "artifacts" / "experiment_plan.yaml")
    store.log_candidate_metrics(
        objective.baseline_run_id,
        objective.baseline_candidate_id,
        baseline_node.node_id,
        {
            objective.primary_metric: 0.38,
            "latency_ms": 10.0,
            "model_size_mb": 5.0,
            "baseline_protocol_hash": objective.baseline_protocol_hash,
        },
        dataset_version="coco2017",
        split="val2017",
        source="test",
    )

    candidate_run_id = f"{objective.baseline_run_id}-r1"
    candidate_nodes: list[ExperimentNode] = []
    for seed, value in zip(candidate_seeds, candidate_values):
        identity = {
            "protocol_hash": objective.baseline_protocol_hash,
            "dataset_manifest_sha256": "dataset-sha",
            "subset_manifest_sha256": f"subset-{seed}",
            "seed": seed,
            "epochs": 100,
            "fidelity": "candidate_full",
            "batch_policy_hash": "batch-policy",
            "ultralytics_version": "9.0.0",
            "imgsz": 640,
            "eval_protocol_hash": "eval-protocol",
        }
        node = ExperimentNode(
            node_id=f"node_candidate_seed_{seed}",
            candidate_config=CandidateConfig(
                candidate_id="candidate-a",
                base_model="yolo26n.pt",
                scale="n",
                framework="ultralytics",
            ),
            data_version="coco2017",
            seed=seed,
            command_spec=CommandSpec.ultralytics_train(
                model="yolo26n.pt",
                data="coco.yaml",
                project="runs/ultralytics",
                name=f"candidate-{seed}",
                metadata={"training_budget_profile": "candidate_full"},
            ),
        )
        candidate_nodes.append(node)
        store.log_candidate_metrics(
            candidate_run_id,
            objective.baseline_candidate_id,
            f"node_matched_control_seed_{seed}",
            {
                objective.primary_metric: 0.38,
                "latency_ms": 10.0,
                "model_size_mb": 5.0,
            },
            dataset_version="coco2017",
            split="val2017",
            source="test",
            evidence_role="baseline_reference",
            **identity,
        )
        store.log_candidate_metrics(
            candidate_run_id,
            "candidate-a",
            node.node_id,
            {
                objective.primary_metric: value,
                "latency_ms": 10.4,
                "model_size_mb": 5.1,
                "baseline_protocol_hash": objective.baseline_protocol_hash,
            },
            dataset_version="coco2017",
            split="val2017",
            source="test",
            **identity,
        )
    candidate_dir = store.create_run(candidate_run_id)
    ExperimentPlan(plan_id="candidate", nodes=candidate_nodes).to_yaml(
        candidate_dir / "artifacts" / "experiment_plan.yaml"
    )
