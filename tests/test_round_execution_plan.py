"""Canonical round execution plan tests."""

from __future__ import annotations

import pytest

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_queue import ExecutionQueue
from yolo_agent.core.experiment_graph import ExperimentNode, MetricEvidence
from yolo_agent.core.round_execution_plan import build_asha_assignment_plan, build_round_execution_plan


def _node(candidate_id: str, changed: str = "mosaic") -> ExperimentNode:
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name=candidate_id,
        epochs=10,
        imgsz=640,
        batch=48,
    )
    return ExperimentNode(
        node_id=f"node_{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco-v1",
        command_spec=command,
        command=command.display(),
        changed_variables={changed: candidate_id},
    )


def _metric(node: ExperimentNode, value: float) -> MetricEvidence:
    stage = "pilot_10" if node.node_id.endswith("__pilot_10") else "pilot_3"
    return MetricEvidence(
        candidate_id=node.candidate_config.candidate_id,
        node_id=node.node_id,
        metric_name="map50_95",
        value=value,
        verified=True,
        validator="test",
        run_id="round-1",
        origin_run_id="round-1",
        protocol_hash="protocol-640",
        evidence_role=("baseline_reference" if "matched_control" in node.node_id else "current_observation"),
        dataset_manifest_sha256="dataset-sha",
        subset_manifest_sha256="subset-sha",
        seed=node.seed,
        epochs=10 if stage == "pilot_10" else 3,
        fidelity=stage,
        batch_policy_hash="batch-policy",
        ultralytics_version="9.0.0",
        imgsz=640,
        eval_protocol_hash="eval-protocol",
        split="val2017",
    )


def _control() -> ExperimentNode:
    node = _node("baseline_matched_control")
    node.changed_variables = {}
    return node


def _candidate_nodes(plan) -> list[ExperimentNode]:
    return [node for node in plan.execution_nodes if "matched_control" not in node.node_id]


def _stage_evidence(plan, values: list[float], baseline: float = 0.30) -> list[MetricEvidence]:
    control = next(node for node in plan.execution_nodes if "matched_control" in node.node_id)
    return [_metric(control, baseline), *[_metric(node, value) for node, value in zip(_candidate_nodes(plan), values)]]


def test_round_plan_materializes_only_pilot_3() -> None:
    plan = build_round_execution_plan(
        run_id="round-1",
        nodes=[_node("a"), _node("b")],
        baseline_control_node=_control(),
        decision_context_hash="context-hash",
        source_decision_bundle_hash="decision-hash",
    )

    queue = ExecutionQueue.from_round_execution_plan("round-1", plan)

    assert plan.active_stage == "pilot_3"
    assert len(queue.items) == 3
    assert all(item.node_id.endswith("__pilot_3") for item in queue.items)
    assert all("epochs=3" in item.command.argv for item in queue.items)
    assert not any("epochs=10" in item.command.argv for item in queue.items)
    assert queue.metadata["source_authority"] == "RoundExecutionPlan"
    assert queue.metadata["source_round_plan_hash"] == plan.plan_hash()
    assert plan.decision_context_hash == "context-hash"
    assert plan.source_decision_bundle_hash == "decision-hash"


def test_round_plan_does_not_promote_without_complete_evidence() -> None:
    plan = build_round_execution_plan(
        run_id="round-1", nodes=[_node("a"), _node("b")], baseline_control_node=_control()
    )

    advanced = plan.reconcile([_metric(plan.execution_nodes[0], 0.40)])

    assert advanced is False
    assert plan.active_stage == "pilot_3"
    assert plan.status == "awaiting_evidence"
    assert _candidate_nodes(plan)[1].node_id in plan.blocked_reason


def test_round_plan_blocks_candidates_without_current_matched_control() -> None:
    plan = build_round_execution_plan(run_id="round-1", nodes=[_node("a"), _node("b")])

    queue = ExecutionQueue.from_round_execution_plan("round-1", plan)

    assert plan.status == "blocked"
    assert plan.execution_nodes == []
    assert queue.items == []
    assert plan.blocked_reason == "matched baseline control is required"


def test_round_plan_uses_imported_metrics_to_select_pilot_10_survivor() -> None:
    plan = build_round_execution_plan(
        run_id="round-1",
        nodes=[_node("a"), _node("b"), _node("c")],
        baseline_control_node=_control(),
    )
    evidence = _stage_evidence(plan, [0.31, 0.44, 0.39])

    advanced = plan.reconcile(evidence)

    assert advanced is True
    assert plan.active_stage == "pilot_10"
    assert {node.candidate_config.candidate_id for node in _candidate_nodes(plan)} == {"b", "c"}
    assert all(node.node_id.endswith("__pilot_10") for node in plan.execution_nodes)
    assert all("epochs=10" in node.command_spec.argv for node in plan.execution_nodes if node.command_spec)
    eliminated = [item for item in plan.survivor_decisions if not item.promoted]
    assert [item.candidate_id for item in eliminated] == ["a"]


def test_pilot_10_survivor_is_deferred_not_queued_for_full() -> None:
    plan = build_round_execution_plan(
        run_id="round-1",
        nodes=[_node("a"), _node("b")],
        baseline_control_node=_control(),
    )
    assert plan.reconcile(_stage_evidence(plan, [0.4, 0.5]))
    assert plan.reconcile(_stage_evidence(plan, [0.45], baseline=0.35))

    assert plan.active_stage == "full_pending_confirmation"
    assert plan.execution_nodes == []
    full = [item for item in plan.assignments if item.stage_id == "candidate_full_seed_1"]
    assert len(full) == 1
    assert full[0].status == "deferred"
    assert ExecutionQueue.from_round_execution_plan("round-1", plan).items == []


def test_round_ranking_uses_paired_gain_not_absolute_score() -> None:
    plan = build_round_execution_plan(
        run_id="round-1",
        nodes=[_node("a"), _node("b")],
        baseline_control_node=_control(),
    )
    candidates = _candidate_nodes(plan)
    records = [
        _metric(candidates[0], 0.60).model_copy(update={"seed": 1, "subset_manifest_sha256": "s1"}),
        _metric(candidates[1], 0.55).model_copy(update={"seed": 2, "subset_manifest_sha256": "s2"}),
        _metric(_control(), 0.59).model_copy(
            update={
                "node_id": "control-s1",
                "seed": 1,
                "subset_manifest_sha256": "s1",
                "epochs": 3,
                "fidelity": "pilot_3",
                "evidence_role": "baseline_reference",
            }
        ),
        _metric(_control(), 0.45).model_copy(
            update={
                "node_id": "control-s2",
                "seed": 2,
                "subset_manifest_sha256": "s2",
                "epochs": 3,
                "fidelity": "pilot_3",
                "evidence_role": "baseline_reference",
            }
        ),
    ]

    assert plan.reconcile(records) is True
    survivor = next(item for item in plan.survivor_decisions if item.promoted)
    assert survivor.candidate_id == "b"
    assert survivor.paired_delta == pytest.approx(0.10)


def test_multivariable_candidate_cannot_enter_round_queue() -> None:
    invalid = _node("coupled")
    invalid.changed_variables = {"loss": 1, "assigner": 2}

    plan = build_round_execution_plan(run_id="round-1", nodes=[invalid])

    assert plan.execution_nodes == []
    assert plan.status == "blocked"
    assert plan.ablation_nodes[0].valid is False


def test_external_asha_plan_materializes_only_assigned_rung_and_seed() -> None:
    plan = build_asha_assignment_plan(
        run_id="round-2",
        source_node=_node("a"),
        baseline_control_node=_control(),
        stage_id="candidate_full_confirmation",
        epochs=100,
        fraction=1.0,
        seed=44,
        seed_index=3,
    )

    queue = ExecutionQueue.from_round_execution_plan("round-2", plan)

    assert plan.scheduler_mode == "external_asha"
    assert plan.active_stage == "candidate_full_confirmation"
    assert len(queue.items) == 2
    candidate = next(item for item in queue.items if not item.command.metadata.get("matched_baseline_control"))
    control = next(item for item in queue.items if item.command.metadata.get("matched_baseline_control"))
    assert candidate.experiment_node.seed == 44
    assert control.experiment_node.seed == 44
    assert "epochs=100" in candidate.command.argv
    assert "fraction=1.0" in candidate.command.argv
    assert "seed=44" in candidate.command.argv


def test_external_asha_assignment_without_control_is_not_executable() -> None:
    plan = build_asha_assignment_plan(
        run_id="round-2",
        source_node=_node("a"),
        stage_id="pilot_3",
        epochs=3,
        fraction=0.1,
        seed=42,
    )

    assert plan.status == "blocked"
    assert plan.execution_nodes == []
    assert ExecutionQueue.from_round_execution_plan("round-2", plan).items == []


def test_external_asha_pilot_assignment_carries_matched_control() -> None:
    plan = build_asha_assignment_plan(
        run_id="round-3",
        source_node=_node("a"),
        baseline_control_node=_control(),
        stage_id="pilot_10",
        epochs=10,
        fraction=0.1,
        seed=42,
        run_name="round-3-a-pilot-10",
    )

    queue = ExecutionQueue.from_round_execution_plan("round-3", plan)

    assert len(queue.items) == 2
    control = next(item for item in queue.items if item.command.metadata.get("matched_baseline_control"))
    candidate = next(item for item in queue.items if not item.command.metadata.get("matched_baseline_control"))
    assert "epochs=10" in control.command.argv
    assert "fraction=0.1" in control.command.argv
    assert "name=round-3-a-pilot-10_matched_control" in control.command.argv
    assert "name=round-3-a-pilot-10" in candidate.command.argv
