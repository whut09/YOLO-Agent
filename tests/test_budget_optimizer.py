"""Guarded bandit budget optimizer tests."""

from __future__ import annotations

from yolo_agent.agents.budget_optimizer import BudgetOptimizer, BudgetOptimizerConfig
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluator
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluation
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint
from yolo_agent.agents.successive_halving import HalvingCandidate, SuccessiveHalvingPlanner
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def _task() -> TaskSpec:
    return TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
        max_latency_ms=30,
        max_model_size_mb=20,
    )


def test_budget_optimizer_uses_only_guard_accepted_candidates() -> None:
    """Bandit allocation must not revive rejected or evidence-blocked policies."""
    proposals = [
        CandidatePolicy(
            policy_id="nwd_loss",
            base_model="yolo11n",
            scale="n",
            framework="ultralytics",
            components=["loss.bbox.nwd"],
            priority_hint=5.0,
            expected_improvement={"expected_gain": {"ap_small": 1.0}, "confidence": 0.7},
        ),
        CandidatePolicy(
            policy_id="ciou_loss",
            base_model="yolo11n",
            scale="n",
            framework="ultralytics",
            components=["loss.bbox.ciou"],
            priority_hint=2.0,
        ),
        CandidatePolicy(
            policy_id="too_slow",
            base_model="yolo11n",
            scale="n",
            framework="ultralytics",
            constraints=[PolicyConstraint(name="estimated_latency_ms", value=45)],
            priority_hint=100.0,
        ),
    ]
    evaluation = LoopPolicyEvaluator(ComponentRegistry.from_path("configs/components")).evaluate(
        proposals,
        _task(),
    )

    report = BudgetOptimizer(BudgetOptimizerConfig(max_candidates=1)).optimize(evaluation.evaluations)

    assert report.input_count == 3
    assert report.guarded_count == 2
    assert report.selected_count == 1
    assert report.selected[0].arm.policy_id == "nwd_loss"
    assert "too_slow" in report.rejected_by_guard
    assert {item.arm.policy_id for item in report.deferred} == {"ciou_loss"}
    assert [item.policy_id for item in report.eligible_arms] == ["nwd_loss", "ciou_loss"]
    assert report.allocation_window == 1


def test_budget_optimizer_preserves_every_guarded_arm_beyond_allocation_window() -> None:
    evaluations = [
        _accepted_evaluation(
            f"paper_{index}",
            utility=10.0 - index,
            risk="low",
            components=[f"loss.paper_{index}"],
        )
        for index in range(10)
    ]

    report = BudgetOptimizer(BudgetOptimizerConfig(max_candidates=6)).optimize(evaluations)

    assert report.guarded_count == 10
    assert report.selected_count == 6
    assert len(report.deferred) == 4
    assert len(report.eligible_cohort) == 10
    assert {item.arm.policy_id for item in report.eligible_cohort} == {
        f"paper_{index}" for index in range(10)
    }
    assert [item.rank for item in report.eligible_cohort] == list(range(1, 11))


def test_budget_optimizer_penalizes_high_risk_when_scores_are_close() -> None:
    """Risk penalties should matter after guard approval."""
    evaluations = [
        _accepted_evaluation("high_risk", utility=2.0, risk="high"),
        _accepted_evaluation("low_risk", utility=2.0, risk="low"),
    ]

    report = BudgetOptimizer(BudgetOptimizerConfig(max_candidates=2)).optimize(evaluations)

    assert [item.arm.policy_id for item in report.selected] == ["low_risk", "high_risk"]
    assert report.selected[0].bandit_score > report.selected[1].bandit_score


def test_budget_optimizer_reserves_slots_across_component_families() -> None:
    """A high-scoring family must not consume every bounded pilot slot."""
    evaluations = [
        _accepted_evaluation(f"augmentation_{index}", utility=5.0 - index * 0.1, risk="low")
        for index in range(3)
    ]
    evaluations.extend(
        [
            _accepted_evaluation(
                "quality_loss", utility=3.0, risk="low", components=["loss.quality.correlation"]
            ),
            _accepted_evaluation(
                "feature_neck", utility=2.0, risk="low", components=["neck.multi_scale_fusion"]
            ),
        ]
    )

    report = BudgetOptimizer(BudgetOptimizerConfig(max_candidates=3)).optimize(evaluations)

    assert [item.arm.policy_id for item in report.selected] == [
        "augmentation_0",
        "quality_loss",
        "feature_neck",
    ]
    assert {item.arm.component_family for item in report.selected} == {"model", "loss", "neck"}


def test_successive_halving_builds_pilot_to_full_ladder() -> None:
    """Successive halving should narrow candidates before full budget."""
    candidates = [
        HalvingCandidate(candidate_id="a", node_id="node_a", score=0.9),
        HalvingCandidate(candidate_id="b", node_id="node_b", score=0.8),
        HalvingCandidate(candidate_id="c", node_id="node_c", score=0.3),
        HalvingCandidate(candidate_id="d", node_id="node_d", score=0.2),
    ]

    plan = SuccessiveHalvingPlanner().plan(candidates)

    assert [item.candidate_id for item in plan.assignments_for_stage("pilot_3") if item.decision == "run"] == [
        "a",
        "b",
    ]
    assert {item.candidate_id for item in plan.assignments_for_stage("pilot_3") if item.decision == "eliminate"} == {
        "c",
        "d",
    }
    assert [item.candidate_id for item in plan.assignments_for_stage("pilot_10") if item.decision == "run"] == [
        "a",
        "b",
    ]
    assert plan.promoted_to_full == ["a"]
    assert "candidate_full" in {item.stage_id for item in plan.assignments}


def _accepted_evaluation(
    policy_id: str,
    utility: float,
    risk: str,
    components: list[str] | None = None,
) -> LoopPolicyEvaluation:
    candidate = CandidateConfig(
        candidate_id=policy_id,
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=components or [],
        risk=risk,  # type: ignore[arg-type]
    )
    node = ExperimentNode(
        node_id=f"node_{policy_id}",
        candidate_config=candidate,
        data_version="test",
    )
    return LoopPolicyEvaluation(
        policy_id=policy_id,
        decision="accepted",
        priority=utility,
        candidate_config=candidate,
        experiment_node=node,
    )
