"""Policy proposal/evaluator boundary tests."""

from __future__ import annotations

from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint, PolicyEvaluator
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def _task() -> TaskSpec:
    return TaskSpec(
        task_type="detect",
        scene="infrared_small_target",
        class_names=["target"],
        primary_metric=MetricPriority(name="recall"),
        device_type="edge_gpu",
        max_latency_ms=40,
        max_model_size_mb=20,
    )


def test_llm_policy_is_proposal_until_evaluator_accepts() -> None:
    """LLM output should become a candidate only after evaluator approval."""
    registry = ComponentRegistry.from_path("configs/components")
    evaluator = PolicyEvaluator(registry)
    policy = CandidatePolicy(
        policy_id="llm_nwd_policy",
        source="llm",
        base_model="yolo11s",
        scale="s",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
        constraints=[
            PolicyConstraint(name="max_latency_ms", value=25),
            PolicyConstraint(name="max_model_size_mb", value=12),
        ],
        expected_effect=["Improve tiny-object recall."],
        risk="medium",
        rationale="Small object miss policy proposal.",
    )

    report = evaluator.evaluate([policy], _task())

    assert len(report.accepted_candidates) == 1
    candidate = report.accepted_candidates[0]
    assert candidate.candidate_id == "llm_nwd_policy"
    assert candidate.components == ["loss.bbox.nwd"]
    assert report.evaluations[0].score > 0


def test_policy_over_hard_constraints_is_rejected() -> None:
    """Hard constraints should stop a policy from becoming a candidate."""
    registry = ComponentRegistry.from_path("configs/components")
    policy = CandidatePolicy(
        policy_id="too_slow",
        source="llm",
        base_model="yolo11s",
        scale="s",
        framework="ultralytics",
        constraints=[PolicyConstraint(name="max_latency_ms", value=60)],
        expected_effect=["Maybe improve accuracy."],
    )

    evaluation = PolicyEvaluator(registry).evaluate_one(policy, _task())

    assert evaluation.accepted is False
    assert evaluation.candidate_config is None
    assert any("exceeds task max_latency_ms" in error for error in evaluation.errors)


def test_policy_unknown_component_is_rejected() -> None:
    """Unknown components should be rejected rather than silently selected."""
    registry = ComponentRegistry.from_path("configs/components")
    policy = CandidatePolicy(
        policy_id="unknown_component",
        source="llm",
        base_model="yolo11s",
        scale="s",
        framework="ultralytics",
        components=["loss.bbox.magic"],
    )

    evaluation = PolicyEvaluator(registry).evaluate_one(policy, _task())

    assert evaluation.accepted is False
    assert evaluation.score == 0
    assert "Unknown component: loss.bbox.magic" in evaluation.errors

