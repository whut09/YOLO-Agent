"""Utility scorer tests."""

from __future__ import annotations

from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint
from yolo_agent.agents.utility_scorer import UtilityPolicy, UtilityScorer
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def _task() -> TaskSpec:
    return TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
        max_latency_ms=30,
        max_model_size_mb=25,
    )


def test_utility_scorer_outputs_gain_confidence_cost_and_decision() -> None:
    """Utility output should explain why a proposal is worth running."""
    proposal = CandidatePolicy(
        policy_id="small_object_nwd",
        source="rule_engine",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
        expected_improvement={"expected_gain": {"ap_small": 1.2, "map50_95": 0.3}, "confidence": 0.6},
        target_error_facts=[
            {
                "fact_type": "area_metric",
                "subject": "small",
                "area": "small",
                "metric_name": "ap_small",
                "action_candidates": ["small_object_recipe"],
            }
        ],
        constraints=[
            PolicyConstraint(name="estimated_gpu_hours", value=4),
            PolicyConstraint(name="estimated_latency_ms", value=20),
        ],
        risk="low",
    )
    error_fact = ErrorFact(
        run_id="exp001",
        candidate_id="baseline",
        node_id="node_baseline",
        fact_type="area_metric",
        subject="small",
        area="small",
        metric_name="ap_small",
        value=0.2,
        severity="high",
        action_candidates=["small_object_recipe"],
    )

    score = UtilityScorer().score(
        proposal,
        _task(),
        changed_variables={"bbox_loss": ["loss.bbox.nwd"]},
        error_facts=[error_fact],
    )

    assert score.expected_gain == {"ap_small": 1.2, "map50_95": 0.3}
    assert score.aggregate_expected_gain == 1.74
    assert score.confidence == 0.9
    assert score.target_error_relevance == 1.0
    assert score.cost.gpu_hours == 4
    assert score.cost.training_cost == 0.08
    assert score.cost.implementation_risk == 0.05
    assert score.utility > 1.0
    assert score.decision == "run_now"


def test_utility_scorer_marks_missing_evidence() -> None:
    """Missing evidence should produce a needs_evidence utility decision."""
    proposal = CandidatePolicy(
        policy_id="needs_ap_small",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        expected_improvement={"expected_gain": {"ap_small": 0.5}},
        evidence_required=["ap_small"],
    )

    score = UtilityScorer().score(
        proposal,
        _task(),
        changed_variables={},
        missing_evidence=["ap_small"],
    )

    assert score.cost.evidence_gap_penalty > 0
    assert score.decision == "needs_evidence"
    assert "missing_evidence=['ap_small']" in score.reasons


def test_utility_policy_can_be_configured() -> None:
    """Policy weights should be configurable for different optimization goals."""
    proposal = CandidatePolicy(
        policy_id="latency_sensitive",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        expected_improvement={"expected_gain": {"map50_95": 0.1}},
        constraints=[PolicyConstraint(name="estimated_latency_ms", value=29)],
    )
    policy = UtilityPolicy(latency_risk_weight=2.0, run_now_threshold=0.0)

    score = UtilityScorer(policy).score(proposal, _task(), changed_variables={})

    assert score.cost.latency_risk > 0
    assert score.utility < 0
    assert score.decision == "reject"
