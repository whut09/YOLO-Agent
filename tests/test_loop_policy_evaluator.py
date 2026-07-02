"""Loop policy evaluator tests."""

from __future__ import annotations

from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluator
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.evidence_contract import EvidenceGateResult, EvidenceStatus
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def _task() -> TaskSpec:
    return TaskSpec(
        task_type="detect",
        scene="infrared_small_target",
        class_names=["target"],
        primary_metric=MetricPriority(name="recall"),
        max_latency_ms=30,
        max_model_size_mb=20,
    )


def _evaluator() -> LoopPolicyEvaluator:
    return LoopPolicyEvaluator(ComponentRegistry.from_path("configs/components"))


def _gate_missing(*names: str) -> EvidenceGateResult:
    return EvidenceGateResult(
        ok=False,
        trusted=False,
        statuses=[
            EvidenceStatus(name=name, kind="metric", present=False, message=f"Missing {name}")
            for name in names
        ],
        missing_required=list(names),
        warning="No evidence, do not trust this result.",
    )


def test_loop_policy_accepts_proposal_and_creates_experiment_node() -> None:
    """Accepted proposals should become CandidateConfig and ExperimentNode."""
    proposal = CandidatePolicy(
        policy_id="nwd_only",
        source="rule_engine",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
        expected_effect=["Improve tiny-object recall."],
        risk="medium",
    )

    report = _evaluator().evaluate([proposal], _task(), data_version="dataset_v3", seed=7)

    evaluation = report.evaluations[0]
    assert evaluation.decision == "accepted"
    assert evaluation.candidate_config is not None
    assert evaluation.candidate_config.components == ["loss.bbox.nwd"]
    assert evaluation.experiment_node is not None
    assert evaluation.experiment_node.data_version == "dataset_v3"
    assert evaluation.experiment_node.seed == 7
    assert evaluation.experiment_node.changed_variables == {"bbox_loss": ["loss.bbox.nwd"]}


def test_loop_policy_rejects_deployment_blocked_proposal() -> None:
    """Deployment constraints should reject proposals before candidate creation."""
    proposal = CandidatePolicy(
        policy_id="too_slow",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        constraints=[PolicyConstraint(name="estimated_latency_ms", value=45)],
    )

    evaluation = _evaluator().evaluate_one(proposal, _task())

    assert evaluation.decision == "rejected"
    assert evaluation.candidate_config is None
    assert evaluation.blocked_by_deployment
    assert "exceeds max_latency_ms" in evaluation.blocked_by_deployment[0]


def test_loop_policy_marks_missing_evidence_before_acceptance() -> None:
    """Evidence-dependent proposals should wait for required evidence."""
    proposal = CandidatePolicy(
        policy_id="needs_recall",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["assigner.stal"],
        evidence_required=["recall", "latency_ms"],
    )

    evaluation = _evaluator().evaluate_one(proposal, _task(), _gate_missing("recall"))

    assert evaluation.decision == "needs_evidence"
    assert evaluation.missing_evidence == ["recall"]
    assert evaluation.candidate_config is None


def test_loop_policy_requires_split_for_multi_variable_proposal() -> None:
    """Policies changing multiple primary variables must be split first."""
    proposal = CandidatePolicy(
        policy_id="nwd_p2_imgsz",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd", "head.p2_small_object"],
        train_overrides={"imgsz": 960},
    )

    evaluation = _evaluator().evaluate_one(proposal, _task())

    assert evaluation.decision == "split_required"
    assert evaluation.candidate_config is None
    assert set(evaluation.changed_variables) == {"bbox_loss", "head_component", "imgsz"}
    assert len(evaluation.split_proposals) == 3
    assert {proposal.policy_id for proposal in evaluation.split_proposals} == {
        "nwd_p2_imgsz_bbox_loss",
        "nwd_p2_imgsz_head_component",
        "nwd_p2_imgsz_imgsz",
    }


def test_loop_policy_orders_actions_by_priority() -> None:
    """Higher-priority accepted proposals should sort first."""
    low = CandidatePolicy(
        policy_id="llm_low",
        source="llm",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.ciou"],
        priority_hint=0.5,
    )
    high = CandidatePolicy(
        policy_id="rule_high",
        source="rule_engine",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
        priority_hint=3.0,
    )

    report = _evaluator().evaluate([low, high], _task())

    assert [evaluation.policy_id for evaluation in report.evaluations][:2] == ["rule_high", "llm_low"]
