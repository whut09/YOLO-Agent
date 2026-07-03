"""Loop policy evaluator tests."""

from __future__ import annotations

from yolo_agent.agents.loop_policy_evaluator import BudgetPolicy, LoopPolicyEvaluator
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


def _budget_evaluator(policy: BudgetPolicy) -> LoopPolicyEvaluator:
    return LoopPolicyEvaluator(ComponentRegistry.from_path("configs/components"), budget_policy=policy)


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
    assert "--candidate" not in evaluation.experiment_node.command
    assert "--plan runs/plan.yaml" in evaluation.experiment_node.command
    assert "--data data.yaml" in evaluation.experiment_node.command


def test_loop_policy_uses_run_paths_for_executable_smoke_command() -> None:
    """Loop-created experiment nodes should use the run plan and data YAML."""
    proposal = CandidatePolicy(
        policy_id="nwd_only",
        source="rule_engine",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
    )

    report = _evaluator().evaluate(
        [proposal],
        _task(),
        plan_path="runs/exp001/plan.yaml",
        data_path="datasets/tiny/data.yaml",
    )

    node = report.evaluations[0].experiment_node
    assert node is not None
    assert node.command == (
        "yolo-agent smoke --plan runs/exp001/plan.yaml "
        "--data datasets/tiny/data.yaml --run-id smoke_nwd_only"
    )


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


def test_budget_allocator_defers_candidates_beyond_round_limit() -> None:
    """Budget policy should select only this-round candidates and defer the rest."""
    proposals = [
        CandidatePolicy(
            policy_id=f"policy_{index}",
            source="rule_engine",
            base_model="yolo11n",
            scale="n",
            framework="ultralytics",
            components=["loss.bbox.ciou"],
            priority_hint=float(10 - index),
        )
        for index in range(4)
    ]

    report = _budget_evaluator(BudgetPolicy(max_candidates_per_round=2, exploration_ratio=0.0)).evaluate(
        proposals,
        _task(),
    )

    assert report.budget_allocation is not None
    assert report.budget_allocation.selected == ["policy_0", "policy_1"]
    assert report.budget_allocation.deferred == ["policy_2", "policy_3"]
    assert [candidate.candidate_id for candidate in report.accepted_candidates] == ["policy_0", "policy_1"]
    assert [evaluation.decision for evaluation in report.evaluations] == [
        "accepted",
        "accepted",
        "deferred",
        "deferred",
    ]


def test_budget_allocator_sends_high_risk_over_quota_to_manual_confirmation() -> None:
    """High-risk proposals beyond budget should require human confirmation."""
    high_a = CandidatePolicy(
        policy_id="high_a",
        source="llm",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.ciou"],
        priority_hint=5.0,
        risk="high",
    )
    high_b = CandidatePolicy(
        policy_id="high_b",
        source="llm",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
        priority_hint=4.0,
        risk="high",
    )

    report = _budget_evaluator(
        BudgetPolicy(max_candidates_per_round=3, max_high_risk_candidates=1, exploration_ratio=1.0)
    ).evaluate([high_a, high_b], _task())

    assert [evaluation.decision for evaluation in report.evaluations] == ["accepted", "needs_approval"]
    assert report.evaluations[1].requires_human_confirmation is True
    assert "High-risk candidate budget exhausted" in report.evaluations[1].budget_reason


def test_budget_allocator_requires_approval_for_near_latency_budget() -> None:
    """Manual latency policy should hold near-budget proposals for confirmation."""
    proposal = CandidatePolicy(
        policy_id="near_latency",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.ciou"],
        constraints=[PolicyConstraint(name="estimated_latency_ms", value=25)],
    )

    report = _budget_evaluator(
        BudgetPolicy(latency_budget_policy="manual_confirm", latency_warning_ratio=0.8)
    ).evaluate([proposal], _task())

    assert report.evaluations[0].decision == "needs_approval"
    assert report.evaluations[0].requires_human_confirmation is True
    assert "near max_latency_ms" in report.evaluations[0].budget_reason


def test_budget_allocator_tracks_exploration_exploitation_ratio() -> None:
    """Budget allocation should preserve exploration/exploitation counts when possible."""
    proposals = [
        CandidatePolicy(
            policy_id="exploit_a",
            source="rule_engine",
            base_model="yolo11n",
            scale="n",
            framework="ultralytics",
            components=["loss.bbox.ciou"],
            priority_hint=5.0,
        ),
        CandidatePolicy(
            policy_id="exploit_b",
            source="rule_engine",
            base_model="yolo11n",
            scale="n",
            framework="ultralytics",
            components=["loss.bbox.nwd"],
            priority_hint=4.0,
        ),
        CandidatePolicy(
            policy_id="explore_a",
            source="llm",
            base_model="yolo11n",
            scale="n",
            framework="ultralytics",
            components=["loss.bbox.ciou"],
            priority_hint=3.0,
        ),
    ]

    report = _budget_evaluator(
        BudgetPolicy(max_candidates_per_round=3, exploration_ratio=0.34)
    ).evaluate(proposals, _task())

    assert report.budget_allocation is not None
    assert report.budget_allocation.selected == ["exploit_a", "exploit_b", "explore_a"]
    assert report.budget_allocation.exploration_selected == 1
    assert report.budget_allocation.exploitation_selected == 2
