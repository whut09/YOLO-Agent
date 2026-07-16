"""Loop policy evaluator tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig
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


def _target_error_fact() -> dict[str, object]:
    return {
        "fact_type": "area_metric",
        "subject": "small",
        "area": "small",
        "metric_name": "ap_small",
        "current_value": 0.2,
        "current_severity": "high",
        "action_candidates": ["small_object_recipe", "bbox_loss_recipe"],
    }


def _expected_improvement() -> dict[str, object]:
    return {
        "metric_name": "ap_small",
        "direction": "increase",
        "target": "small",
        "minimum_expected_delta": "pilot_positive_delta",
    }


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
    assert evaluation.experiment_node.command_spec is not None
    assert evaluation.experiment_node.command_spec.command_type == "smoke"
    assert evaluation.experiment_node.command_spec.shell is False
    assert evaluation.experiment_node.command_spec.argv == [
        "yolo-agent",
        "smoke",
        "--plan",
        "runs/plan.yaml",
        "--data",
        "data.yaml",
        "--run-id",
        "smoke_nwd_only",
    ]


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
    assert node.command_spec is not None
    assert node.command_spec.argv == [
        "yolo-agent",
        "smoke",
        "--plan",
        "runs/exp001/plan.yaml",
        "--data",
        "datasets/tiny/data.yaml",
        "--run-id",
        "smoke_nwd_only",
    ]
    assert sorted(node.command_spec.expected_artifacts) == ["generated_models", "smoke_result"]


def test_loop_policy_builds_ultralytics_train_command_when_training_config_is_present() -> None:
    """A loop with training config should materialize train commands, not smoke commands."""
    proposal = CandidatePolicy(
        policy_id="coco_baseline",
        source="rule_engine",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
    )
    training_config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=Path("configs/datasets/coco.yaml"),
        project=Path("runs/ultralytics"),
        imgsz=640,
        epochs=1,
        device="cpu",
    )

    report = _evaluator().evaluate(
        [proposal],
        _task(),
        data_version="coco2017",
        seed=3,
        data_path="E:/datatset/coco.yaml",
        run_id="coco_loop",
        training_config=training_config,
    )

    node = report.evaluations[0].experiment_node
    assert node is not None
    assert node.command_spec is not None
    assert node.command_spec.command_type == "train"
    assert node.command_spec.argv[:3] == ["yolo", "detect", "train"]
    assert "data=E:/datatset/coco.yaml" in node.command_spec.argv
    assert "imgsz=640" in node.command_spec.argv
    assert node.command_spec.metadata["run_id"] == "coco_loop"
    assert node.command_spec.metadata["candidate_id"] == "coco_baseline"
    assert node.command_spec.metadata["node_id"] == "node_coco_baseline"
    assert node.command_spec.metadata["seed"] == 3


def test_pilot_only_mode_blocks_candidate_full_training_profile() -> None:
    """A next-round pilot proposal cannot jump directly to candidate_full."""
    proposal = CandidatePolicy(
        policy_id="pilot_nwd",
        source="rule_engine",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
        target_error_facts=[_target_error_fact()],
        expected_improvement=_expected_improvement(),
    )
    training_config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=Path("configs/datasets/coco.yaml"),
        budget_profile="candidate_full",
    )

    evaluation = _evaluator().evaluate_one(
        proposal,
        _task(),
        training_config=training_config,
        proposal_mode="pilot_only",
        allowed_training_profiles=["debug", "pilot"],
        required_proposal_bindings=["target_error_facts", "expected_improvement"],
    )

    assert evaluation.decision == "rejected"
    assert "candidate_full_blocked_by_pilot_only_proposal_mode" in evaluation.errors


def test_pilot_only_mode_requires_target_error_fact_and_expected_improvement() -> None:
    """Pilot-only proposals must state which error facts they target and expected movement."""
    proposal = CandidatePolicy(
        policy_id="generic_nwd",
        source="rule_engine",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
    )
    training_config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=Path("configs/datasets/coco.yaml"),
        budget_profile="debug",
    )

    evaluation = _evaluator().evaluate_one(
        proposal,
        _task(),
        training_config=training_config,
        proposal_mode="pilot_only",
        allowed_training_profiles=["debug", "pilot"],
        required_proposal_bindings=["target_error_facts", "expected_improvement"],
    )

    assert evaluation.decision == "rejected"
    assert "missing_target_error_facts_binding" in evaluation.errors
    assert "missing_expected_improvement" in evaluation.errors


def test_loop_policy_rejects_imgsz_increase_when_fixed_baseline_imgsz_is_set() -> None:
    """COCO/YOLO26 planning should not increase input size against the baseline."""
    proposal = CandidatePolicy(
        policy_id="higher_imgsz",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        train_overrides={"imgsz": 960},
    )

    evaluation = LoopPolicyEvaluator(
        ComponentRegistry.from_path("configs/components"),
        fixed_imgsz=640,
    ).evaluate_one(proposal, _task())

    assert evaluation.decision == "rejected"
    assert any("imgsz increase is blocked" in error for error in evaluation.errors)


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


def test_loop_policy_accepts_evidence_action_without_training_command() -> None:
    """Evidence acquisition is a first-class non-training action."""
    proposal = CandidatePolicy(
        policy_id="collect_ap_small",
        action_domain="evidence",
        action_id="import_metrics",
        execution_action="import_metrics",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        train_overrides={"evidence_action": "import_metrics", "missing_evidence": ["ap_small", "per_class_ap"]},
        priority_hint=3.0,
    )

    evaluation = _evaluator().evaluate_one(
        proposal,
        _task(),
        plan_path="runs/exp001/experiment_plan.yaml",
        run_id="exp001",
        proposal_mode="pilot_only",
        allowed_training_profiles=["debug", "pilot"],
        required_proposal_bindings=["target_error_facts", "expected_improvement"],
    )

    assert evaluation.decision == "accepted"
    assert evaluation.changed_variables == {"evidence_action": "import_metrics"}
    assert evaluation.experiment_node is not None
    assert evaluation.experiment_node.command_spec is not None
    assert evaluation.experiment_node.command_spec.command_type == "import_metrics"
    assert evaluation.experiment_node.command_spec.metadata["execution_action"] == "import_metrics"
    assert evaluation.experiment_node.command_spec.metadata["action_domain"] == "evidence"


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


def test_fixed_imgsz_override_is_not_an_ablation_change() -> None:
    """The baseline protocol value stays effective without creating a fake variable."""
    proposal = CandidatePolicy(
        policy_id="nwd_fixed_640",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        constraints=[PolicyConstraint(name="fixed_imgsz", value=640, hard=True)],
    )

    evaluation = LoopPolicyEvaluator(
        ComponentRegistry.from_path("configs/components"),
        fixed_imgsz=640,
    ).evaluate_one(proposal, _task())

    assert evaluation.decision == "accepted"
    assert evaluation.fixed_variables["imgsz"] == 640
    assert evaluation.effective_overrides["imgsz"] == 640
    assert evaluation.changed_variables == {"bbox_loss": ["loss.bbox.nwd"]}
    assert evaluation.split_proposals == []
    assert evaluation.experiment_node is not None
    assert evaluation.experiment_node.fixed_variables["imgsz"] == 640
    assert evaluation.experiment_node.effective_overrides["imgsz"] == 640
    assert "imgsz" not in evaluation.experiment_node.changed_variables


def test_imgsz_different_from_baseline_remains_a_changed_variable() -> None:
    proposal = CandidatePolicy(
        policy_id="nwd_unfair_960",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
        train_overrides={"imgsz": 960},
    )

    evaluation = LoopPolicyEvaluator(
        ComponentRegistry.from_path("configs/components"),
        fixed_imgsz=640,
    ).evaluate_one(proposal, _task())

    assert evaluation.decision == "rejected"
    assert evaluation.changed_variables["imgsz"] == 960


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


def test_loop_policy_orders_by_utility_model_when_expected_gain_is_explicit() -> None:
    """Explicit utility should beat legacy priority hints."""
    legacy_high_hint = CandidatePolicy(
        policy_id="legacy_high_hint",
        source="rule_engine",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.ciou"],
        priority_hint=5.0,
    )
    utility_high = CandidatePolicy(
        policy_id="utility_high",
        source="rule_engine",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
        expected_improvement={"expected_gain": {"ap_small": 2.0}, "confidence": 0.8},
        target_error_facts=[_target_error_fact()],
        priority_hint=1.0,
        risk="low",
    )

    report = _evaluator().evaluate([legacy_high_hint, utility_high], _task())

    assert report.evaluations[0].policy_id == "utility_high"
    assert report.evaluations[0].utility_score is not None
    assert report.evaluations[0].utility_score.expected_gain == {"ap_small": 2.0}
    assert report.evaluations[0].utility_score.decision == "run_now"


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
