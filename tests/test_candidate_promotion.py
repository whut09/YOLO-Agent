"""Candidate pilot promotion gate tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.adapters.ultralytics.baseline_acceptance import BaselineAcceptanceResult
from yolo_agent.adapters.ultralytics.candidate_promotion import (
    CandidatePromotionGate,
    CandidatePromotionResult,
)
from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluator
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def test_candidate_promotion_allows_full_when_pilot_improves_target_error_fact(tmp_path: Path) -> None:
    """Pilot promotion should require debug pass, pilot pass, target error improvement, and runtime guard."""
    store = _promotion_store(tmp_path)
    facts = [
        _fact("baseline", "node_baseline", "bottle", 0.20, ["small_object_recipe"], severity="high"),
        _fact("candidate_nwd", "node_candidate_pilot", "bottle", 0.30, ["small_object_recipe"], severity="medium"),
    ]

    result = CandidatePromotionGate().check(
        store.load_run("exp001"),
        facts,
        candidate_id="candidate_nwd",
        target_actions=["small_object_recipe"],
        target_error_facts=[_target_error_fact("bottle")],
    )

    assert result.candidate_full_allowed is True
    assert result.candidate_promotion_rejection_reason == []
    assert len(result.improved_error_facts) == 1
    assert result.target_error_facts == [_target_error_fact("bottle")]
    assert result.runtime_comparisons["runtime_avg_it_per_sec"]["candidate"] == 90.0


def test_candidate_promotion_rejects_improvement_of_unbound_error_fact(tmp_path: Path) -> None:
    """Pilot promotion must improve the bound target fact, not any fact with the same action."""
    store = _promotion_store(tmp_path)
    facts = [
        _fact("baseline", "node_baseline", "bottle", 0.20, ["small_object_recipe"], severity="high"),
        _fact("baseline", "node_baseline", "cup", 0.10, ["small_object_recipe"], severity="high"),
        _fact("candidate_nwd", "node_candidate_pilot", "bottle", 0.19, ["small_object_recipe"], severity="high"),
        _fact("candidate_nwd", "node_candidate_pilot", "cup", 0.30, ["small_object_recipe"], severity="medium"),
    ]

    result = CandidatePromotionGate().check(
        store.load_run("exp001"),
        facts,
        candidate_id="candidate_nwd",
        target_actions=["small_object_recipe"],
        target_error_facts=[_target_error_fact("bottle")],
    )

    assert result.candidate_full_allowed is False
    assert result.improved_error_facts == []
    assert "insufficient_target_error_fact_improvement:0/1" in result.candidate_promotion_rejection_reason


def test_candidate_promotion_rejects_pilot_without_error_fact_improvement(tmp_path: Path) -> None:
    """A pilot that merely ran does not earn full budget without solving a target error."""
    store = _promotion_store(tmp_path)
    facts = [
        _fact("baseline", "node_baseline", "bottle", 0.20, ["small_object_recipe"], severity="high"),
        _fact("candidate_nwd", "node_candidate_pilot", "bottle", 0.19, ["small_object_recipe"], severity="high"),
    ]

    result = CandidatePromotionGate().check(
        store.load_run("exp001"),
        facts,
        candidate_id="candidate_nwd",
        target_actions=["small_object_recipe"],
    )

    assert result.candidate_full_allowed is False
    assert "insufficient_target_error_fact_improvement:0/1" in result.candidate_promotion_rejection_reason


def test_candidate_promotion_rejects_runtime_regression(tmp_path: Path) -> None:
    """A target improvement should not hide an unacceptable runtime regression."""
    store = _promotion_store(tmp_path, candidate_latency=12.5)
    facts = [
        _fact("baseline", "node_baseline", "bottle", 0.20, ["small_object_recipe"], severity="high"),
        _fact("candidate_nwd", "node_candidate_pilot", "bottle", 0.30, ["small_object_recipe"], severity="medium"),
    ]

    result = CandidatePromotionGate().check(
        store.load_run("exp001"),
        facts,
        candidate_id="candidate_nwd",
        target_actions=["small_object_recipe"],
    )

    assert result.candidate_full_allowed is False
    assert any(reason.startswith("latency_regression") for reason in result.candidate_promotion_rejection_reason)


def test_candidate_promotion_cannot_use_inherited_candidate_metrics(tmp_path: Path) -> None:
    """Inherited parent records must not masquerade as this child's pilot result."""
    store = _promotion_store(tmp_path)
    records_path = store._run_dir("exp001") / "metrics_by_node.jsonl"
    current = store.load_run("exp001").metric_records
    records_path.unlink()
    store.log_metric_records(
        "exp001",
        [
            record
            for record in current
            if record.candidate_id == "baseline"
        ]
        + [
            MetricEvidence(
                candidate_id="candidate_nwd",
                node_id="node_candidate_pilot",
                run_id="exp001",
                origin_run_id="parent",
                evidence_role="baseline_reference",
                inheritance_depth=1,
                metric_name="fast_baseline_pilot_passed",
                value=True,
                source="inherited:parent:test",
            )
        ],
    )
    facts = [
        _fact("baseline", "node_baseline", "bottle", 0.20, ["small_object_recipe"], severity="high"),
        _fact("candidate_nwd", "node_candidate_pilot", "bottle", 0.30, ["small_object_recipe"], severity="medium"),
    ]

    result = CandidatePromotionGate().check(
        store.load_run("exp001"),
        facts,
        candidate_id="candidate_nwd",
        target_actions=["small_object_recipe"],
    )

    assert result.candidate_full_allowed is False
    assert "missing_candidate_debug_passed" in result.candidate_promotion_rejection_reason
    assert "missing_candidate_pilot_passed" in result.candidate_promotion_rejection_reason


def test_candidate_full_policy_waits_for_candidate_promotion() -> None:
    """The loop evaluator should block candidate_full without a positive promotion decision."""
    proposal = _proposal()
    config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=Path("configs/datasets/coco.yaml"),
        budget_profile="candidate_full",
    )
    promotion = CandidatePromotionResult(
        candidate_id="candidate_nwd",
        candidate_full_allowed=False,
        candidate_promotion_rejection_reason=["insufficient_target_error_fact_improvement:0/1"],
    )

    evaluation = _evaluator().evaluate_one(
        proposal,
        _task(),
        training_config=config,
        baseline_acceptance=BaselineAcceptanceResult(baseline_trusted=True),
        candidate_promotions={"candidate_nwd": promotion},
        error_facts=[_fact("baseline", "node_baseline", "bottle", 0.20, ["small_object_recipe"], severity="high")],
    )

    assert evaluation.decision == "needs_evidence"
    assert evaluation.missing_evidence == ["candidate_full_allowed"]
    assert "insufficient_target_error_fact_improvement:0/1" in evaluation.warnings


def test_candidate_full_policy_runs_after_baseline_and_candidate_promotion() -> None:
    """A full candidate is planned only after both baseline and pilot-promotion gates pass."""
    proposal = _proposal()
    config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=Path("configs/datasets/coco.yaml"),
        budget_profile="candidate_full",
    )
    promotion = CandidatePromotionResult(candidate_id="candidate_nwd", candidate_full_allowed=True)

    evaluation = _evaluator().evaluate_one(
        proposal,
        _task(),
        training_config=config,
        baseline_acceptance=BaselineAcceptanceResult(baseline_trusted=True),
        candidate_promotions={"candidate_nwd": promotion},
        error_facts=[_fact("baseline", "node_baseline", "bottle", 0.20, ["small_object_recipe"], severity="high")],
    )

    assert evaluation.decision == "accepted"
    assert evaluation.experiment_node is not None
    assert evaluation.experiment_node.command_spec is not None
    assert evaluation.experiment_node.command_spec.metadata["training_budget_profile"] == "candidate_full"


def test_candidate_full_policy_waits_for_target_error_facts() -> None:
    """A full candidate cannot be planned when no targeted COCO error facts exist."""
    proposal = _proposal()
    config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=Path("configs/datasets/coco.yaml"),
        budget_profile="candidate_full",
    )
    promotion = CandidatePromotionResult(candidate_id="candidate_nwd", candidate_full_allowed=True)

    evaluation = _evaluator().evaluate_one(
        proposal,
        _task(),
        training_config=config,
        baseline_acceptance=BaselineAcceptanceResult(baseline_trusted=True),
        candidate_promotions={"candidate_nwd": promotion},
        error_facts=[],
    )

    assert evaluation.decision == "needs_evidence"
    assert evaluation.missing_evidence == ["error_facts"]
    assert "missing_error_facts" in evaluation.warnings


def _promotion_store(
    tmp_path: Path,
    candidate_latency: float = 10.5,
    candidate_it_per_sec: float = 90.0,
) -> EvidenceStore:
    store = EvidenceStore(tmp_path / "runs")
    run_id = "exp001"
    store.create_run(run_id)
    matched = {
        "dataset_manifest_sha256": "manifest-1",
        "subset_manifest_sha256": "subset-1",
        "seed": 1,
        "epochs": 10,
        "fidelity": "pilot_10",
        "batch_policy_hash": "batch-policy",
        "ultralytics_version": "9.0.0",
        "imgsz": 640,
        "eval_protocol_hash": "eval-protocol",
    }
    store.log_candidate_metrics(
        run_id,
        "baseline",
        "node_baseline",
        {
            "latency_ms": 10.0,
            "runtime_avg_it_per_sec": 100.0,
            "runtime_epoch_time_seconds": 100.0,
        },
        dataset_version="coco2017",
        split="runtime",
        source="test",
        evidence_role="baseline_reference",
        **matched,
    )
    store.log_candidate_metrics(
        run_id,
        "candidate_nwd",
        "node_candidate_debug",
        {"fast_baseline_sanity_passed": True},
        dataset_version="coco2017",
        split="runtime",
        source="test",
        **matched,
    )
    store.log_candidate_metrics(
        run_id,
        "candidate_nwd",
        "node_candidate_pilot",
        {
            "fast_baseline_pilot_passed": True,
            "latency_ms": candidate_latency,
            "runtime_avg_it_per_sec": candidate_it_per_sec,
            "runtime_epoch_time_seconds": 110.0,
        },
        dataset_version="coco2017",
        split="runtime",
        source="test",
        **matched,
    )
    return store


def _fact(
    candidate_id: str,
    node_id: str,
    class_name: str,
    value: float,
    actions: list[str],
    severity: str = "medium",
) -> ErrorFact:
    return ErrorFact(
        run_id="exp001",
        candidate_id=candidate_id,
        node_id=node_id,
        dataset_version="coco2017",
        split="val2017",
        fact_type="class_low_ap",
        subject=class_name,
        class_name=class_name,
        metric_name="per_class_ap",
        value=value,
        severity=severity,  # type: ignore[arg-type]
        action_candidates=actions,
        dataset_manifest_sha256="manifest-1",
        subset_manifest_sha256="subset-1",
        seed=1,
        epochs=10,
        fidelity="pilot_10",
        batch_policy_hash="batch-policy",
        ultralytics_version="9.0.0",
        imgsz=640,
        eval_protocol_hash="eval-protocol",
        evidence_role=("baseline_reference" if candidate_id == "baseline" else "current_observation"),
    )


def _target_error_fact(class_name: str) -> dict[str, object]:
    return {
        "fact_type": "class_low_ap",
        "subject": class_name,
        "class_name": class_name,
        "metric_name": "per_class_ap",
        "current_value": 0.2,
        "current_severity": "high",
        "action_candidates": ["small_object_recipe"],
    }


def _proposal() -> CandidatePolicy:
    return CandidatePolicy(
        policy_id="candidate_nwd",
        source="rule_engine",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
        target_error_facts=[
            _target_error_fact("bottle")
        ],
        expected_improvement={
            "metric_name": "per_class_ap",
            "direction": "increase",
            "target": "bottle",
            "minimum_expected_delta": "pilot_positive_delta",
        },
    )


def _task() -> TaskSpec:
    return TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    )


def _evaluator() -> LoopPolicyEvaluator:
    return LoopPolicyEvaluator(ComponentRegistry.from_path("configs/components"), fixed_imgsz=640)
