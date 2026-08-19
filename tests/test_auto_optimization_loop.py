"""Auto optimization loop driver tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import yaml
import pytest

from yolo_agent.agents.auto_optimization_loop import (
    AutoOptimizationLoopDriver,
    AutoOptimizationResult,
    AutoRoundResult,
    CandidateExecutionAssessment,
    _empty_diversity_round_reason,
    _empty_recipe_round_reason,
    _apply_paper_method_profile_gate,
    _executed_candidate_effect_delta,
    _adopt_frozen_retry_nodes,
    _is_inheritable_metric_record,
    _paper_progress_context,
    _paper_summary,
    _write_paper_candidate_coverage,
    _planning_error_facts,
    _asha_observation,
    _objective_stop_requires_method_replan,
    _overall_map_method_family_coverage,
    _load_frozen_assignment_retry_queue,
    _enqueue_coco_evidence_recovery,
    _merge_evidence_recovery_loop,
    _mark_paper_candidate_disposition,
    _record_paper_candidate_terminal,
    _candidate_policies_from_recipe,
    _candidate_training_failure_isolated,
    _candidate_training_failure_reason_codes,
    _register_guarded_pilot_trials,
    _reopen_retryable_resource_assignments,
    _repeated_executable_candidates,
    _tried_action_ids,
    _next_round_without_conflicting_queue,
    assess_candidate_execution,
)
from yolo_agent.agents.asha_scheduler import (
    ASHAAssignment,
    ASHAObservation,
    ASHAScheduler,
    ASHAStudyStore,
)
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluation, LoopPolicyEvaluationReport
from yolo_agent.agents.llm_decision_advisor import LLMDecisionAdvisorResult
from yolo_agent.agents.optimize_runner import OptimizeRunner
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.agents.orchestrator import TrainingLoopResult
from yolo_agent.agents.paper_recipe_planner import PaperRecipePlan, PlannedRecipe
from yolo_agent.agents.paper_proposal_ledger import PaperCandidateCoverage
from yolo_agent.agents.policy_stage_runner import _synthetic_executable_pilot_policies
from yolo_agent.certification.code_identity import certification_code_hash
from yolo_agent.certification.schemas import (
    CertificationCapabilityClaim,
    CertificationReport,
    CertificationStage,
)
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.execution_queue import (
    ExecutionQueue,
    ExecutionQueueItem,
    ExecutionQueueStore,
)
from yolo_agent.core.execution_failure import ExecutionFailure
from yolo_agent.core.executor import ExecutionResult
from yolo_agent.core.experiment_graph import Evidence, ExperimentNode, MetricEvidence
from yolo_agent.core.optimization_readiness import OptimizationReadinessGate
from yolo_agent.core.optimization_objective import OptimizationObjectiveStatus
from yolo_agent.core.task_spec import MetricPriority, TaskSpec
from yolo_agent.core.round_execution_plan import (
    RoundExecutionPlan,
    build_asha_assignment_plan,
    build_round_execution_plan,
)
from yolo_agent.core.pilot_evidence import PilotEvidenceCompletenessResult
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.core.run_context import RunContext
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.recipes.schemas import AtomicRecipe
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)
from tests.paired_result_helpers import verified_paired_result
from tests.neck_fixtures import neck_contracts
from tests.maturity_helpers import with_smoke_artifact


def _make_dataset(root: Path) -> Path:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "img1.jpg").write_bytes(b"image")
    (label_dir / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "names:",
                "  0: object",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def _asha_registration_node(
    tmp_path: Path,
    *,
    candidate_id: str,
    search_tier: Literal["method", "scalar_hpo"],
    matched_control: bool = False,
) -> ExperimentNode:
    metadata = {
        "matched_baseline_control": matched_control,
        "matched_pilot_required": not matched_control,
    }
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "data.yaml",
        project=tmp_path / "ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=640,
        batch=16,
        metadata=metadata,
    )
    return ExperimentNode(
        node_id=f"node_{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            action_domain="augmentation" if search_tier == "method" else "training",
            action_id=candidate_id,
            search_tier=search_tier,
            train_overrides={"scale": 0.3} if search_tier == "method" else {"lr0": 0.005},
            target_error_facts=[
                {
                    "fact_type": "false_negative_heavy_class",
                    "subject": "person",
                }
            ],
        ),
        data_version="fixture",
        command_spec=command,
    )


def test_reopens_asha_assignment_blocked_by_recoverable_gpu_failure(tmp_path: Path) -> None:
    scheduler = ASHAScheduler.create("improve-map")
    source = _asha_registration_node(
        tmp_path,
        candidate_id="scale_aug_0_7",
        search_tier="method",
    )
    control = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    trial = scheduler.register_trial(
        trial_id="trial-scale",
        candidate_id="scale_aug_0_7",
        source_run_id="improve-map-r6",
        source_node=source,
        baseline_control_node=control,
    )
    trial.status = "promotion_pending"
    trial.pending_stage = "pilot_10"
    assignment = scheduler.next_assignment()
    assert assignment is not None
    scheduler.mark_running(
        assignment,
        run_id="improve-map-r10",
        node_id="node_scale_aug_0_7__pilot_10",
    )
    scheduler.report(
        trial.trial_id,
        ASHAObservation(
            stage_id="pilot_10",
            node_id="node_scale_aug_0_7__pilot_10",
            seed=42,
            evidence_complete=False,
            failure_reason="queue_blocked",
        ),
    )
    child_dir = tmp_path / "runs" / "improve-map-r10"
    child_dir.mkdir(parents=True)
    item = ExecutionQueueItem.from_node("improve-map-r10", control)
    item.mark_running()
    item.mark_result(
        ExecutionResult(
            run_id="improve-map-r10",
            node_id=item.node_id,
            candidate_id=item.candidate_id,
            status="failed",
            command=item.command,
            stdout="torch.AcceleratorError: CUDA error: out of memory",
        )
    )
    item.status = "needs_resume"
    ExecutionQueue(run_id="improve-map-r10", items=[item]).to_yaml(
        child_dir / "execution_queue.yaml"
    )

    reopened = _reopen_retryable_resource_assignments(
        SimpleNamespace(run_root=tmp_path / "runs"),
        scheduler,
    )

    assert reopened == 1
    assert trial.status == "promotion_pending"
    assert trial.pending_stage == "pilot_10"
    assert trial.observation("pilot_10") is None
    assert assignment.status == "issued"
    retried = scheduler.next_assignment()
    assert retried is not None and retried.assignment_id == assignment.assignment_id


def test_reopens_assignment_bound_to_conflicting_stale_child_queue(tmp_path: Path) -> None:
    scheduler = ASHAScheduler.create("base")
    source = _asha_registration_node(tmp_path, candidate_id="active", search_tier="method")
    control = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    trial = scheduler.register_trial(
        trial_id="trial-active",
        candidate_id="active",
        source_run_id="base-r7",
        source_node=source,
        baseline_control_node=control,
    )
    assignment = scheduler.next_assignment()
    assert assignment is not None
    scheduler.mark_running(assignment, run_id="base-r7", node_id=source.node_id)
    scheduler.report(
        trial.trial_id,
        ASHAObservation(
            stage_id="pilot_3",
            node_id=source.node_id,
            seed=42,
            evidence_complete=False,
            failure_reason="training_failed",
        ),
    )

    child_dir = tmp_path / "runs" / "base-r7"
    child_dir.mkdir(parents=True)
    plan = build_asha_assignment_plan(
        run_id="base-r7",
        source_node=source,
        stage_id="pilot_3",
        epochs=3,
        fraction=0.1,
        seed=42,
        seed_index=1,
        run_name="base-r7-active",
        baseline_control_node=control,
        assignment_id=assignment.assignment_id,
    )
    plan.to_yaml(child_dir / "artifacts" / "round_execution_plan.yaml")
    item = ExecutionQueueItem.from_node("base-r7", source)
    item.status = "running"
    ExecutionQueue(
        run_id="base-r7",
        items=[item],
        metadata={
            "source_round_plan_hash": "old-conflicting-plan",
            "asha_assignment_id": "old:pilot_10:seed1",
        },
    ).to_yaml(child_dir / "execution_queue.yaml")

    reopened = _reopen_retryable_resource_assignments(
        SimpleNamespace(run_root=tmp_path / "runs"), scheduler
    )

    assert reopened == 1
    assert assignment.status == "issued"
    assert assignment.assigned_run_id is None
    assert trial.status == "waiting"
    assert trial.pending_stage == "pilot_3"
    retried = scheduler.next_assignment()
    assert retried is not None and retried.assignment_id == assignment.assignment_id


def test_asha_retry_keeps_frozen_paired_protocol_after_code_change(tmp_path: Path) -> None:
    scheduler = ASHAScheduler.create("improve-map")
    source = _asha_registration_node(
        tmp_path,
        candidate_id="scale_aug_0_7",
        search_tier="method",
    )
    control = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    trial = scheduler.register_trial(
        trial_id="trial-scale",
        candidate_id="scale_aug_0_7",
        source_run_id="improve-map-r6",
        source_node=source,
        baseline_control_node=control,
    )
    trial.status = "promotion_pending"
    trial.pending_stage = "pilot_10"
    assignment = scheduler.next_assignment()
    assert assignment is not None
    old_plan = build_asha_assignment_plan(
        run_id="improve-map-r10",
        source_node=source,
        stage_id="pilot_10",
        epochs=10,
        fraction=0.1,
        seed=42,
        seed_index=1,
        run_name="improve-map-r10-scale",
        baseline_control_node=control,
        assignment_id=assignment.assignment_id,
    )
    for node in old_plan.execution_nodes:
        assert node.command_spec is not None
        node.command_spec.metadata["run_protocol_hash"] = "frozen-protocol"
        node.command = node.command_spec.display()
    old_plan.run_protocol_hash = "frozen-protocol"
    queue = ExecutionQueue.from_round_execution_plan("improve-map-r10", old_plan)
    candidate = next(item for item in queue.items if item.candidate_id != "matched_baseline_control")
    candidate.mark_running()
    candidate.mark_result(
        ExecutionResult(
            run_id="improve-map-r10",
            node_id=candidate.node_id,
            candidate_id=candidate.candidate_id,
            status="completed",
            command=candidate.command,
            metrics={"map50_95": 0.39},
        )
    )
    baseline = next(item for item in queue.items if item.candidate_id == "matched_baseline_control")
    baseline.mark_running()
    baseline.mark_result(
        ExecutionResult(
            run_id="improve-map-r10",
            node_id=baseline.node_id,
            candidate_id=baseline.candidate_id,
            status="failed",
            command=baseline.command,
            stdout="torch.AcceleratorError: CUDA error: out of memory",
        )
    )
    baseline.status = "needs_resume"
    child_dir = tmp_path / "improve-map-r10"
    child_dir.mkdir()
    queue.to_yaml(child_dir / "execution_queue.yaml")
    fresh_plan = build_asha_assignment_plan(
        run_id="improve-map-r10",
        source_node=source,
        stage_id="pilot_10",
        epochs=10,
        fraction=0.1,
        seed=42,
        seed_index=1,
        run_name="improve-map-r10-scale",
        baseline_control_node=control,
        assignment_id=assignment.assignment_id,
    )
    fresh_plan.run_protocol_hash = "new-code-protocol"

    retry_queue = _load_frozen_assignment_retry_queue(child_dir, assignment, fresh_plan)
    assert retry_queue is not None
    adopted = _adopt_frozen_retry_nodes(fresh_plan, retry_queue)

    assert adopted.run_protocol_hash == "frozen-protocol"
    assert {
        node.command_spec.metadata["run_protocol_hash"]
        for node in adopted.execution_nodes
        if node.command_spec is not None
    } == {"frozen-protocol"}


def test_method_cohort_registers_with_asha_while_scalar_hpo_stays_disabled(
    tmp_path: Path,
) -> None:
    context = RunContext(
        run_id="guarded-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    baseline = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    methods = [
        _asha_registration_node(
            tmp_path,
            candidate_id=candidate_id,
            search_tier="method",
        )
        for candidate_id in ("scale_aug_0_3", "copy_paste_0_1", "mixup_0_05")
    ]
    scalar = _asha_registration_node(
        tmp_path,
        candidate_id="lr0_0_005",
        search_tier="scalar_hpo",
    )
    RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[baseline, *methods, scalar],
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    scheduler = ASHAScheduler.create("guarded")

    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        [*methods, scalar],
    )

    assert registered == 3
    assert [trial.candidate_id for trial in scheduler.study.trials] == [
        "scale_aug_0_3",
        "copy_paste_0_1",
        "mixup_0_05",
    ]
    events = EventLog(context.events_path).read()
    assert any("lr0_0_005: scalar HPO is disabled" in event.message for event in events)


def test_overall_map_registers_general_adapter_before_small_object_and_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="overall-map-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    objective = OptimizationObjective(
        goal_description="Improve overall mAP",
        primary_metric="map50_95",
        baseline_run_id="overall-map",
        baseline_candidate_id="baseline",
        baseline_protocol_hash="protocol",
    )
    objective_path = context.artifact_path("optimization_objective.yaml")
    objective.to_yaml(objective_path)
    context.metadata["optimization_objective_path"] = objective_path.as_posix()

    baseline = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    native = _asha_registration_node(
        tmp_path,
        candidate_id="mixup_0_05",
        search_tier="method",
    )
    small = _asha_registration_node(
        tmp_path,
        candidate_id="yolo26_small_object_sampling",
        search_tier="method",
    )
    general = _asha_registration_node(
        tmp_path,
        candidate_id="paper_loss_quality_correlation",
        search_tier="method",
    )
    for node, component_id in (
        (small, "sampling.small_object"),
        (general, "loss.quality.correlation"),
    ):
        node.candidate_config.components = [component_id]
        node.command_spec = node.command_spec.model_copy(
            update={
                "metadata": {
                    **node.command_spec.metadata,
                    "adapter_runtime_entrypoint": (
                        "yolo_agent.adapters.ultralytics.runtime_entrypoint"
                    ),
                }
            }
        )
    RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[baseline, native, small, general],
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            blockers=[],
            report_path=None,
            report_hash="certified",
        ),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.AutomaticRuntimeReadinessGate.evaluate_node",
        lambda self, node: SimpleNamespace(allowed=True),
    )
    scheduler = ASHAScheduler.create("overall-map")

    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        [native, small, general],
    )

    assert registered == 1
    assert [trial.candidate_id for trial in scheduler.study.trials] == [
        "paper_loss_quality_correlation"
    ]
    coverage = yaml.safe_load(
        context.artifact_path("paper_candidate_coverage.yaml").read_text(
            encoding="utf-8"
        )
    )
    records = {item["candidate_id"]: item for item in coverage["records"]}
    assert records["paper_loss_quality_correlation"]["disposition"] == "queued"
    assert records["yolo26_small_object_sampling"]["disposition"] == "incompatible"
    events = EventLog(context.events_path).read()
    reasons = {str(event.details.get("reason")) for event in events}
    assert "small_object_method_out_of_scope_for_overall_map" in reasons
    assert "native_fallback_deferred_for_adapter_methods" in reasons


def test_improve_map_11_registers_full_overall_paper_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="improve-map-11-fixture",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    baseline = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    atomic_components = [
        "loss.hard_negative_classification",
        "sampling.hard_negative_replay",
        "loss.quality.correlation",
        "loss.quality.pseudo_iou",
        "assigner.task_aligned",
        "assigner.optimal_transport",
        "distillation.yolo26_teacher_student",
        "neck.rtmdet_large_kernel",
    ]
    nodes: list[ExperimentNode] = []
    for index, component_id in enumerate(atomic_components):
        node = _asha_registration_node(
            tmp_path,
            candidate_id=f"paper_atomic_{index}",
            search_tier="method",
        )
        node.candidate_config.components = [component_id]
        node.command_spec = node.command_spec.model_copy(
            update={
                "metadata": {
                    **node.command_spec.metadata,
                    "adapter_runtime_entrypoint": "mock.paper.runtime",
                }
            }
        )
        nodes.append(node)
    for candidate_id, components, reason in (
        (
            "paper_coupled_hard_negative",
            [
                "loss.hard_negative_classification",
                "sampling.hard_negative_replay",
            ],
            "hard-negative loss and replay are complementary",
        ),
        (
            "paper_coupled_neck_quality",
            ["neck.rtmdet_large_kernel", "loss.quality.correlation"],
            "neck features and quality alignment share localization evidence",
        ),
    ):
        node = _asha_registration_node(
            tmp_path,
            candidate_id=candidate_id,
            search_tier="method",
        )
        node.candidate_config.components = components
        node.command_spec = node.command_spec.model_copy(
            update={
                "metadata": {
                    **node.command_spec.metadata,
                    "adapter_runtime_entrypoint": "mock.paper.runtime",
                    "coupling_reason": reason,
                    "internal_ablation_plan": "baseline,A,B,A+B",
                    "ablation_combination_id": "A+B",
                }
            }
        )
        nodes.append(node)

    plan = build_round_execution_plan(
        run_id=context.run_id,
        nodes=nodes[:6],
        deferred_candidate_nodes=nodes[6:],
        baseline_control_node=baseline,
        ranks={node.candidate_config.candidate_id: index for index, node in enumerate(nodes)},
    )
    plan.to_yaml(context.artifact_path("round_execution_plan.yaml"))
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            blockers=[],
            report_path=None,
            report_hash="mock-certified",
        ),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.AutomaticRuntimeReadinessGate.evaluate_node",
        lambda self, node: SimpleNamespace(allowed=True),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop._distillation_runtime_blockers",
        lambda *args, **kwargs: [],
    )

    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(scheduler, child, nodes)

    assert len(nodes) == 10
    assert registered == 10
    assert len(scheduler.study.trials) == 10
    summary = context.metadata["asha_registration_summary"]
    assert summary["registered"] == 10
    assert summary["newly_registered"] == 10
    assert summary["deferred"] == 4
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    assert len(coverage.records) == 10
    assert sum(record.disposition == "deferred_budget" for record in coverage.records) == 4


def test_overall_map_marks_small_object_only_registration_as_exhausted(
    tmp_path: Path,
) -> None:
    context = RunContext(
        run_id="overall-map-small-only",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    objective = OptimizationObjective(
        goal_description="Improve overall mAP",
        primary_metric="map50_95",
        baseline_run_id="overall-map-small-only",
        baseline_candidate_id="baseline",
        baseline_protocol_hash="protocol",
    )
    objective_path = context.artifact_path("optimization_objective.yaml")
    objective.to_yaml(objective_path)
    context.metadata["optimization_objective_path"] = objective_path.as_posix()
    baseline = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    small = _asha_registration_node(
        tmp_path,
        candidate_id="yolo26_small_object_p2",
        search_tier="method",
    )
    small.candidate_config.components = ["head.p2_small_object"]
    RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[baseline, small],
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))

    registered = _register_guarded_pilot_trials(
        ASHAScheduler.create(context.run_id),
        child,
        [small],
    )

    assert registered == 0
    assert context.metadata["asha_registration_terminal_exhaustion"] is True
    assert context.metadata["asha_registration_all_candidates_dispositioned"] is True
    assert context.metadata["asha_registration_summary"] == {
        "considered": 1,
        "registered": 0,
        "newly_registered": 0,
        "already_registered": 0,
        "queued": 0,
        "deferred": 0,
        "terminal_rejections": 1,
        "retryable_rejections": 0,
    }


def test_existing_waiting_asha_trial_counts_as_registered_cohort(
    tmp_path: Path,
) -> None:
    context = RunContext(
        run_id="existing-cohort-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    baseline = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    candidate = _asha_registration_node(
        tmp_path,
        candidate_id="paper_quality",
        search_tier="method",
    )
    RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[baseline, candidate],
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    scheduler = ASHAScheduler.create(context.run_id)

    assert _register_guarded_pilot_trials(scheduler, child, [candidate]) == 1
    assert _register_guarded_pilot_trials(scheduler, child, [candidate]) == 1
    assert len(scheduler.study.trials) == 1
    assert context.metadata["asha_registration_summary"] == {
        "considered": 1,
        "registered": 1,
        "newly_registered": 0,
        "already_registered": 1,
        "queued": 1,
        "deferred": 0,
        "terminal_rejections": 0,
        "retryable_rejections": 0,
    }


def test_coupled_ablation_arms_all_register_as_independent_asha_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="coupled-asha-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    baseline = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    arms = [
        _asha_registration_node(
            tmp_path,
            candidate_id=f"paper_recipe_hard_negative__{suffix}",
            search_tier="method",
        )
        for suffix in ("a", "b", "a_b")
    ]
    component_sets = [
        ["loss.hard_negative_classification"],
        ["sampling.hard_negative_replay"],
        ["loss.hard_negative_classification", "sampling.hard_negative_replay"],
    ]
    ablation_plan = [
        {"name": "baseline", "components": []},
        {"name": "A", "components": component_sets[0]},
        {"name": "B", "components": component_sets[1]},
        {"name": "A+B", "components": component_sets[2]},
    ]
    for arm, components, combination_id in zip(
        arms,
        component_sets,
        ("A", "B", "A+B"),
        strict=True,
    ):
        arm.candidate_config.components = components
        arm.candidate_config.action_id = "yolo26_hard_negative_pair"
        arm.command_spec = arm.command_spec.model_copy(
            update={
                "metadata": {
                    **arm.command_spec.metadata,
                    "adapter_runtime_entrypoint": (
                        "yolo_agent.adapters.ultralytics.runtime_entrypoint"
                    ),
                    "component_recipe_id": "yolo26_hard_negative_pair",
                    "component_recipe_version": "v1.0.0",
                    "coupling_reason": "Hard-negative loss and replay are complementary.",
                    "coupling_source_papers": json.dumps(["paper:hard-negative"]),
                    "internal_ablation_plan": json.dumps(ablation_plan),
                    "ablation_combination_id": combination_id,
                }
            }
        )
    RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[baseline, *arms],
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            blockers=[],
            report_path=None,
            report_hash="certified",
        ),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.AutomaticRuntimeReadinessGate.evaluate_node",
        lambda self, node: SimpleNamespace(allowed=True),
    )
    scheduler = ASHAScheduler.create(context.run_id)

    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        arms,
    )

    assert registered == 3
    assert [trial.candidate_id for trial in scheduler.study.trials] == [
        arm.candidate_config.candidate_id for arm in arms
    ]
    assert all(
        trial.baseline_control_node is not None
        and trial.baseline_control_node.candidate_config.candidate_id
        == "matched_baseline_control"
        for trial in scheduler.study.trials
    )
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    assert len(coverage.records) == 3
    assert {record.disposition for record in coverage.records} == {"queued"}
    assert {record.combination_id for record in coverage.records} == {
        "A",
        "B",
        "A+B",
    }
    assert all(record.coupling_reason for record in coverage.records)
    assert all(record.internal_ablation_plan == ablation_plan for record in coverage.records)
    assert all(
        record.combination_fingerprint == record.execution_fingerprint
        for record in coverage.records
    )


def test_asha_registration_recovers_executable_node_missing_from_deferred_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="recover-asha-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    baseline = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    candidate = _asha_registration_node(
        tmp_path,
        candidate_id="paper_quality_recovered",
        search_tier="method",
    )
    candidate.candidate_config.components = ["loss.quality.correlation"]
    candidate.candidate_config.action_id = "yolo26_correlation_auxiliary_loss"
    candidate.command_spec = candidate.command_spec.model_copy(
        update={
            "metadata": {
                **candidate.command_spec.metadata,
                "adapter_runtime_entrypoint": (
                    "yolo_agent.adapters.ultralytics.runtime_entrypoint"
                ),
                "component_recipe_id": "yolo26_correlation_auxiliary_loss",
                "component_recipe_version": "v1.0.0",
            }
        }
    )
    RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[baseline],
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            blockers=[],
            report_path=None,
            report_hash="certified",
        ),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.AutomaticRuntimeReadinessGate.evaluate_node",
        lambda self, node: SimpleNamespace(allowed=True),
    )

    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        [candidate],
    )

    assert registered == 1
    assert scheduler.study.trials[0].source_node.node_id == candidate.node_id
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    assert coverage.records[0].disposition == "queued"


def test_runtime_readiness_failure_isolated_from_other_asha_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="readiness-isolation-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    baseline = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    blocked = _asha_registration_node(
        tmp_path,
        candidate_id="paper_readiness_failed",
        search_tier="method",
    )
    ready = _asha_registration_node(
        tmp_path,
        candidate_id="paper_readiness_ready",
        search_tier="method",
    )
    for node, component_id in (
        (blocked, "loss.quality.correlation"),
        (ready, "loss.quality.pseudo_iou"),
    ):
        node.candidate_config.components = [component_id]
    RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[baseline, blocked, ready],
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            blockers=[],
            report_path=None,
            report_hash="certified",
        ),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.AutomaticRuntimeReadinessGate.evaluate_node",
        lambda self, node: SimpleNamespace(
            allowed=node.candidate_config.candidate_id == "paper_readiness_ready",
            blockers=(
                []
                if node.candidate_config.candidate_id == "paper_readiness_ready"
                else ["adapter_forward_smoke_failed"]
            ),
            artifact_path=None,
        ),
    )

    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        [blocked, ready],
    )

    assert registered == 1
    assert [trial.candidate_id for trial in scheduler.study.trials] == [
        "paper_readiness_ready"
    ]
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    dispositions = {item.candidate_id: item.disposition for item in coverage.records}
    assert dispositions == {
        "paper_readiness_failed": "blocked_runtime",
        "paper_readiness_ready": "queued",
    }


def test_missing_matched_baseline_blocks_only_that_paper_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="baseline-isolation-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    baseline = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    blocked = _asha_registration_node(
        tmp_path,
        candidate_id="paper_without_matching_control",
        search_tier="method",
    )
    blocked.candidate_config.components = ["loss.quality.correlation"]
    blocked.command_spec = blocked.command_spec.model_copy(
        update={
            "metadata": {
                **blocked.command_spec.metadata,
                "paper_id": "paper:missing-control",
                "matched_control_candidate_id": "control-that-does-not-exist",
                "adapter_runtime_entrypoint": "mock.paper.runtime",
            }
        }
    )
    ready = _asha_registration_node(
        tmp_path,
        candidate_id="paper_with_matching_control",
        search_tier="method",
    )
    ready.candidate_config.components = ["loss.quality.pseudo_iou"]
    ready.command_spec = ready.command_spec.model_copy(
        update={
            "metadata": {
                **ready.command_spec.metadata,
                "paper_id": "paper:matching-control",
                "adapter_runtime_entrypoint": "mock.paper.runtime",
            }
        }
    )
    RoundExecutionPlan(
        run_id=context.run_id,
        round_id="round-1",
        deferred_nodes=[baseline, blocked, ready],
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            blockers=[],
            report_path=None,
            report_hash="certified",
        ),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.AutomaticRuntimeReadinessGate.evaluate_node",
        lambda self, node: SimpleNamespace(allowed=True),
    )

    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(scheduler, child, [blocked, ready])

    assert registered == 1
    assert [trial.candidate_id for trial in scheduler.study.trials] == [
        "paper_with_matching_control"
    ]
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    dispositions = {item.candidate_id: item.disposition for item in coverage.records}
    assert dispositions == {
        "paper_without_matching_control": "blocked_runtime",
        "paper_with_matching_control": "queued",
    }
    assert context.metadata["asha_registration_failures_by_paper_id"] == {
        "paper:missing-control": 1,
    }


def test_candidate_adapter_failure_isolated_but_control_failure_is_not(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "isolation-r1"
    candidate = _asha_registration_node(
        tmp_path,
        candidate_id="paper_candidate",
        search_tier="method",
    )
    control = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    candidate_item = ExecutionQueueItem.from_node("isolation-r1", candidate)
    control_item = ExecutionQueueItem.from_node("isolation-r1", control)
    candidate_item.status = "failed"
    control_item.status = "completed"
    queue = ExecutionQueue(
        run_id="isolation-r1",
        items=[candidate_item, control_item],
    )
    store = ExecutionQueueStore(run_dir)
    store.save(queue)

    assert _candidate_training_failure_isolated(
        run_dir,
        candidate_id="paper_candidate",
    )

    control_item.status = "failed"
    store.save(queue)
    assert not _candidate_training_failure_isolated(
        run_dir,
        candidate_id="paper_candidate",
    )


def test_isolated_candidate_failure_updates_paper_coverage_blocker(
    tmp_path: Path,
) -> None:
    context = RunContext(
        run_id="isolation-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    candidate = _asha_registration_node(
        tmp_path,
        candidate_id="paper_candidate",
        search_tier="method",
    )
    candidate.candidate_config.components = ["loss.quality.correlation"]
    candidate.command_spec = candidate.command_spec.model_copy(
        update={
            "metadata": {
                **candidate.command_spec.metadata,
                "component_recipe_id": "quality-correlation",
                "component_recipe_version": "v1",
            }
        }
    )
    _mark_paper_candidate_disposition(
        child,
        candidate,
        disposition="queued",
        reasons=[],
        source_stage="asha_registration",
    )

    candidate_item = ExecutionQueueItem.from_node(context.run_id, candidate)
    candidate_item.mark_running()
    candidate_item.mark_result(
        ExecutionResult(
            run_id=context.run_id,
            node_id=candidate.node_id,
            candidate_id="paper_candidate",
            status="failed",
            command=candidate.command_spec,
            failure=ExecutionFailure(
                kind="adapter_runtime_failed",
                summary="Adapter hook failed.",
                root_cause="Invalid runtime hook.",
            ),
        )
    )
    ExecutionQueue(run_id=context.run_id, items=[candidate_item]).to_yaml(
        context.run_dir / "execution_queue.yaml"
    )

    reasons = _candidate_training_failure_reason_codes(
        context.run_dir,
        candidate_id="paper_candidate",
    )
    _mark_paper_candidate_disposition(
        child,
        candidate,
        disposition="blocked_runtime",
        reasons=reasons,
        source_stage="asha_execution",
    )

    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    assert coverage.records[0].disposition == "blocked_runtime"
    assert coverage.records[0].reason_codes == ["adapter_runtime_failed"]
    assert coverage.records[0].source_stage == "asha_execution"
    assert [event.source_stage for event in coverage.records[0].stage_history] == [
        "asha_registration",
        "asha_execution",
    ]


def test_verified_paired_candidate_writes_already_tested_terminal_state(
    tmp_path: Path,
) -> None:
    context = RunContext(
        run_id="terminal-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    candidate = _asha_registration_node(
        tmp_path,
        candidate_id="paper_candidate",
        search_tier="method",
    )
    candidate.candidate_config.components = ["loss.quality.correlation"]
    _mark_paper_candidate_disposition(
        child,
        candidate,
        disposition="queued",
        reasons=[],
        source_stage="asha_registration",
        asha_trial_id="terminal-r1:paper_candidate",
    )
    scheduler = ASHAScheduler.create(context.run_id)
    trial = scheduler.register_trial(
        trial_id="terminal-r1:paper_candidate",
        candidate_id="paper_candidate",
        source_run_id=context.run_id,
        source_node=candidate,
    )
    paired = verified_paired_result(
        candidate_id="paper_candidate",
        node_id=candidate.node_id,
        delta=0.01,
        dataset_manifest_hash=(
            context.dataset_manifest_sha256
            or context.dataset_version
            or "dataset"
        ),
    )
    observation = ASHAObservation(
        stage_id="pilot_10",
        node_id=candidate.node_id,
        seed=42,
        paired_delta=0.01,
        paired_result_verified=True,
        paired_experiment_result=paired,
        evidence_complete=True,
    )

    _record_paper_candidate_terminal(
        child,
        candidate,
        trial=trial,
        observation=observation,
    )

    record = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    ).records[0]
    assert record.disposition == "already_tested"
    assert record.source_stage == "candidate_completion"
    assert record.asha_trial_id == trial.trial_id


def test_candidate_terminal_rejects_unmatched_evidence_and_records_failure(
    tmp_path: Path,
) -> None:
    context = RunContext(
        run_id="terminal-invalid-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    candidate = _asha_registration_node(
        tmp_path,
        candidate_id="paper_candidate",
        search_tier="method",
    )
    candidate.candidate_config.components = ["loss.quality.correlation"]
    _mark_paper_candidate_disposition(
        child,
        candidate,
        disposition="queued",
        reasons=[],
        source_stage="asha_registration",
        asha_trial_id="terminal-invalid-r1:paper_candidate",
    )
    trial = ASHAScheduler.create(context.run_id).register_trial(
        trial_id="terminal-invalid-r1:paper_candidate",
        candidate_id="paper_candidate",
        source_run_id=context.run_id,
        source_node=candidate,
    )
    paired = verified_paired_result(
        candidate_id="paper_candidate",
        node_id=candidate.node_id,
        delta=0.01,
    )
    unmatched = ASHAObservation(
        stage_id="pilot_3",
        node_id=candidate.node_id,
        seed=42,
        paired_delta=0.01,
        paired_result_verified=True,
        paired_experiment_result=paired,
        evidence_complete=True,
    )
    _record_paper_candidate_terminal(
        child,
        candidate,
        trial=trial,
        observation=unmatched,
    )
    recovery = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    ).records[0]
    assert recovery.disposition == "evidence_recovery"
    assert recovery.required_evidence == [
        "candidate_completed_without_valid_paired_evidence"
    ]

    failed = unmatched.model_copy(
        update={
            "paired_delta": None,
            "paired_result_verified": False,
            "paired_experiment_result": None,
            "evidence_complete": False,
            "failure_reason": "adapter_runtime_failed",
        }
    )
    _record_paper_candidate_terminal(
        child,
        candidate,
        trial=trial,
        observation=failed,
    )
    failure = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    ).records[0]
    assert failure.disposition == "blocked_runtime"
    assert failure.reason_codes == ["adapter_runtime_failed"]
    assert failure.source_stage == "candidate_failure"


def test_execute_mode_stops_before_candidate_search_without_gpu_certification(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="readiness-run",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )

    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=True,
        executor="ultralytics-train",
    )

    assert result.stopped_reason == "optimization_readiness_blocked"
    assert result.rounds == []
    assert result.readiness is not None and result.readiness.ready is False
    assert (base.run_dir / "artifacts" / "optimization_readiness.yaml").is_file()


class _PassingCertificationSuite:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> CertificationReport:
        self.calls.append(kwargs)
        report = CertificationReport(
            certification_id="automatic-mini",
            level="mini_gpu_pilot",
            status="passed",
            model=str(kwargs["model"]),
            data_yaml="mini.yaml",
            device=str(kwargs["device"]),
            protocol_hash="automatic-protocol",
            certified_code_hash=certification_code_hash(),
            executed_recipe_id=str(kwargs["recipe_id"]),
            executed_changed_variable="mosaic",
            stages=[
                CertificationStage(stage_id=stage_id, status="passed")
                for stage_id in {
                    "environment",
                    "train_entrypoint",
                    "debug",
                    "pilot_3_control",
                    "pilot_3_candidates",
                    "post_eval",
                    "error_facts",
                    "paired_delta",
                    "asha_decision",
                    "pilot_10",
                    "catalog_import",
                    "snapshot_creation",
                    "diagnosis_linked_paper_prior",
                    "eligibility_gate",
                    "executable_recipe",
                    "policy_memory_update",
                    "recipe_execution_contract",
                }
            ],
            capability_claims=[
                CertificationCapabilityClaim(
                    capability_id=capability_id,
                    local_reproduction="locally_pilot_reproduced",
                    certification_level="mini_gpu_pilot",
                    recipe_id="reduce_mosaic",
                    snapshot_hash="snapshot",
                    evidence_hash=f"evidence-{capability_id}",
                )
                for capability_id in OptimizationReadinessGate.required_capabilities
            ],
        )
        report.to_yaml(
            Path(kwargs["workdir"]) / "certification_report.yaml",
            exclude_none=True,
            sort_keys=False,
        )
        return report


def test_train_mode_automatically_repairs_missing_gpu_certification(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="auto-certification-run",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )
    suite = _PassingCertificationSuite()

    result = AutoOptimizationLoopDriver(
        auto_certify_gpu=True,
        certification_suite=suite,  # type: ignore[arg-type]
    ).run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=True,
        executor="ultralytics-train",
    )

    assert len(suite.calls) == 1
    assert suite.calls[0]["execute_real_gpu"] is True
    assert result.certification_attempted is True
    assert result.certification_status == "passed"
    assert result.readiness is not None and result.readiness.ready is True
    assert result.stopped_reason == "missing_error_facts"
    event_types = [entry.event_type for entry in EventLog(base.run_dir / "events.jsonl").read()]
    assert "gpu_certification_started" in event_types
    assert "gpu_certification_completed" in event_types


def test_train_mode_stops_cleanly_when_automatic_gpu_certification_fails(tmp_path: Path) -> None:
    class FailingSuite:
        def run(self, **kwargs: object) -> CertificationReport:
            del kwargs
            raise RuntimeError("CUDA test backend unavailable")

    data_yaml = _make_dataset(tmp_path / "dataset")
    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="failed-auto-certification",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )

    result = AutoOptimizationLoopDriver(
        auto_certify_gpu=True,
        certification_suite=FailingSuite(),  # type: ignore[arg-type]
    ).run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=True,
        executor="ultralytics-train",
    )

    assert result.stopped_reason == "optimization_readiness_blocked"
    assert result.certification_status == "failed"
    assert result.certification_failure == "CUDA test backend unavailable"
    assert result.rounds == []
    event_types = [entry.event_type for entry in EventLog(base.run_dir / "events.jsonl").read()]
    assert "gpu_certification_failed" in event_types


def test_verified_inherited_latency_can_continue_across_rounds() -> None:
    """Verified lineage metrics should not disappear after one child generation."""
    assert _is_inheritable_metric_record(
        {
            "metric_name": "latency_ms",
            "value": 44.27,
            "verified": True,
            "source": "inherited:coco-yolo26n-r1:benchmark",
        }
    )


def test_planning_error_facts_fall_back_to_nearest_same_dataset_ancestor(tmp_path: Path) -> None:
    """ASHA registration children may reuse diagnosis context without inheriting evidence."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="planning-facts",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )
    fact = ErrorFact(
        run_id=base.run_id,
        candidate_id="baseline",
        node_id="node_baseline",
        dataset_version="coco2017",
        fact_type="area_metric",
        subject="small",
        severity="high",
        value=0.12,
        metric_name="ap_small",
    )
    ErrorFactStore(tmp_path / "runs").append(base.run_id, [fact])
    base_orchestrator = LoopOrchestrator.from_run_dir(base.run_dir)
    base_orchestrator.next_round()
    next_round = yaml.safe_load(
        (base.run_dir / "artifacts" / "next_round.yaml").read_text(encoding="utf-8")
    )
    assert next_round["coco_error_selection"]["baseline_node_ids"] == ["node_baseline"]
    assert next_round["proposal_mode"] == "pilot_only"
    child = base_orchestrator.fork_next("planning-facts-r1")

    facts, source_run_id = _planning_error_facts(child)

    assert facts == [fact]
    assert source_run_id == base_orchestrator.context.run_id
    assert ErrorFactStore(tmp_path / "runs").read(child.context.run_id) == []


def test_empty_diversity_round_distinguishes_deferral_and_exhaustion(tmp_path: Path) -> None:
    deferred_path = tmp_path / "deferred.yaml"
    exhausted_path = tmp_path / "exhausted.yaml"
    deferred_report = LoopPolicyEvaluationReport(
        evaluations=[
            LoopPolicyEvaluation(
                policy_id="box", decision="deferred",
                diversity_reason="component_family_cooldown:loss:box:last_round=3",
            )
        ]
    )
    exhausted_report = LoopPolicyEvaluationReport(
        evaluations=[
            LoopPolicyEvaluation(
                policy_id="box", decision="deferred",
                diversity_reason="component_family_exhausted",
            )
        ]
    )
    deferred_path.write_text(
        yaml.safe_dump(deferred_report.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    exhausted_path.write_text(
        yaml.safe_dump(exhausted_report.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    assert _empty_diversity_round_reason(deferred_path) == "diversity_deferred"
    assert _empty_diversity_round_reason(exhausted_path) == "family_exhaustion"


def test_empty_recipe_round_stops_when_methods_are_exhausted_and_scalar_hpo_is_disabled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "loop_plan.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "training_recipe_plan": {
                    "policies": [],
                    "family_decisions": [
                        {
                            "family": "scale_augmentation",
                            "decision": "exhausted",
                            "reason": "All configured variants were already tested.",
                        },
                        {
                            "family": "optimizer",
                            "decision": "not_relevant",
                            "reason": "Scalar HPO is disabled; prefer paper and method recipes.",
                        },
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert _empty_recipe_round_reason(path) == "method_candidates_exhausted"


def test_frozen_method_profile_authorizes_only_reusable_adapter_route(
    tmp_path: Path,
) -> None:
    recipe = AtomicRecipe(
        recipe_id="paper-sampling",
        version="v1",
        component_ids=["sampling.small_object"],
        target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
        target_metrics=["ap_small"],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="data.sampling_policy",
        stop_conditions=["no_gain"],
        maturity="smoke_passed",
    )
    registry = RecipeRegistry([recipe])
    plan = PaperRecipePlan(
        selected_recipes=[
            PlannedRecipe(
                recipe_id=recipe.recipe_id,
                version=recipe.version,
                decision="selected",
            )
        ]
    )
    coverage = tmp_path / "paper_method_coverage.yaml"
    coverage.write_text(
        yaml.safe_dump(
            {
                "profiles": [
                    {
                        "profile_id": "profile-1",
                        "paper_id": "paper-1",
                        "canonical_component_ids": ["sampling.small_object"],
                    }
                ],
                "decisions": [
                    {
                        "profile_id": "profile-1",
                        "decision": "reuse_existing_adapter",
                        "canonical_component_ids": ["sampling.small_object"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    gated, bindings = _apply_paper_method_profile_gate(
        plan,
        recipe_registry=registry,
        coverage_path=coverage,
        require_frozen_coverage=True,
    )

    assert [item.recipe_id for item in gated.selected_recipes] == ["paper-sampling"]
    assert bindings == {"paper-sampling": ["paper-1"]}
    assert gated.selected_recipes[0].related_papers == ["paper-1"]
    assert gated.selected_recipes[0].related_method_profile_ids == ["profile-1"]


def test_paper_specific_resolution_authorizes_matching_recipe(
    tmp_path: Path,
) -> None:
    recipe = AtomicRecipe(
        recipe_id="paper-relation-distillation",
        version="v1",
        component_ids=["distillation.relation"],
        target_error_facts=[{"fact_type": "representation_gap"}],
        target_metrics=["map50_95"],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="loss.distillation.relation.weight",
        stop_conditions=["no_gain"],
        maturity="smoke_passed",
    )
    plan = PaperRecipePlan(selected_recipes=[PlannedRecipe(
        recipe_id=recipe.recipe_id,
        version=recipe.version,
        decision="selected",
    )])
    coverage = tmp_path / "paper_method_coverage.yaml"
    coverage.write_text(yaml.safe_dump({
        "profiles": [{
            "profile_id": "profile-relation",
            "paper_id": "paper-relation",
            "canonical_component_ids": ["distillation.relation"],
            "paper_mechanism_resolutions": [{
                "paper_specific_mechanism_id": "relation_distillation",
                "canonical_component_id": "distillation.relation",
                "compatibility": "compatible",
                "required_adapter": "distillation.relation",
                "unresolved_reason": None,
                "execution_fingerprint": "a" * 64,
            }],
        }],
        "decisions": [{
            "profile_id": "profile-relation",
            "decision": "reuse_existing_adapter",
            "canonical_component_ids": ["distillation.relation"],
        }],
    }, sort_keys=False), encoding="utf-8")

    gated, bindings = _apply_paper_method_profile_gate(
        plan,
        recipe_registry=RecipeRegistry([recipe]),
        coverage_path=coverage,
        require_frozen_coverage=True,
    )

    assert bindings == {recipe.recipe_id: ["paper-relation"]}
    planned = gated.selected_recipes[0]
    assert planned.paper_specific_mechanism_ids == ["relation_distillation"]
    assert planned.paper_execution_fingerprints == ["a" * 64]


def test_unresolved_generic_profile_cannot_authorize_recipe(
    tmp_path: Path,
) -> None:
    recipe = AtomicRecipe(
        recipe_id="paper-generic-distillation",
        version="v1",
        component_ids=["distillation.yolo26_teacher_student"],
        target_error_facts=[{"fact_type": "capacity_gap"}],
        target_metrics=["map50_95"],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="loss.distillation.weight",
        stop_conditions=["no_gain"],
        maturity="smoke_passed",
    )
    plan = PaperRecipePlan(selected_recipes=[PlannedRecipe(
        recipe_id=recipe.recipe_id,
        version=recipe.version,
        decision="selected",
    )])
    coverage = tmp_path / "paper_method_coverage.yaml"
    coverage.write_text(yaml.safe_dump({
        "profiles": [{
            "profile_id": "profile-generic",
            "paper_id": "paper-generic",
            "canonical_component_ids": [
                "distillation.yolo26_teacher_student"
            ],
            "paper_mechanism_resolutions": [{
                "paper_specific_mechanism_id": None,
                "canonical_component_id": None,
                "compatibility": "unknown",
                "required_adapter": None,
                "unresolved_reason": "generic mechanism is unresolved",
                "execution_fingerprint": "b" * 64,
            }],
        }],
        "decisions": [{
            "profile_id": "profile-generic",
            "decision": "reuse_existing_adapter",
            "canonical_component_ids": [
                "distillation.yolo26_teacher_student"
            ],
        }],
    }, sort_keys=False), encoding="utf-8")

    gated, bindings = _apply_paper_method_profile_gate(
        plan,
        recipe_registry=RecipeRegistry([recipe]),
        coverage_path=coverage,
        require_frozen_coverage=True,
    )

    assert gated.selected_recipes == []
    assert gated.rejected_recipes[0].decision == "implementation_proposal"
    assert bindings == {}


def test_missing_frozen_method_coverage_rejects_selected_paper_recipe(
    tmp_path: Path,
) -> None:
    recipe = AtomicRecipe(
        recipe_id="paper-sampling",
        version="v1",
        component_ids=["sampling.small_object"],
        target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
        target_metrics=["ap_small"],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="data.sampling_policy",
        stop_conditions=["no_gain"],
        maturity="smoke_passed",
    )
    plan = PaperRecipePlan(
        selected_recipes=[
            PlannedRecipe(
                recipe_id=recipe.recipe_id,
                version=recipe.version,
                decision="selected",
            )
        ]
    )

    gated, bindings = _apply_paper_method_profile_gate(
        plan,
        recipe_registry=RecipeRegistry([recipe]),
        coverage_path=tmp_path / "missing.yaml",
        require_frozen_coverage=True,
    )

    assert gated.selected_recipes == []
    assert gated.rejected_recipes[0].reasons == ["paper_method_coverage_missing"]
    assert bindings == {}


def test_unbound_deferred_paper_recipe_becomes_implementation_request(
    tmp_path: Path,
) -> None:
    recipe = AtomicRecipe(
        recipe_id="paper-quality",
        version="v1",
        component_ids=["loss.quality.correlation"],
        target_error_facts=[{"fact_type": "localization_heavy_class"}],
        target_metrics=["map50_95"],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="loss.quality.correlation.weight",
        stop_conditions=["no_gain"],
        maturity="smoke_passed",
    )
    plan = PaperRecipePlan(
        deferred_recipes=[
            PlannedRecipe(
                recipe_id=recipe.recipe_id,
                version=recipe.version,
                decision="deferred",
            )
        ]
    )

    gated, bindings = _apply_paper_method_profile_gate(
        plan,
        recipe_registry=RecipeRegistry([recipe]),
        coverage_path=tmp_path / "coverage.yaml",
        require_frozen_coverage=True,
    )

    assert gated.selected_recipes == []
    assert gated.deferred_recipes == []
    assert gated.rejected_recipes[0].decision == "implementation_proposal"
    assert gated.rejected_recipes[0].required_adapters == [
        "adapter_for:loss.quality.correlation"
    ]
    assert bindings == {}


def test_local_runtime_recipe_does_not_require_a_paper_method_profile(
    tmp_path: Path,
) -> None:
    recipe = AtomicRecipe(
        recipe_id="yolo26_small_object_sampling",
        version="v1",
        component_ids=["sampling.small_object"],
        target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
        target_metrics=["ap_small"],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="data.sampling_policy",
        stop_conditions=["no_gain"],
        maturity="smoke_passed",
    )
    plan = PaperRecipePlan(
        selected_recipes=[
            PlannedRecipe(
                recipe_id=recipe.recipe_id,
                version=recipe.version,
                decision="selected",
            )
        ]
    )

    gated, bindings = _apply_paper_method_profile_gate(
        plan,
        recipe_registry=RecipeRegistry([recipe]),
        coverage_path=tmp_path / "missing.yaml",
        require_frozen_coverage=True,
    )

    assert [item.recipe_id for item in gated.selected_recipes] == [recipe.recipe_id]
    assert gated.rejected_recipes == []
    assert bindings == {}


def test_candidate_coverage_artifact_preserves_every_planner_disposition(
    tmp_path: Path,
) -> None:
    context = RunContext(
        run_id="coverage-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    recipes = [
        AtomicRecipe(
            recipe_id=recipe_id,
            version="v1",
            component_ids=[component_id],
            target_error_facts=[{"fact_type": "localization_heavy_class"}],
            target_metrics=["map50_95"],
            train_overrides={"imgsz": 640},
            fixed_variables={"imgsz": 640},
            primary_changed_variable=f"{component_id}.weight",
            stop_conditions=["latency_guard", "model_size_guard"],
            maturity="smoke_passed",
        )
        for recipe_id, component_id in (
            ("quality-ready", "loss.quality.correlation"),
            ("distillation-needs-evidence", "distillation.yolo26_teacher_student"),
            ("paper-missing-adapter", "assigner.optimal_transport"),
        )
    ]
    plan = PaperRecipePlan(
        selected_recipes=[
            PlannedRecipe(
                recipe_id=recipes[0].recipe_id,
                version="v1",
                decision="selected",
                related_method_profile_ids=["profile-quality"],
                matched_error_fact_ids=["fact-quality-localization"],
            )
        ],
        deferred_recipes=[
            PlannedRecipe(
                recipe_id=recipes[1].recipe_id,
                version="v1",
                decision="needs_evidence",
                reasons=["missing_teacher_checkpoint_evidence"],
            )
        ],
        rejected_recipes=[
            PlannedRecipe(
                recipe_id=recipes[2].recipe_id,
                version="v1",
                decision="implementation_proposal",
                reasons=["paper_method_profile_not_trainable"],
                required_adapters=["OptimalTransportAssignerAdapter"],
            )
        ],
    )

    _write_paper_candidate_coverage(
        child=child,
        plan=plan,
        recipe_registry=RecipeRegistry(recipes),
        method_profile_bindings={"quality-ready": ["paper-quality"]},
        critic_reports=[
            {"recipe_id": recipe.recipe_id, "accepted": True, "findings": []}
            for recipe in recipes
        ],
    )

    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    assert coverage.disposition_counts == {
        "evidence_recovery": 1,
        "implementation_request": 1,
        "queued": 1,
    }
    records = {item.recipe_id: item for item in coverage.records}
    assert records["distillation-needs-evidence"].required_evidence == [
        "missing_teacher_checkpoint_evidence"
    ]
    assert records["paper-missing-adapter"].required_adapters == [
        "OptimalTransportAssignerAdapter"
    ]
    assert records["quality-ready"].paper_ids == ["paper-quality"]
    assert records["quality-ready"].method_profile_ids == ["profile-quality"]
    assert records["quality-ready"].matched_error_fact_ids == [
        "fact-quality-localization"
    ]
    assert [
        event.source_stage for event in records["quality-ready"].stage_history
    ] == ["paper_recipe_planner", "recipe_critic"]


def test_candidate_coverage_seeds_all_83_papers_and_seals_planner_critic(
    tmp_path: Path,
) -> None:
    fixture = yaml.safe_load(
        Path("tests/fixtures/paper_recipe_coverage.yaml").read_text(encoding="utf-8")
    )
    bindings = fixture["unresolved_bindings"]
    records = sorted(
        [
            PaperExecutionSpec(
                paper_id=item["paper_id"],
                profile_id=f"profile-{index:03d}",
                title=f"Paper {index:03d}",
                source_locations=[f"paper_recipe_coverage.yaml#{item['paper_id']}"],
                canonical_component_ids=[item["paper_specific_mechanism_id"]],
                paper_specific_mechanism_ids=[item["paper_specific_mechanism_id"]],
                required_evidence=["target_error_facts"],
                recipe_ids=[item["recipe_id"]],
                execution_fingerprint=item["execution_fingerprint"],
                current_disposition="evidence_recovery",
                disposition_reason="target error facts are missing",
            )
            for index, item in enumerate(bindings)
        ],
        key=lambda item: item.paper_id,
    )
    inventory = PaperExecutionInventory(
        source_method_coverage_hash="a" * 64,
        all_paper_count=728,
        compatible_paper_count=83,
        exact_reproduction_candidates=0,
        records=records,
    ).with_hash()
    inventory_path = tmp_path / "paper_execution_inventory.yaml"
    inventory.to_yaml(inventory_path, sort_keys=False)
    context = RunContext(
        run_id="coverage-83-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
        dataset_manifest_sha256="dataset-83",
        metadata={"paper_execution_inventory_path": inventory_path.as_posix()},
    )
    child = LoopOrchestrator(context)

    _write_paper_candidate_coverage(
        child=child,
        plan=PaperRecipePlan(),
        recipe_registry=RecipeRegistry([]),
        method_profile_bindings={},
        critic_reports=[],
    )

    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    assert coverage.expected_paper_count == 83
    assert len(coverage.paper_coverage) == 83
    assert len(coverage.current_by_paper) == 83
    assert all(
        [event.boundary for event in paper.stage_history]
        == ["inventory", "planner", "critic"]
        for paper in coverage.paper_coverage
    )


def test_candidate_coverage_records_recipe_missing_from_registry(tmp_path: Path) -> None:
    context = RunContext(
        run_id="missing-recipe-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)

    _write_paper_candidate_coverage(
        child=child,
        plan=PaperRecipePlan(
            candidate_inventory=[
                PlannedRecipe(
                    recipe_id="paper.recipe.missing",
                    version="v1",
                    decision="selected",
                    related_papers=["paper-missing"],
                    related_method_profile_ids=["profile-missing"],
                )
            ]
        ),
        recipe_registry=RecipeRegistry([]),
        method_profile_bindings={},
        critic_reports=[],
    )

    record = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    ).records[0]
    assert record.disposition == "implementation_request"
    assert record.reason_codes == ["recipe_registry_entry_missing"]
    assert record.paper_ids == ["paper-missing"]
    assert record.method_profile_ids == ["profile-missing"]


def test_coupled_recipe_expands_to_every_declared_training_arm(
    tmp_path: Path,
) -> None:
    context = RunContext(
        run_id="coupled-r1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    child = LoopOrchestrator(context)
    recipe = RecipeRegistry.from_path(
        Path("configs/recipes/yolo26_paper_coupled.yaml")
    ).get("yolo26_hard_negative_pair")
    assert recipe is not None
    facts = [
        ErrorFact(
            run_id=context.run_id,
            candidate_id="baseline",
            node_id="node-baseline",
            fact_type="background_false_positive_class",
            subject="person",
        )
    ]

    policies = _candidate_policies_from_recipe(
        child,
        recipe,
        facts,
        utility=9.0,
    )

    assert [policy.policy_id.rsplit("__", 1)[-1] for policy in policies] == [
        "a",
        "b",
        "a_b",
    ]
    assert [policy.components for policy in policies] == [
        ["loss.hard_negative_classification"],
        ["sampling.hard_negative_replay"],
        ["loss.hard_negative_classification", "sampling.hard_negative_replay"],
    ]
    assert policies[0].train_overrides["loss.hard_negative_classification.weight"] == 0.05
    assert policies[1].train_overrides["data.hard_negative_replay"] == "enabled"
    assert any(
        constraint.name == "coupled_recipe" and constraint.value is True
        for constraint in policies[2].constraints
    )

    _write_paper_candidate_coverage(
        child=child,
        plan=PaperRecipePlan(
            selected_recipes=[
                PlannedRecipe(
                    recipe_id=recipe.recipe_id,
                    version=recipe.version,
                    decision="selected",
                )
            ]
        ),
        recipe_registry=RecipeRegistry([recipe]),
        method_profile_bindings={},
        critic_reports=[
            {"recipe_id": recipe.recipe_id, "accepted": True, "findings": []}
        ],
    )
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    assert sorted(record.combination_id for record in coverage.records if record.combination_id) == [
        "A",
        "A+B",
        "B",
        "baseline",
    ]
    baseline = next(
        record for record in coverage.records if record.combination_id == "baseline"
    )
    assert baseline.disposition == "already_tested"
    assert baseline.coupling_reason == recipe.coupling_reason
    assert baseline.internal_ablation_plan == recipe.internal_ablation_plan
    assert baseline.combination_fingerprint == baseline.execution_fingerprint
    assert len({record.execution_fingerprint for record in coverage.records}) == 4


def test_paper_progress_does_not_attribute_unrelated_candidate_to_first_recipe(tmp_path: Path) -> None:
    path = tmp_path / "paper_recipe_plan.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "rule_plan": {
                    "selected_recipes": [
                        {"recipe_id": "yolo26_small_object_p2", "primary_changed_variable": "head"}
                    ]
                },
                "llm_proposal": {"primary_problem": "AP_small low"},
                "executable_pilot_policies": [
                    {
                        "policy_id": "paper_recipe_p2",
                        "action_id": "yolo26_small_object_p2",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    optimizer = CandidateExecutionAssessment(
        policy_id="next_training_optimizer_adamw",
        candidate_id="next_training_optimizer_adamw",
        execution_class="executable",
        action_id="optimizer_adamw",
    )
    paper = CandidateExecutionAssessment(
        policy_id="paper_recipe_p2",
        candidate_id="yolo26_small_object_p2",
        execution_class="executable",
        action_id="yolo26_small_object_p2",
    )

    assert _paper_progress_context(path, assessment=optimizer) == {
        "diagnosis": "AP_small low",
        "recipe": "",
        "changed_variable": "",
    }
    assert _paper_progress_context(path, assessment=paper) == {
        "diagnosis": "AP_small low",
        "recipe": "yolo26_small_object_p2",
        "changed_variable": "head",
    }


def test_paper_summary_only_adopts_assessed_executable_paper_recipes(tmp_path: Path) -> None:
    paper_plan = tmp_path / "paper_recipe_plan.yaml"
    paper_plan.write_text(
        yaml.safe_dump(
            {
                "recipe_critic_reports": [],
                "executable_pilot_policies": [
                    {"policy_id": "paper_recipe_p2", "action_id": "yolo26_small_object_p2"}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    optimizer_round = AutoRoundResult(
        round_index=1,
        run_id="run-r1",
        run_dir=tmp_path / "run-r1",
        parent_run_id="run",
        auto_round_summary_path=tmp_path / "run-r1" / "summary.yaml",
        paper_recipe_plan_path=paper_plan,
        candidate_assessments=[
            CandidateExecutionAssessment(
                policy_id="optimizer_adamw",
                candidate_id="optimizer_adamw",
                execution_class="executable",
                action_id="optimizer_adamw",
            )
        ],
    )
    result = AutoOptimizationResult(
        base_run_id="run",
        base_run_dir=tmp_path / "run",
        requested_rounds=1,
        executed=False,
        rounds=[optimizer_round],
        summary_path=tmp_path / "summary.md",
        full_candidate_recommendations_path=tmp_path / "recommendations.yaml",
    )

    assert _paper_summary(result)["adopted"] == []

    result.rounds[0].candidate_assessments = [
        CandidateExecutionAssessment(
            policy_id="paper_recipe_p2",
            candidate_id="yolo26_small_object_p2",
            execution_class="executable",
            action_id="yolo26_small_object_p2",
        )
    ]
    assert _paper_summary(result)["adopted"] == ["yolo26_small_object_p2"]


def test_repetition_guard_counts_only_duplicate_executions_of_the_same_asha_rung(
    tmp_path: Path,
) -> None:
    def round_result(index: int, stage: str, *, trained: bool) -> AutoRoundResult:
        return AutoRoundResult(
            round_index=index,
            run_id=f"run-r{index}",
            run_dir=tmp_path / f"run-r{index}",
            parent_run_id="run",
            auto_round_summary_path=tmp_path / f"run-r{index}" / "summary.yaml",
            training_loop=(
                {
                    "run_id": f"run-r{index}",
                    "profile": "pilot",
                    "executor": "ultralytics-train",
                    "max_steps": 1,
                    "completed": True,
                }
                if trained
                else None
            ),
            candidate_assessments=[
                CandidateExecutionAssessment(
                    policy_id=f"candidate:{stage}",
                    candidate_id="candidate",
                    execution_class="executable",
                    action_id=stage,
                )
            ],
        )

    result = AutoOptimizationResult(
        base_run_id="run",
        base_run_dir=tmp_path / "run",
        requested_rounds=4,
        executed=True,
        rounds=[
            round_result(1, "recipe_registration", trained=False),
            round_result(2, "pilot_3", trained=True),
            round_result(3, "pilot_10", trained=True),
        ],
        summary_path=tmp_path / "summary.md",
        full_candidate_recommendations_path=tmp_path / "recommendations.yaml",
    )

    assert _repeated_executable_candidates(result) == []

    result.rounds.append(round_result(4, "pilot_3", trained=True))
    repeated = _repeated_executable_candidates(result)
    assert len(repeated) == 1
    assert repeated[0]["stage"] == "pilot_3"
    assert repeated[0]["count"] == 2


def test_diversity_deferred_action_advances_future_recipe_variant(tmp_path: Path) -> None:
    artifacts = tmp_path / "base-r1" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "auto_round_summary.yaml").write_text(
        yaml.safe_dump({"candidate_assessments": []}), encoding="utf-8"
    )
    candidate = CandidateConfig(
        candidate_id="box_8_25", base_model="yolo26n.pt", scale="n",
        framework="ultralytics", action_id="tune_box_loss_gain_8_25",
    )
    report = LoopPolicyEvaluationReport(
        evaluations=[
            LoopPolicyEvaluation(
                policy_id="box_8_25", decision="deferred", candidate_config=candidate,
                diversity_reason="minimum_semantic_distance:0.03<0.15",
            )
        ]
    )
    (artifacts / "policy_evaluation.yaml").write_text(
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    assert _tried_action_ids(tmp_path, "base") == ["tune_box_loss_gain_8_25"]


def test_budget_deferred_action_remains_untried_for_next_method_batch(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "base-r1" / "artifacts"
    artifacts.mkdir(parents=True)
    candidate = CandidateConfig(
        candidate_id="paper_neck",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=["neck.multi_scale_fusion"],
        action_id="paper.neck.multi_scale_fusion",
    )
    report = LoopPolicyEvaluationReport(
        evaluations=[
            LoopPolicyEvaluation(
                policy_id="paper_neck",
                decision="deferred",
                candidate_config=candidate,
                diversity_reason="diversity_guards_passed",
                budget_reason="High-risk pilot quota exhausted; deferred to a later automatic round.",
            )
        ]
    )
    (artifacts / "auto_round_summary.yaml").write_text(
        yaml.safe_dump({"candidate_assessments": []}), encoding="utf-8"
    )
    (artifacts / "policy_evaluation.yaml").write_text(
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    assert _tried_action_ids(tmp_path, "base") == []


def test_patience_waits_for_registered_and_followup_method_batches() -> None:
    status = OptimizationObjectiveStatus(
        objective_hash="objective",
        primary_metric="map50_95",
        no_improvement_rounds=4,
        should_stop=True,
        stop_reason="no_improvement_patience_reached",
    )
    assignment = ASHAAssignment(
        trial_id="trial",
        candidate_id="candidate",
        stage_id="pilot_3",
        seed=42,
        epochs=3,
        fraction=0.1,
        reason="test",
    )
    assignment_round = AutoRoundResult(
        round_index=2,
        run_id="run-r2",
        run_dir=Path("runs/run-r2"),
        parent_run_id="run-r1",
        status="completed",
        stop_reason="asha_assignment_completed",
        auto_round_summary_path=Path("runs/run-r2/artifacts/summary.yaml"),
    )
    registration_round = assignment_round.model_copy(
        update={"stop_reason": "asha_candidates_registered"}
    )
    empty_round = assignment_round.model_copy(
        update={"stop_reason": "no_new_asha_trials", "status": "blocked"}
    )

    assert _objective_stop_requires_method_replan(
        status, asha_assignment=assignment, round_result=assignment_round
    )
    assert _objective_stop_requires_method_replan(
        status, asha_assignment=None, round_result=registration_round
    )
    assert not _objective_stop_requires_method_replan(
        status, asha_assignment=None, round_result=empty_round
    )


def test_overall_map_family_coverage_excludes_native_augmentation(
    tmp_path: Path,
) -> None:
    scheduler = ASHAScheduler.create("overall-family-coverage")
    family_components = [
        "loss.hard_negative_classification",
        "loss.quality.correlation",
        "assigner.task_aligned",
        "distillation.yolo26_teacher_student",
        "neck.rtmdet_large_kernel",
    ]
    for index, component_id in enumerate(family_components):
        node = _asha_registration_node(
            tmp_path,
            candidate_id=f"paper-family-{index}",
            search_tier="method",
        )
        node.candidate_config.components = [component_id]
        scheduler.register_trial(
            trial_id=f"trial-{index}",
            candidate_id=node.candidate_config.candidate_id,
            source_run_id="overall-family-coverage-r1",
            source_node=node,
            target_error_facts=[{"fact_type": "localization_error"}],
        )

    assert _overall_map_method_family_coverage(scheduler) == {
        "hard_negative",
        "quality",
        "assignment",
        "distillation",
        "neck",
    }

    native_scheduler = ASHAScheduler.create("native-only")
    native = _asha_registration_node(
        tmp_path,
        candidate_id="mixup_0_05",
        search_tier="method",
    )
    native_scheduler.register_trial(
        trial_id="native-mixup",
        candidate_id="mixup_0_05",
        source_run_id="native-only-r1",
        source_node=native,
        target_error_facts=[{"fact_type": "localization_error"}],
    )

    assert _overall_map_method_family_coverage(native_scheduler) == set()


def test_active_asha_assignment_skips_child_with_conflicting_queue(tmp_path: Path) -> None:
    queue_dir = tmp_path / "runs" / "base-r7"
    queue_dir.mkdir(parents=True)
    queue = ExecutionQueue(
        run_id="base-r7",
        items=[],
        metadata={
            "asha_assignment_id": "old:pilot_10:seed1",
            "source_round_stage": "pilot_10",
        },
    )
    old_item = _asha_registration_node(tmp_path, candidate_id="old", search_tier="method")
    item = ExecutionQueueItem.from_node("base-r7", old_item)
    item.status = "running"
    queue.items = [item]
    queue.to_yaml(queue_dir / "execution_queue.yaml")

    assignment = ASHAAssignment(
        trial_id="base:active",
        candidate_id="active",
        stage_id="pilot_3",
        seed=42,
        epochs=3,
        fraction=0.1,
        reason="active method",
    )

    assert _next_round_without_conflicting_queue(
        tmp_path / "runs", "base", 7, assignment
    ) == 8


def test_executed_candidate_effect_uses_exact_paired_control() -> None:
    identity = {
        "run_id": "run-r1", "origin_run_id": "run-r1",
        "protocol_hash": "protocol-640", "dataset_manifest_sha256": "dataset",
        "subset_manifest_sha256": "subset", "split": "val", "seed": 42,
        "epochs": 3, "fidelity": "pilot_3", "batch_policy_hash": "batch",
        "ultralytics_version": "9.0", "imgsz": 640, "eval_protocol_hash": "eval",
        "metric_name": "map50_95", "verified": True,
    }
    evidence = Evidence(
        run_id="run-r1",
        metric_records=[
            MetricEvidence(
                candidate_id="baseline", node_id="control", value=0.30,
                evidence_role="baseline_reference", **identity,
            ),
            MetricEvidence(candidate_id="candidate", node_id="node_candidate", value=0.315, **identity),
        ],
    )
    assert _executed_candidate_effect_delta(
        evidence, candidate_id="candidate", node_id="node_candidate"
    ) == 0.015000000000000013


def test_asha_observation_uses_frozen_ap_small_objective(tmp_path: Path) -> None:
    run_id = "small-object-r1"
    run_root = tmp_path / "runs"
    artifact_dir = run_root / run_id / "artifacts"
    artifact_dir.mkdir(parents=True)
    objective_path = artifact_dir / "optimization_objective.yaml"
    OptimizationObjective(
        primary_metric="ap_small",
        baseline_run_id=run_id,
        baseline_candidate_id="matched_baseline_control",
        baseline_protocol_hash="protocol-640",
    ).to_yaml(objective_path)
    store = EvidenceStore(run_root)
    protocol = {
        "dataset_manifest_sha256": "dataset",
        "subset_manifest_sha256": "subset",
        "protocol_hash": "protocol-640",
        "eval_protocol_hash": "eval",
        "seed": 42,
        "epochs": 3,
        "fidelity": "pilot_3",
        "batch_policy_hash": "batch",
        "ultralytics_version": "9.0",
        "imgsz": 640,
    }
    store.log_candidate_metrics(
        run_id,
        "matched_baseline_control",
        "node_baseline",
        {
            "ap_small": 0.20,
            "map50_95": 0.40,
            "latency_ms": 10.0,
            "model_size_mb": 5.0,
        },
        evidence_role="baseline_reference",
        **protocol,
    )
    store.log_candidate_metrics(
        run_id,
        "small_object_candidate",
        "node_candidate",
        {
            "ap_small": 0.23,
            "map50_95": 0.39,
            "latency_ms": 10.1,
            "model_size_mb": 5.0,
        },
        **protocol,
    )
    context = SimpleNamespace(
        run_id=run_id,
        run_root=run_root,
        metadata={"optimization_objective_path": str(objective_path)},
        artifact_path=lambda name: artifact_dir / name,
    )
    child = SimpleNamespace(context=context, evidence_store=store)
    node = ExperimentNode(
        node_id="node_candidate",
        candidate_config=CandidateConfig(
            candidate_id="small_object_candidate",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco",
    )
    assignment = ASHAAssignment(
        trial_id="small-object",
        candidate_id="small_object_candidate",
        stage_id="pilot_3",
        seed=42,
        epochs=3,
        fraction=0.1,
        reason="test",
    )

    observation = _asha_observation(
        child,
        node=node,
        assignment=assignment,
        target_error_facts=[],
    )

    assert observation.evidence_complete is True
    assert observation.paired_delta == pytest.approx(0.03)
    assert observation.paired_experiment_result is not None
    deltas = observation.paired_experiment_result.metric_deltas
    assert deltas["ap_small"].effect_delta == pytest.approx(0.03)
    assert deltas["map50_95"].effect_delta == pytest.approx(-0.01)


def test_synthetic_pilot_uses_next_untried_parameter_variant(tmp_path: Path) -> None:
    """A tried action should advance its finite parameter ladder instead of disappearing."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="ladder-run",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )
    context = LoopOrchestrator.from_run_dir(result.run_dir).context
    policies = _synthetic_executable_pilot_policies(
        context,
        focus_items=[
            {
                "fact_type": "localization_heavy_class",
                "class_name": "person",
                "action_candidates": ["bbox_loss_recipe"],
            }
        ],
        allowed_actions={"bbox_loss_recipe", "increase_box_loss_gain"},
        tried_actions={"increase_box_loss_gain"},
        existing_policy_ids=set(),
    )

    by_id = {policy.policy_id: policy for policy in policies}
    assert "next_training_tune_box_loss_gain_8_25" in by_id
    assert by_id["next_training_tune_box_loss_gain_8_25"].train_overrides["box"] == 8.25


def test_assess_candidate_execution_splits_real_and_metadata_only_candidates(tmp_path: Path) -> None:
    """The auto loop must not fake-train metadata-only component proposals."""
    executable_candidate = CandidateConfig(
        candidate_id="safe_optimizer",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        action_domain="training",
        action_id="optimizer",
        train_overrides={"optimizer": "AdamW"},
    )
    executable_node = ExperimentNode(
        node_id="node_safe_optimizer",
        candidate_config=executable_candidate,
        data_version="coco2017",
    )
    executable_node.command_spec = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "data.yaml",
        project=tmp_path / "ultralytics",
        name="safe",
    )

    adapter_candidate = CandidateConfig(
        candidate_id="nwd_loss",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
    )
    adapter_node = ExperimentNode(
        node_id="node_nwd_loss",
        candidate_config=adapter_candidate,
        data_version="coco2017",
    )
    adapter_node.command_spec = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "data.yaml",
        project=tmp_path / "ultralytics",
        name="nwd",
    )

    advisory_candidate = CandidateConfig(
        candidate_id="postprocess",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        action_domain="postprocess",
        action_id="soft_nms",
        train_overrides={"postprocess": ["soft_nms"]},
    )
    advisory_node = ExperimentNode(
        node_id="node_postprocess",
        candidate_config=advisory_candidate,
        data_version="coco2017",
    )
    advisory_node.command_spec = CommandSpec.smoke(
        plan_path=tmp_path / "plan.yaml",
        data_path=tmp_path / "data.yaml",
        run_id="smoke",
    )

    report = LoopPolicyEvaluationReport(
        evaluations=[
            LoopPolicyEvaluation(
                policy_id="safe_optimizer",
                decision="accepted",
                candidate_config=executable_candidate,
                experiment_node=executable_node,
            ),
            LoopPolicyEvaluation(
                policy_id="nwd_loss",
                decision="accepted",
                candidate_config=adapter_candidate,
                experiment_node=adapter_node,
            ),
            LoopPolicyEvaluation(
                policy_id="postprocess",
                decision="accepted",
                candidate_config=advisory_candidate,
                experiment_node=advisory_node,
            ),
        ]
    )

    by_policy = {item.policy_id: item for item in assess_candidate_execution(report)}

    assert by_policy["safe_optimizer"].execution_class == "executable"
    assert by_policy["nwd_loss"].execution_class == "adapter_required"
    assert "component_adapter:loss.bbox.nwd" in by_policy["nwd_loss"].required_adapters
    assert by_policy["postprocess"].execution_class == "recommendation_only"


def test_selected_neck_recipe_reaches_executable_runtime_node(tmp_path: Path) -> None:
    contract = with_smoke_artifact(
        neck_contracts()["neck.gold_gather_distribute"].model_copy(
            update={"maturity": "smoke_passed"}
        )
    )
    candidate = CandidateConfig(
        candidate_id="paper_recipe_gold_neck",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=[contract.component_id],
        action_domain="model",
        action_id="paper.neck.gold_gather_distribute",
        train_overrides={
            "imgsz": 640,
            "target_actions": ["paper.neck.gold_gather_distribute"],
        },
        target_error_facts=[
            {
                "fact_type": "area_metric",
                "area": "small",
                "metric_name": "ap_small",
            }
        ],
    )
    node = ExperimentNode(
        node_id="node_paper_recipe_gold_neck",
        candidate_config=candidate,
        data_version="coco2017",
    )
    node.command_spec = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "data.yaml",
        project=tmp_path / "ultralytics",
        name="gold-neck",
    )
    report = LoopPolicyEvaluationReport(
        evaluations=[
            LoopPolicyEvaluation(
                policy_id="paper_recipe_gold_neck",
                decision="accepted",
                candidate_config=candidate,
                experiment_node=node,
                fixed_variables={"imgsz": 640},
                changed_variables={
                    "neck_component": ["neck.gold_gather_distribute"]
                },
            )
        ]
    )

    assessment = assess_candidate_execution(
        report,
        component_contracts=[contract],
        workspace=tmp_path / "component_execution",
        run_id="paper-neck-run",
        protocol_hash="paper-neck-protocol",
    )[0]

    assert assessment.execution_class == "executable"
    assert assessment.adapter_ids == ["neck.gold_gather_distribute"]
    assert assessment.adapter_patch_hash
    runtime_node = report.evaluations[0].experiment_node
    assert runtime_node is not None and runtime_node.command_spec is not None
    assert runtime_node.command_spec.metadata["adapter_runtime_payload_hash"]


def test_auto_optimization_driver_stops_without_fake_executable_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A dry auto round should produce artifacts and stop if all candidates need adapters."""
    llm_calls = 0

    class FakeAdvisor:
        def propose(self, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal llm_calls
            llm_calls += 1
            assert kwargs["inherited_context"]["decision_context_hash"]
            return LLMDecisionAdvisorResult(
                status="failed",
                provider="test",
                model="test-model",
                warnings=["forced_test_fallback"],
            )

    monkeypatch.setattr("yolo_agent.agents.policy_stage_runner.LLMDecisionAdvisor", lambda: FakeAdvisor())
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    task_path = run_root / "coco-yolo26n" / "task.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    ).to_yaml(task_path)

    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=run_root,
        profile="pilot",
        execute=False,
    )
    ErrorFactStore(run_root).append(
        base.run_id,
        [
            ErrorFact(
                run_id=base.run_id,
                candidate_id="yolo26n_coco_pilot",
                node_id="node_yolo26n_coco_pilot",
                dataset_version="coco2017",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.1,
                severity="high",
                action_candidates=["small_object_recipe", "bbox_loss_recipe"],
            )
        ],
    )

    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=False,
        executor="dry-run",
        max_steps=4,
    )

    child_dir = run_root / "coco-yolo26n-r1"
    assert result.rounds
    assert result.rounds[0].run_id == "coco-yolo26n-r1"
    assert result.rounds[0].auto_round_summary_path.exists()
    assert (child_dir / "artifacts" / "llm_decision.yaml").exists()
    assert (child_dir / "artifacts" / "paper_recipe_plan.yaml").exists()
    assert (child_dir / "artifacts" / "component_compatibility.yaml").exists()
    assert (child_dir / "artifacts" / "decision_ledger.jsonl").exists()
    assert (child_dir / "artifacts" / "policy_evaluation.yaml").exists()
    assert result.summary_path.exists()
    assert result.full_candidate_recommendations_path.exists()
    recommendations = yaml.safe_load(result.full_candidate_recommendations_path.read_text(encoding="utf-8-sig"))
    assert recommendations["full_run_started"] is False
    assert result.stopped_reason in {"no_executable_candidates", "requested_rounds_completed"}
    paper_plan = yaml.safe_load(
        (child_dir / "artifacts" / "paper_recipe_plan.yaml").read_text(encoding="utf-8-sig")
    )
    assert paper_plan["paper_claims_are_prior_only"] is True
    assert paper_plan["llm_status"] == "deferred_to_unified_decision_bundle"
    assert "recipe_critic_reports" in paper_plan
    assert "executable_pilot_policies" in paper_plan
    assert llm_calls == 1
    assert "Paper Intelligence" in result.summary_path.read_text(encoding="utf-8-sig")


def test_auto_optimization_driver_generates_executable_mosaic_pilot_from_background_fp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Background FP facts should unlock a real Ultralytics pilot instead of stopping."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    task_path = run_root / "coco-yolo26n" / "task.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    ).to_yaml(task_path)

    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=run_root,
        profile="pilot",
        execute=False,
    )
    ErrorFactStore(run_root).append(
        base.run_id,
        [
            ErrorFact(
                run_id=base.run_id,
                candidate_id="yolo26n_coco_pilot",
                node_id="node_yolo26n_coco_pilot",
                dataset_version="coco2017",
                fact_type="background_false_positive_class",
                subject="person",
                class_name="person",
                count=1200,
                severity="high",
                action_candidates=[
                    "hard_negative_mining",
                    "background_only_sampling",
                    "precision_threshold_tuning",
                ],
            )
        ],
    )

    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=False,
        executor="dry-run",
        max_steps=4,
    )

    assert result.stopped_reason == "requested_rounds_completed"
    assert result.rounds[0].executable_count >= 1
    assessments = {item.policy_id: item for item in result.rounds[0].candidate_assessments}
    mosaic = assessments["next_augmentation_reduce_mosaic_strength"]
    assert mosaic.execution_class == "executable"
    assert "model=yolo26n.pt" in mosaic.command
    assert "mosaic=0.2" in mosaic.command
    assert "imgsz=640" in mosaic.command

    def fail_if_round_is_reexecuted(*args: object, **kwargs: object) -> object:
        raise AssertionError("completed auto round should be reused, not re-executed")

    monkeypatch.setattr(AutoOptimizationLoopDriver, "_run_one_round", fail_if_round_is_reexecuted)
    reused = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=False,
        executor="dry-run",
        max_steps=4,
    )

    assert reused.rounds[0].run_id == "coco-yolo26n-r1"
    assert reused.rounds[0].status == "completed"
    assert reused.stopped_reason == "requested_rounds_completed"


def test_auto_optimization_execute_does_not_reuse_dry_run_round(tmp_path: Path, monkeypatch) -> None:
    """Execute mode must not treat a dry-run auto round as trained evidence."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    task_path = run_root / "coco-yolo26n" / "task.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    ).to_yaml(task_path)

    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=run_root,
        profile="pilot",
        execute=False,
    )
    ErrorFactStore(run_root).append(
        base.run_id,
        [
            ErrorFact(
                run_id=base.run_id,
                candidate_id="yolo26n_coco_pilot",
                node_id="node_yolo26n_coco_pilot",
                dataset_version="coco2017",
                fact_type="background_false_positive_class",
                subject="person",
                class_name="person",
                count=1200,
                severity="high",
                action_candidates=["hard_negative_mining"],
            )
        ],
    )
    AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=False,
        executor="dry-run",
        max_steps=4,
    )

    calls: list[str] = []

    def fake_execute_round(self: AutoOptimizationLoopDriver, **kwargs: object) -> AutoRoundResult:
        child = kwargs["child"]
        calls.append(child.context.run_id)
        return AutoRoundResult(
            round_index=1,
            run_id=child.context.run_id,
            run_dir=child.context.run_dir,
            parent_run_id=kwargs["parent"].context.run_id,
            status="completed",
            stop_reason="round_completed",
            auto_round_summary_path=child.context.artifact_path("auto_round_summary.yaml"),
        )

    monkeypatch.setattr(AutoOptimizationLoopDriver, "_run_one_round", fake_execute_round)
    executed = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=True,
        require_gpu_certification=False,
        executor="ultralytics-train",
        max_steps=4,
    )

    assert calls == ["coco-yolo26n-r1"]
    assert executed.rounds[0].run_id == "coco-yolo26n-r1"


def test_diversity_deferred_round_keeps_last_real_parent(tmp_path: Path, monkeypatch) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    base = OptimizeRunner().run(
        kind="coco", model="yolo26n.pt", data_yaml=data_yaml,
        run_id="diversity-parent", run_root=tmp_path / "runs",
        profile="pilot", execute=False,
    )
    ErrorFactStore(tmp_path / "runs").append(
        base.run_id,
        [
            ErrorFact(
                run_id=base.run_id, candidate_id="baseline", node_id="baseline",
                dataset_version="coco2017", fact_type="area_metric", subject="small",
                area="small", metric_name="ap_small", value=0.1, severity="high",
                action_candidates=["small_object_recipe"],
            )
        ],
    )
    parent_ids: list[str] = []

    def fake_round(self: AutoOptimizationLoopDriver, **kwargs: object) -> AutoRoundResult:
        parent = kwargs["parent"]
        child = kwargs["child"]
        round_index = int(kwargs["round_index"])
        parent_ids.append(parent.context.run_id)
        return AutoRoundResult(
            round_index=round_index, run_id=child.context.run_id,
            run_dir=child.context.run_dir, parent_run_id=parent.context.run_id,
            status="completed", stop_reason="diversity_deferred",
            auto_round_summary_path=child.context.artifact_path("auto_round_summary.yaml"),
        )

    monkeypatch.setattr(AutoOptimizationLoopDriver, "_run_one_round", fake_round)
    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir, auto_rounds=2, execute=True,
        require_gpu_certification=False,
        executor="ultralytics-train", max_steps=1,
    )
    assert parent_ids == [base.run_id, base.run_id]
    assert [item.run_id for item in result.rounds] == ["diversity-parent-r1", "diversity-parent-r2"]


def test_auto_optimization_execute_continues_after_completed_executed_round(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Execute mode should treat auto_rounds as additional work after completed executed rounds."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    task_path = run_root / "coco-yolo26n" / "task.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    ).to_yaml(task_path)
    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=run_root,
        profile="pilot",
        execute=False,
    )
    ErrorFactStore(run_root).append(
        base.run_id,
        [
            ErrorFact(
                run_id=base.run_id,
                candidate_id="yolo26n_coco_pilot",
                node_id="node_yolo26n_coco_pilot",
                dataset_version="coco2017",
                fact_type="background_false_positive_class",
                subject="person",
                class_name="person",
                count=1200,
                severity="high",
                action_candidates=["hard_negative_mining"],
            )
        ],
    )
    child_dir = run_root / "coco-yolo26n-r1"
    (child_dir / "artifacts").mkdir(parents=True)
    completed = AutoRoundResult(
        round_index=1,
        run_id="coco-yolo26n-r1",
        run_dir=child_dir,
        parent_run_id="coco-yolo26n",
        status="completed",
        stop_reason="round_completed",
        auto_round_summary_path=child_dir / "artifacts" / "auto_round_summary.yaml",
        training_loop={
            "run_id": "coco-yolo26n-r1",
            "profile": "pilot",
            "executor": "ultralytics-train",
            "auto_import": True,
            "max_steps": 1,
            "steps": [],
            "queue_counts": {"completed": 1},
            "stopped_reason": "complete",
            "completed": True,
        },
    )
    completed.auto_round_summary_path.write_text(
        yaml.safe_dump(completed.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_execute_round(self: AutoOptimizationLoopDriver, **kwargs: object) -> AutoRoundResult:
        child = kwargs["child"]
        calls.append(child.context.run_id)
        return AutoRoundResult(
            round_index=2,
            run_id=child.context.run_id,
            run_dir=child.context.run_dir,
            parent_run_id=kwargs["parent"].context.run_id,
            status="completed",
            stop_reason="round_completed",
            auto_round_summary_path=child.context.artifact_path("auto_round_summary.yaml"),
        )

    monkeypatch.setattr(AutoOptimizationLoopDriver, "_run_one_round", fake_execute_round)
    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=True,
        require_gpu_certification=False,
        executor="ultralytics-train",
        max_steps=4,
    )

    assert calls == ["coco-yolo26n-r2"]
    assert result.rounds[0].round_index == 2
    assert result.rounds[0].run_id == "coco-yolo26n-r2"


def test_auto_loop_consumes_cross_round_asha_promotion_before_new_proposal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A pending ASHA method must run despite a stale patience stop."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    task_path = run_root / "coco-yolo26n" / "task.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    ).to_yaml(task_path)
    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=run_root,
        profile="pilot",
        execute=False,
    )
    ErrorFactStore(run_root).append(
        base.run_id,
        [
            ErrorFact(
                run_id=base.run_id,
                candidate_id="baseline",
                node_id="baseline",
                dataset_version="coco2017",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.1,
                severity="high",
                action_candidates=["small_object_recipe"],
            )
        ],
    )
    scheduler = ASHAScheduler.create(base.run_id)
    for candidate_id, delta in (("a", 0.01), ("b", 0.04), ("c", 0.02)):
        node = ExperimentNode(
            node_id=f"node_{candidate_id}",
            candidate_config=CandidateConfig(
                candidate_id=candidate_id,
                base_model="yolo26n.pt",
                scale="n",
                framework="ultralytics",
            ),
            data_version="coco2017",
            command_spec=CommandSpec.ultralytics_train(
                model="yolo26n.pt",
                data=data_yaml,
                project=tmp_path / "ultralytics",
                name=candidate_id,
                epochs=3,
                imgsz=640,
                batch=48,
            ),
        )
        scheduler.register_trial(
            trial_id=candidate_id,
            candidate_id=candidate_id,
            source_run_id=f"run-{candidate_id}",
            source_node=node,
            target_error_facts=[{"fact_type": "area_metric", "subject": "small"}],
        )
        scheduler.report(
            candidate_id,
            ASHAObservation(
                stage_id="pilot_3",
                node_id=f"node_{candidate_id}__pilot_3",
                seed=42,
                paired_delta=delta,
                paired_result_verified=True,
                paired_experiment_result=verified_paired_result(
                    candidate_id=candidate_id,
                    node_id=f"node_{candidate_id}__pilot_3",
                    delta=delta,
                ),
            ),
        )
    ASHAStudyStore(base.run_dir / "artifacts" / "asha_state.yaml").save(scheduler)

    objective = OptimizationObjective(
        baseline_run_id=base.run_id,
        baseline_candidate_id="matched_baseline_control",
        baseline_protocol_hash="protocol-640",
    )
    stale_patience = OptimizationObjectiveStatus(
        objective_hash=objective.objective_hash,
        primary_metric="map50_95",
        no_improvement_rounds=4,
        should_stop=True,
        stop_reason="no_improvement_patience_reached",
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.load_optimization_objective",
        lambda _: objective,
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop._refresh_objective_status",
        lambda *_: stale_patience,
    )

    calls: list[str] = []

    def fail_new_proposal(*args: object, **kwargs: object) -> object:
        raise AssertionError("ASHA promotion should be consumed before generating a new proposal")

    def fake_asha_round(self: AutoOptimizationLoopDriver, **kwargs: object) -> AutoRoundResult:
        assignment = kwargs["assignment"]
        child = kwargs["child"]
        calls.append(f"{assignment.candidate_id}:{assignment.stage_id}")
        return AutoRoundResult(
            round_index=1,
            run_id=child.context.run_id,
            run_dir=child.context.run_dir,
            parent_run_id=kwargs["parent"].context.run_id,
            status="completed",
            stop_reason="asha_assignment_completed",
            auto_round_summary_path=child.context.artifact_path("auto_round_summary.yaml"),
        )

    monkeypatch.setattr(AutoOptimizationLoopDriver, "_run_one_round", fail_new_proposal)
    monkeypatch.setattr(AutoOptimizationLoopDriver, "_run_asha_assignment_round", fake_asha_round)

    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=True,
        require_gpu_certification=False,
        executor="ultralytics-train",
        max_steps=4,
    )

    assert calls == ["b:pilot_10"]
    assert result.stopped_reason == "requested_rounds_completed"


def test_auto_loop_honors_patience_stop_without_pending_asha_assignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="patience-without-assignment",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )
    objective = OptimizationObjective(
        baseline_run_id=base.run_id,
        baseline_candidate_id="matched_baseline_control",
        baseline_protocol_hash="protocol-640",
    )
    stale_patience = OptimizationObjectiveStatus(
        objective_hash=objective.objective_hash,
        primary_metric="map50_95",
        no_improvement_rounds=4,
        should_stop=True,
        stop_reason="no_improvement_patience_reached",
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.load_optimization_objective",
        lambda _: objective,
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop._refresh_objective_status",
        lambda *_: stale_patience,
    )

    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=True,
        require_gpu_certification=False,
        executor="ultralytics-train",
        max_steps=4,
    )

    assert result.stopped_reason == "no_improvement_patience_reached"
    assert result.rounds == []


def test_auto_loop_resumes_assignment_in_its_persisted_child_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A running assignment bound to r2 must never be rebound to the computed r1."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    task_path = run_root / "resume-bound" / "task.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    ).to_yaml(task_path)
    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="resume-bound",
        run_root=run_root,
        profile="pilot",
        execute=False,
    )
    ErrorFactStore(run_root).append(
        base.run_id,
        [
            ErrorFact(
                run_id=base.run_id,
                candidate_id="baseline",
                node_id="baseline",
                dataset_version="coco2017",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.1,
                severity="high",
                action_candidates=["small_object_recipe"],
            )
        ],
    )
    source = _asha_registration_node(
        tmp_path,
        candidate_id="scale_aug_0_3",
        search_tier="method",
    )
    control = _asha_registration_node(
        tmp_path,
        candidate_id="matched_baseline_control",
        search_tier="method",
        matched_control=True,
    )
    scheduler = ASHAScheduler.create(base.run_id)
    scheduler.register_trial(
        trial_id="scale-trial",
        candidate_id="scale_aug_0_3",
        source_run_id=f"{base.run_id}-r1",
        source_node=source,
        baseline_control_node=control,
    )
    assignment = scheduler.next_assignment()
    assert assignment is not None
    scheduler.mark_running(
        assignment,
        run_id=f"{base.run_id}-r2",
        node_id="node_scale_aug_0_3__pilot_3",
    )
    ASHAStudyStore(base.run_dir / "artifacts" / "asha_state.yaml").save(scheduler)
    LoopOrchestrator.from_run_dir(base.run_dir).fork_next(f"{base.run_id}-r2")

    calls: list[tuple[int, str, str | None]] = []

    def fake_asha_round(self: AutoOptimizationLoopDriver, **kwargs: object) -> AutoRoundResult:
        child = kwargs["child"]
        persisted = kwargs["assignment"]
        calls.append((kwargs["round_index"], child.context.run_id, persisted.assigned_run_id))
        return AutoRoundResult(
            round_index=kwargs["round_index"],
            run_id=child.context.run_id,
            run_dir=child.context.run_dir,
            parent_run_id=kwargs["parent"].context.run_id,
            status="completed",
            stop_reason="asha_assignment_completed",
            auto_round_summary_path=child.context.artifact_path("auto_round_summary.yaml"),
        )

    monkeypatch.setattr(AutoOptimizationLoopDriver, "_run_asha_assignment_round", fake_asha_round)

    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=True,
        require_gpu_certification=False,
        executor="ultralytics-train",
        max_steps=4,
    )

    assert calls == [(2, "resume-bound-r2", "resume-bound-r2")]
    assert result.rounds[0].round_index == 2


def test_evidence_recovery_queue_preserves_wrapped_source_training_argv(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="recovery",
        run_root=run_root,
        profile="pilot",
        execute=False,
    )
    orchestrator = LoopOrchestrator.from_run_dir(result.run_dir)
    checkpoint = tmp_path / "ultralytics" / "candidate" / "weights" / "best.pt"
    source = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=data_yaml,
        project=checkpoint.parent.parent.parent,
        name="candidate",
        epochs=3,
        imgsz=640,
        batch=48,
    )
    wrapped_argv = [
        r"C:\Python312\python.exe",
        "-m",
        "yolo_agent.adapters.ultralytics.runtime_entrypoint",
        "--payload",
        "E:/runs/payload.yaml",
        "--",
        *source.argv,
    ]
    wrapped = source.model_copy(
        update={
            "command": wrapped_argv[0],
            "args": wrapped_argv[1:],
            "argv": wrapped_argv,
            "expected_artifacts": {**source.expected_artifacts, "best_pt": checkpoint},
            "metadata": {
                **source.metadata,
                "adapter_runtime_entrypoint": "yolo_agent.adapters.ultralytics.runtime_entrypoint",
            },
        }
    )
    node = ExperimentNode(
        node_id="node_candidate__pilot_3",
        candidate_config=CandidateConfig(
            candidate_id="candidate",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        command_spec=wrapped,
        command=wrapped.display(),
    )
    incomplete = PilotEvidenceCompletenessResult(
        run_id=result.run_id,
        candidate_id="candidate",
        node_id=node.node_id,
        protocol_hash="protocol",
        complete=False,
        missing_metrics=["coco_post_eval_complete"],
        evidence_actions=["run_coco_post_eval"],
    )

    queue = _enqueue_coco_evidence_recovery(orchestrator, [node], [incomplete])

    assert len(queue.items) == 1
    recovery = queue.items[0].command
    assert json.loads(str(recovery.metadata["source_training_argv"])) == wrapped_argv
    assert recovery.metadata["training_run_dir"] == checkpoint.parent.parent.as_posix()
    assert recovery.command_type == "benchmark"


def test_merge_evidence_recovery_loop_marks_recovered_round_complete() -> None:
    original = TrainingLoopResult(
        run_id="run-r1",
        profile="pilot",
        executor="ultralytics-train",
        max_steps=8,
        queue_counts={"completed": 2},
        stopped_reason="complete",
        completed=True,
    )
    recovery = TrainingLoopResult(
        run_id="run-r1",
        profile="pilot",
        executor="ultralytics-train",
        max_steps=1,
        queue_counts={"completed": 1},
        stopped_reason="max_steps_reached",
        completed=False,
    )

    merged = _merge_evidence_recovery_loop(original, recovery, evidence_complete=True)

    assert merged is not None
    assert merged.completed is True
    assert merged.stopped_reason == "complete"
    assert merged.queue_counts == {"completed": 1}
