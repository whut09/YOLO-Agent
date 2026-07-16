"""Automatic pilot optimization loop driver.

The driver connects completed pilot evidence to the guarded loop machinery:

PilotResult -> LLMAnalysis -> PolicyProposal -> GuardedCandidate -> PilotRun -> DeltaAnalysis

It intentionally does not pretend that every metadata proposal can be trained.
Each accepted candidate is classified before queue materialization so the loop
only executes candidates backed by real adapter support.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.adapters.ultralytics.training import TrainingBudgetProfileName, UltralyticsTrainingConfig
from yolo_agent.adapters.ultralytics.training import HARNESS_ONLY_TRAIN_OVERRIDE_KEYS
from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopEngine
from yolo_agent.agents.error_to_action import DetectionErrorObservation, DetectionErrorType
from yolo_agent.agents.asha_scheduler import (
    ASHAAssignment,
    ASHAObservation,
    ASHAScheduler,
    ASHAStudy,
    ASHAStudyStore,
)
from yolo_agent.agents.diagnosis_promotion import (
    DiagnosisPromotionGate,
    DiagnosisPromotionPolicy,
)
from yolo_agent.agents.loop_evidence import error_fact_delta
from yolo_agent.agents.loop_io import read_json, read_yaml, write_json, write_yaml
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluationReport
from yolo_agent.agents.orchestrator import LoopOrchestrator, TrainingLoopResult
from yolo_agent.agents.paper_recipe_planner import PaperRecipePlanner
from yolo_agent.agents.recipe_critic import RecipeCritic
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint
from yolo_agent.core.coco_error_selection import select_coco_error_facts
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.experiment_graph import Evidence, ExperimentNode, ExperimentPlan, MetricEvidence
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.evidence_selector import EvidenceSelector, select_metric_evidence
from yolo_agent.core.matched_baseline import paired_metric_delta
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.components.contracts import load_contracts
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.policy_memory import PolicyMemoryStore
from yolo_agent.core.pilot_evidence import PilotEvidenceCompletenessGate, PilotEvidenceCompletenessResult
from yolo_agent.core.optimization_objective import (
    OptimizationObjective,
    OptimizationObjectiveStatus,
    evaluate_optimization_objective,
    load_optimization_objective,
)
from yolo_agent.core.round_execution_plan import RoundExecutionPlan, build_asha_assignment_plan
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.recipes.schemas import AtomicRecipe
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.reproduction_pipeline import ReproductionPipeline
from yolo_agent.research.snapshot import load_research_snapshot
from yolo_agent.resources import ResourcePaths
from yolo_agent.tools.dataset_stats import DatasetReport


CandidateExecutionClass = Literal["executable", "recommendation_only", "adapter_required"]


ADAPTER_REQUIRED_COMPONENT_PREFIXES = (
    "loss.bbox.",
    "head.",
    "neck.",
    "assigner.",
    "backbone.",
    "backbone_block.",
)

SAFE_ULTRALYTICS_OVERRIDE_KEYS = {
    "optimizer",
    "patience",
    "amp",
    "workers",
    "device",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "box",
    "cls",
    "dfl",
    "mosaic",
    "mixup",
    "copy_paste",
    "close_mosaic",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "erasing",
    "crop_fraction",
    "target_actions",
}

NON_TRAINING_DOMAINS = {"data", "label", "postprocess", "evidence"}

ACTION_EXPANSIONS: dict[str, list[str]] = {
    "hard_negative_mining": ["reduce_mosaic_strength"],
    "background_only_sampling": ["reduce_mosaic_strength"],
    "precision_threshold_tuning": ["reduce_mosaic_strength"],
    "bbox_loss_recipe": ["increase_box_loss_gain", "reduce_cls_loss_gain"],
    "assigner_recipe": ["increase_box_loss_gain"],
    "increase_recall_recipe": ["reduce_cls_loss_gain", "light_copy_paste", "light_mixup"],
    "class_balanced_sampling": ["light_copy_paste", "light_mixup"],
}


class CandidateExecutionAssessment(BaseModel):
    """Whether a guarded candidate can be executed by the current harness."""

    policy_id: str
    candidate_id: str | None = None
    node_id: str | None = None
    execution_class: CandidateExecutionClass
    command_type: str | None = None
    action_domain: str = ""
    action_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    required_adapters: list[str] = Field(default_factory=list)
    command: str = ""


class AutoRoundResult(BaseModel):
    """One automatic optimization round."""

    round_index: int
    run_id: str
    run_dir: Path
    parent_run_id: str
    status: Literal["completed", "blocked", "failed", "skipped"] = "completed"
    stop_reason: str = ""
    llm_decision_path: Path | None = None
    doctor_report_path: Path | None = None
    policy_evaluation_path: Path | None = None
    auto_round_summary_path: Path
    next_round_path: Path | None = None
    paper_recipe_plan_path: Path | None = None
    component_compatibility_path: Path | None = None
    reproduction_state_paths: list[Path] = Field(default_factory=list)
    training_loop: TrainingLoopResult | None = None
    candidate_assessments: list[CandidateExecutionAssessment] = Field(default_factory=list)

    @field_serializer(
        "run_dir",
        "llm_decision_path",
        "doctor_report_path",
        "policy_evaluation_path",
        "auto_round_summary_path",
        "next_round_path",
        "paper_recipe_plan_path",
        "component_compatibility_path",
    )
    def serialize_path(self, value: Path | None) -> str | None:
        """Serialize paths portably."""
        return value.as_posix() if value is not None else None

    @field_serializer("reproduction_state_paths")
    def serialize_reproduction_paths(self, value: list[Path]) -> list[str]:
        return [item.as_posix() for item in value]

    @property
    def executable_count(self) -> int:
        """Return how many accepted candidates can truly execute."""
        return sum(1 for item in self.candidate_assessments if item.execution_class == "executable")


class AutoOptimizationResult(BaseModel):
    """Summary for an automatic pilot optimization loop."""

    base_run_id: str
    base_run_dir: Path
    requested_rounds: int
    executed: bool
    profile: TrainingBudgetProfileName = "pilot"
    rounds: list[AutoRoundResult] = Field(default_factory=list)
    stopped_reason: str = ""
    summary_path: Path
    full_candidate_recommendations_path: Path
    asha_state_path: Path | None = None
    objective_status: OptimizationObjectiveStatus | None = None

    @field_serializer("base_run_dir", "summary_path", "full_candidate_recommendations_path", "asha_state_path")
    def serialize_path(self, value: Path | None) -> str | None:
        """Serialize paths portably."""
        return value.as_posix() if value is not None else None


def _log_auto_round_event(
    context: Any,
    *,
    event_type: Literal["auto_round_started", "auto_round_completed", "auto_round_blocked"],
    round_index: int,
    total_rounds: int,
    status: Literal["running", "completed", "blocked", "failed", "skipped"],
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Write one base-run auto-loop progress event."""
    event_details = {
        "round_index": round_index,
        "total_rounds": total_rounds,
        **(details or {}),
    }
    EventLog(context.run_dir / "events.jsonl").append(
        run_id=context.run_id,
        event_type=event_type,
        status=status,
        message=message,
        details=event_details,
    )


def _log_candidate_decisions(
    orchestrator: LoopOrchestrator,
    *,
    round_index: int,
    total_rounds: int,
    assessments: list[CandidateExecutionAssessment],
) -> None:
    """Write concise candidate strategy decisions for a round."""
    paper_context = _paper_progress_context(orchestrator.context.artifact_path("paper_recipe_plan.yaml"))
    remaining = sum(1 for item in assessments if item.execution_class == "executable")
    for assessment in assessments[:8]:
        strategy = assessment.action_id or assessment.action_domain or assessment.policy_id
        EventLog(orchestrator.context.run_dir / "events.jsonl").append(
            run_id=orchestrator.context.run_id,
            event_type="auto_round_decision",
            status="completed",
            message=(
                f"Round {round_index}/{total_rounds} strategy={strategy} "
                f"class={assessment.execution_class}."
            ),
            details={
                "round_index": round_index,
                "total_rounds": total_rounds,
                "policy_id": assessment.policy_id,
                "candidate_id": assessment.candidate_id,
                "node_id": assessment.node_id,
                "strategy": strategy,
                "execution_class": assessment.execution_class,
                "reasons": assessment.reasons,
                "diagnosis": paper_context.get("diagnosis"),
                "recipe": paper_context.get("recipe") or strategy,
                "changed_variable": paper_context.get("changed_variable") or assessment.action_id,
                "remaining_candidates": remaining,
            },
        )


def _paper_progress_context(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    raw = read_yaml(path)
    rule_plan = raw.get("rule_plan", {})
    selected = rule_plan.get("selected_recipes", []) if isinstance(rule_plan, dict) else []
    first = selected[0] if isinstance(selected, list) and selected and isinstance(selected[0], dict) else {}
    llm = raw.get("llm_proposal") if isinstance(raw.get("llm_proposal"), dict) else {}
    return {
        "diagnosis": str(llm.get("primary_problem") or ""),
        "recipe": str(first.get("recipe_id") or llm.get("selected_recipe") or ""),
        "changed_variable": str(first.get("primary_changed_variable") or ""),
    }


def _assessment_count(round_result: AutoRoundResult, execution_class: CandidateExecutionClass) -> int:
    """Count candidate assessments by execution class."""
    return sum(1 for item in round_result.candidate_assessments if item.execution_class == execution_class)


class AutoOptimizationLoopDriver:
    """Drive bounded automatic pilot rounds from error facts and guarded policy evaluation."""

    def run(
        self,
        base_run_dir: Path | str,
        auto_rounds: int,
        *,
        execute: bool,
        executor: str,
        max_steps: int = 8,
        auto_import: bool = True,
        profile: TrainingBudgetProfileName = "pilot",
        confirm_full_run: bool = False,
    ) -> AutoOptimizationResult:
        """Run up to ``auto_rounds`` child pilot rounds."""
        base_orchestrator = LoopOrchestrator.from_run_dir(base_run_dir)
        base_context = base_orchestrator.context
        summary_path = base_context.artifact_path("auto_optimization_summary.md")
        recommendations_path = base_context.artifact_path("full_candidate_recommendations.yaml")
        result = AutoOptimizationResult(
            base_run_id=base_context.run_id,
            base_run_dir=base_context.run_dir,
            requested_rounds=auto_rounds,
            executed=execute,
            profile=profile,
            summary_path=summary_path,
            full_candidate_recommendations_path=recommendations_path,
            asha_state_path=base_context.artifact_path("asha_state.yaml"),
        )
        objective = load_optimization_objective(base_context.metadata.get("optimization_objective_path"))
        asha_store = ASHAStudyStore(base_context.artifact_path("asha_state.yaml"))
        asha_scheduler = asha_store.load_or_create(base_context.run_id)
        if objective is not None:
            result.objective_status = _refresh_objective_status(base_context, objective)
        if auto_rounds <= 0:
            result.stopped_reason = "auto_rounds_zero"
            _write_final_outputs(result)
            return result
        if result.objective_status is not None and result.objective_status.should_stop:
            result.stopped_reason = result.objective_status.stop_reason
            _write_final_outputs(result)
            return result

        start_round_index = _next_executable_auto_round_index(base_context.run_root, base_context.run_id) if execute else 1
        end_round_index = start_round_index + auto_rounds - 1
        parent = (
            _latest_completed_auto_child(base_orchestrator, start_round_index - 1)
            if start_round_index > 1
            else base_orchestrator
        )
        for round_index in range(start_round_index, end_round_index + 1):
            _log_auto_round_event(
                base_context,
                event_type="auto_round_started",
                round_index=round_index,
                total_rounds=end_round_index,
                status="running",
                message=f"Auto optimization round {round_index}/{end_round_index} started.",
                details={"parent_run_id": parent.context.run_id},
            )
            parent_next = _ensure_next_round(parent)
            parent_facts = ErrorFactStore(parent.context.run_root).read(parent.context.run_id)
            if not parent_facts:
                round_result = _empty_round(
                    round_index=round_index,
                    parent=parent,
                    status="blocked",
                    stop_reason="missing_error_facts",
                )
                result.rounds.append(round_result)
                result.stopped_reason = "missing_error_facts"
                _log_auto_round_event(
                    base_context,
                    event_type="auto_round_blocked",
                    round_index=round_index,
                    total_rounds=end_round_index,
                    status="blocked",
                    message="Auto optimization blocked: missing error facts.",
                    details={"parent_run_id": parent.context.run_id, "stop_reason": "missing_error_facts"},
                )
                break

            child_run_id = f"{base_context.run_id}-r{round_index}"
            child = _fork_or_load_child(parent, child_run_id)
            _log_auto_round_event(
                base_context,
                event_type="auto_round_started",
                round_index=round_index,
                total_rounds=end_round_index,
                status="running",
                message=f"Auto round {round_index}/{end_round_index} using child run {child.context.run_id}.",
                details={"parent_run_id": parent.context.run_id, "child_run_id": child.context.run_id},
            )
            existing_round = _load_completed_round(child, round_index, parent.context.run_id, execute=execute)
            if existing_round is not None:
                result.rounds.append(existing_round)
                _log_auto_round_event(
                    base_context,
                    event_type="auto_round_completed",
                    round_index=round_index,
                    total_rounds=end_round_index,
                    status="completed",
                    message=f"Auto round {round_index}/{end_round_index} reused existing result.",
                    details={
                        "parent_run_id": parent.context.run_id,
                        "child_run_id": child.context.run_id,
                        "stop_reason": existing_round.stop_reason,
                    },
                )
                parent = child
                if objective is not None:
                    result.objective_status = _refresh_objective_status(base_context, objective)
                    if result.objective_status.should_stop:
                        result.stopped_reason = result.objective_status.stop_reason
                        break
                continue
            asha_assignment = asha_scheduler.next_assignment(confirm_full_run=confirm_full_run)
            assignment_profile: TrainingBudgetProfileName = (
                "candidate_full"
                if asha_assignment is not None and asha_assignment.stage_id.startswith("candidate_full")
                else profile
            )
            _prepare_child_training_context(child, parent, assignment_profile)
            _inherit_parent_dataset_report(child, parent)
            _inherit_parent_annotation_advice(child, parent)
            _inherit_parent_metric_evidence(child, parent)
            _repair_child_proposal_context(child, parent_facts)
            if asha_assignment is not None:
                round_result = self._run_asha_assignment_round(
                    round_index=round_index,
                    parent=parent,
                    child=child,
                    parent_facts=parent_facts,
                    parent_next_round=parent_next,
                    assignment=asha_assignment,
                    scheduler=asha_scheduler,
                    execute=execute,
                    executor=executor,
                    max_steps=max_steps,
                    auto_import=auto_import,
                    total_rounds=end_round_index,
                )
            else:
                round_result = self._run_one_round(
                    round_index=round_index,
                    parent=parent,
                    child=child,
                    parent_facts=parent_facts,
                    parent_next_round=parent_next,
                    execute=execute,
                    executor=executor,
                    max_steps=max_steps,
                    auto_import=auto_import,
                    profile=profile,
                    total_rounds=end_round_index,
                )
                if execute and round_result.status == "completed":
                    _register_completed_pilot_3_trials(asha_scheduler, child)
            asha_store.save(asha_scheduler)
            result.rounds.append(round_result)
            _log_auto_round_event(
                base_context,
                event_type="auto_round_completed" if round_result.status == "completed" else "auto_round_blocked",
                round_index=round_index,
                total_rounds=end_round_index,
                status=round_result.status,
                message=(
                    f"Auto round {round_index}/{end_round_index} {round_result.status}; "
                    f"stop={round_result.stop_reason} executable={round_result.executable_count}."
                ),
                details={
                    "parent_run_id": parent.context.run_id,
                    "child_run_id": child.context.run_id,
                    "stop_reason": round_result.stop_reason,
                    "executable_count": round_result.executable_count,
                    "adapter_required_count": _assessment_count(round_result, "adapter_required"),
                    "recommendation_only_count": _assessment_count(round_result, "recommendation_only"),
                },
            )
            if objective is not None and round_result.status == "completed":
                result.objective_status = _refresh_objective_status(base_context, objective)
                if result.objective_status.should_stop:
                    result.stopped_reason = result.objective_status.stop_reason
                    break
            if round_result.status != "completed" or round_result.stop_reason in {
                "no_guarded_candidates",
                "no_executable_candidates",
                "queue_blocked",
                "training_failed",
            }:
                result.stopped_reason = round_result.stop_reason or round_result.status
                break
            parent = child
        if not result.stopped_reason:
            result.stopped_reason = "requested_rounds_completed"
        _write_final_outputs(result)
        return result

    def _run_asha_assignment_round(
        self,
        *,
        round_index: int,
        parent: LoopOrchestrator,
        child: LoopOrchestrator,
        parent_facts: list[ErrorFact],
        parent_next_round: dict[str, Any],
        assignment: ASHAAssignment,
        scheduler: ASHAScheduler,
        execute: bool,
        executor: str,
        max_steps: int,
        auto_import: bool,
        total_rounds: int,
    ) -> AutoRoundResult:
        """Execute one cross-round ASHA promotion without generating a new recipe."""
        trial = scheduler.study.trial(assignment.trial_id)
        if execute:
            scheduler.mark_running(assignment)
        diagnosis_path = _ensure_loop_diagnosis_from_error_facts(child, parent_facts, parent_next_round)
        run_name = (
            f"{child.context.run_id}_{assignment.candidate_id}_{assignment.stage_id}"
            f"_seed{assignment.seed_index}"
        )
        round_plan = build_asha_assignment_plan(
            run_id=child.context.run_id,
            source_node=trial.source_node,
            stage_id=assignment.stage_id,
            epochs=assignment.epochs,
            fraction=assignment.fraction,
            seed=int(assignment.seed),
            seed_index=assignment.seed_index,
            run_name=run_name,
            baseline_control_node=trial.baseline_control_node,
        )
        candidate_node = next(node for node in round_plan.execution_nodes if not _matched_baseline_node(node))
        round_plan_path = child.context.artifact_path("round_execution_plan.yaml")
        experiment_plan_path = child.context.artifact_path("experiment_plan.yaml")
        round_plan.to_yaml(round_plan_path)
        round_plan.experiment_projection().to_yaml(experiment_plan_path)
        child.evidence_store.log_artifact_manifest(
            run_id=child.context.run_id,
            name="round_execution_plan",
            artifact_path=round_plan_path,
            producer_stage="asha_scheduler",
        )
        assessment = CandidateExecutionAssessment(
            policy_id=f"asha:{assignment.trial_id}:{assignment.stage_id}",
            candidate_id=assignment.candidate_id,
            node_id=candidate_node.node_id,
            execution_class="executable",
            command_type="train",
            action_domain="training",
            action_id=assignment.stage_id,
            reasons=[assignment.reason],
            command=candidate_node.command,
        )
        _log_candidate_decisions(
            child,
            round_index=round_index,
            total_rounds=total_rounds,
            assessments=[assessment],
        )
        training_loop = child.run_training_loop(
            profile=("pilot" if assignment.stage_id.startswith("pilot") else "candidate_full"),
            executor=executor if execute else "dry-run",
            max_steps=max_steps,
            auto_import=auto_import,
        )
        status: Literal["completed", "blocked", "failed", "skipped"] = "completed"
        stop_reason = "asha_assignment_completed"
        if execute and not training_loop.completed:
            status = "blocked"
            stop_reason = (
                "training_failed"
                if training_loop.queue_counts.get("failed", 0)
                else "queue_blocked"
            )
            scheduler.report(
                assignment.trial_id,
                ASHAObservation(
                    stage_id=assignment.stage_id,
                    node_id=candidate_node.node_id,
                    seed_index=assignment.seed_index,
                    seed=assignment.seed,
                    evidence_complete=False,
                    failure_reason=stop_reason,
                ),
            )
        elif execute:
            observation = _asha_observation(
                child,
                node=candidate_node,
                assignment=assignment,
                target_error_facts=trial.target_error_facts,
            )
            scheduler.report(assignment.trial_id, observation)
            if not observation.evidence_complete:
                status = "blocked"
                stop_reason = "asha_evidence_incomplete"
        child.next_round()
        summary_path = child.context.artifact_path("auto_round_summary.yaml")
        result = AutoRoundResult(
            round_index=round_index,
            run_id=child.context.run_id,
            run_dir=child.context.run_dir,
            parent_run_id=parent.context.run_id,
            status=status,
            stop_reason=stop_reason,
            doctor_report_path=diagnosis_path,
            auto_round_summary_path=summary_path,
            next_round_path=_existing_or_none(child.context.artifact_path("next_round.yaml")),
            training_loop=training_loop,
            candidate_assessments=[assessment],
        )
        write_yaml(summary_path, result.model_dump(mode="json"))
        return result

    def _run_one_round(
        self,
        *,
        round_index: int,
        parent: LoopOrchestrator,
        child: LoopOrchestrator,
        parent_facts: list[ErrorFact],
        parent_next_round: dict[str, Any],
        execute: bool,
        executor: str,
        max_steps: int,
        auto_import: bool,
        profile: TrainingBudgetProfileName,
        total_rounds: int,
    ) -> AutoRoundResult:
        """Run one child round through diagnosis, policy evaluation, and pilot execution."""
        status: Literal["completed", "blocked", "failed", "skipped"] = "completed"
        stop_reason = ""
        training_loop: TrainingLoopResult | None = None

        diagnosis_path = _ensure_loop_diagnosis_from_error_facts(child, parent_facts, parent_next_round)
        paper_recipe_paths = _ensure_paper_intelligence(child, parent_facts, diagnosis_path)
        for stage in ["generate_loop_plan", "evaluate_policies", "generate_candidates", "ablate"]:
            stage_result = child.run_stage(stage)  # type: ignore[arg-type]
            if stage_result.status in {"blocked", "failed"}:
                status = stage_result.status
                stop_reason = f"{stage}_{stage_result.status}"
                break

        assessments = _assess_policy_evaluation(child.context.artifact_path("policy_evaluation.yaml"))
        _log_candidate_decisions(child, round_index=round_index, total_rounds=total_rounds, assessments=assessments)
        if status == "completed":
            if not assessments:
                status = "blocked"
                stop_reason = "no_guarded_candidates"
            else:
                executable_nodes = _executable_nodes(child.context.artifact_path("experiment_plan.yaml"), assessments)
                if not executable_nodes:
                    status = "blocked"
                    stop_reason = "no_executable_candidates"
                else:
                    _write_filtered_experiment_plan(child, executable_nodes, assessments)
                    if execute:
                        training_loop = child.run_training_loop(
                            profile=profile,
                            executor=executor,
                            max_steps=max_steps,
                            auto_import=auto_import,
                        )
                        if not training_loop.completed:
                            status = "blocked"
                            stop_reason = (
                                "training_failed"
                                if training_loop.queue_counts.get("failed", 0)
                                else "queue_blocked"
                            )
                    else:
                        training_loop = child.run_training_loop(
                            profile=profile,
                            executor="dry-run",
                            max_steps=max_steps,
                            auto_import=auto_import,
                        )
                    completeness_results: list[PilotEvidenceCompletenessResult] = []
                    if execute and training_loop is not None and training_loop.completed:
                        completeness_results = _persist_pilot_evidence_completeness(
                            child,
                            [node for node in executable_nodes if not _matched_baseline_node(node)],
                        )
                        if any(not item.complete for item in completeness_results):
                            status = "blocked"
                            stop_reason = "pilot_evidence_incomplete"
                    child.next_round()
                    if completeness_results:
                        _apply_pilot_evidence_gate_to_next_round(child, completeness_results)
                    _update_reproduction_after_round(
                        child,
                        parent,
                        paper_recipe_paths.get("paper_recipe_plan"),
                        paper_recipe_paths.get("reproduction_states", []),
                        training_loop,
                        assessments,
                    )

        next_round_path = child.context.artifact_path("next_round.yaml")
        round_result = AutoRoundResult(
            round_index=round_index,
            run_id=child.context.run_id,
            run_dir=child.context.run_dir,
            parent_run_id=parent.context.run_id,
            status=status,
            stop_reason=stop_reason or "round_completed",
            llm_decision_path=_existing_or_none(child.context.artifact_path("llm_decision.yaml")),
            doctor_report_path=diagnosis_path,
            policy_evaluation_path=_existing_or_none(child.context.artifact_path("policy_evaluation.yaml")),
            auto_round_summary_path=child.context.artifact_path("auto_round_summary.yaml"),
            next_round_path=_existing_or_none(next_round_path),
            paper_recipe_plan_path=paper_recipe_paths.get("paper_recipe_plan"),
            component_compatibility_path=paper_recipe_paths.get("component_compatibility"),
            reproduction_state_paths=paper_recipe_paths.get("reproduction_states", []),
            training_loop=training_loop,
            candidate_assessments=assessments,
        )
        write_yaml(round_result.auto_round_summary_path, round_result.model_dump(mode="json"))
        child.evidence_store.log_artifact_manifest(
            run_id=child.context.run_id,
            name="auto_round_summary",
            artifact_path=round_result.auto_round_summary_path,
            producer_stage="auto_optimization_loop",
        )
        return round_result


def _persist_pilot_evidence_completeness(
    orchestrator: LoopOrchestrator,
    nodes: list[ExperimentNode],
) -> list[PilotEvidenceCompletenessResult]:
    """Evaluate current-node evidence and persist a machine-readable gate report."""
    gate = PilotEvidenceCompletenessGate(orchestrator.evidence_store)
    results = [
        gate.evaluate(
            run_id=orchestrator.context.run_id,
            candidate_id=node.candidate_config.candidate_id,
            node_id=node.node_id,
        )
        for node in nodes
    ]
    path = orchestrator.context.artifact_path("pilot_evidence_completeness.yaml")
    write_yaml(
        path,
        {
            "schema_version": "1.0",
            "run_id": orchestrator.context.run_id,
            "complete": bool(results) and all(item.complete for item in results),
            "nodes": [item.model_dump(mode="json") for item in results],
        },
    )
    orchestrator.evidence_store.log_artifact_manifest(
        run_id=orchestrator.context.run_id,
        name="pilot_evidence_completeness",
        artifact_path=path,
        producer_stage="pilot_evidence_completeness_gate",
    )
    if any(not item.complete for item in results):
        EventLog(orchestrator.context.events_path).append(
            run_id=orchestrator.context.run_id,
            event_type="contract_blocked",
            status="blocked",
            message="Pilot evidence is incomplete; only evidence collection actions are allowed.",
            details={
                "node_ids": [item.node_id for item in results if not item.complete],
                "evidence_actions": list(
                    dict.fromkeys(action for item in results for action in item.evidence_actions)
                ),
                "artifact": path.as_posix(),
            },
        )
    return results


def _register_completed_pilot_3_trials(
    scheduler: ASHAScheduler,
    child: LoopOrchestrator,
) -> None:
    """Add completed pilot-3 nodes to the base-run ASHA cohort."""
    plan_path = child.context.artifact_path("round_execution_plan.yaml")
    if not plan_path.is_file():
        return
    plan = RoundExecutionPlan.from_yaml(plan_path)
    source_by_candidate = {
        node.candidate_config.candidate_id: node
        for node in plan.deferred_nodes
        if not _matched_baseline_node(node)
    }
    baseline_control = next(
        (node for node in plan.deferred_nodes if _matched_baseline_node(node)),
        None,
    )
    for node in plan.execution_nodes:
        if _matched_baseline_node(node):
            continue
        source = source_by_candidate.get(node.candidate_config.candidate_id)
        if source is None:
            continue
        trial_id = f"{child.context.run_id}:{node.candidate_config.candidate_id}"
        raw_targets = source.candidate_config.train_overrides.get("target_error_facts", [])
        target_error_facts = [
            dict(item)
            for item in raw_targets
            if isinstance(raw_targets, list) and isinstance(item, dict)
        ]
        scheduler.register_trial(
            trial_id=trial_id,
            candidate_id=node.candidate_config.candidate_id,
            source_run_id=child.context.run_id,
            source_node=source,
            baseline_control_node=baseline_control,
            target_error_facts=target_error_facts,
        )
        scheduler.report(
            trial_id,
            _asha_observation(
                child,
                node=node,
                assignment=ASHAAssignment(
                    trial_id=trial_id,
                    candidate_id=node.candidate_config.candidate_id,
                    stage_id="pilot_3",
                    seed_index=1,
                    seed=node.seed,
                    epochs=3,
                    fraction=0.1,
                    reason="initial_guarded_pilot_3",
                ),
                target_error_facts=target_error_facts,
            ),
        )


def _asha_observation(
    child: LoopOrchestrator,
    *,
    node: ExperimentNode,
    assignment: ASHAAssignment,
    target_error_facts: list[dict[str, object]],
) -> ASHAObservation:
    """Build one strict paired ASHA observation from imported local evidence."""
    evidence = child.evidence_store.load_run(child.context.run_id)
    candidate_records = [
        record
        for record in evidence.metric_records
        if record.run_id == child.context.run_id
        and record.node_id == node.node_id
        and record.candidate_id == node.candidate_config.candidate_id
        and record.metric_name == "map50_95"
        and record.evidence_role == "current_observation"
        and record.inheritance_depth == 0
        and record.verified
        and isinstance(record.value, (int, float))
    ]
    baseline_records = [
        record
        for record in evidence.metric_records
        if record.metric_name == "map50_95"
        and record.evidence_role == "baseline_reference"
        and record.verified
    ]
    paired_delta_value: float | None = None
    if candidate_records:
        candidate = max(candidate_records, key=lambda record: record.created_at)
        _, delta = paired_metric_delta(candidate, baseline_records)
        if delta is not None:
            paired_delta_value = delta.effect_delta

    facts = ErrorFactStore(child.context.run_root).read(child.context.run_id)
    fact_delta = error_fact_delta(facts, facts)
    improved_count = sum(
        1
        for item in fact_delta.get("improved_errors", [])
        if isinstance(item, dict) and _matches_target_error_fact(item, target_error_facts)
    )
    requires_target_facts = assignment.stage_id in {
        "pilot_10",
        "candidate_full_seed_1",
        "candidate_full_confirmation",
    }
    diagnosis_result = None
    if assignment.stage_id != "pilot_3":
        objective = load_optimization_objective(child.context.metadata.get("optimization_objective_path"))
        policy = DiagnosisPromotionPolicy(
            max_latency_regression=(objective.max_latency_regression if objective is not None else 0.05),
            max_model_size_regression=(objective.max_model_size_regression if objective is not None else 0.10),
        )
        diagnosis_result = DiagnosisPromotionGate(policy).evaluate(
            candidate_id=node.candidate_config.candidate_id,
            node_id=node.node_id,
            target_error_facts=[dict(item) for item in target_error_facts],
            metric_records=evidence.metric_records,
            error_facts=facts,
        )
    missing_diagnosis_evidence = bool(
        diagnosis_result is not None
        and any(check.status == "missing" for check in diagnosis_result.checks)
    )
    evidence_complete = (
        paired_delta_value is not None
        and (not requires_target_facts or (bool(target_error_facts) and bool(facts)))
        and not missing_diagnosis_evidence
    )
    latency_regression = _diagnosis_observed_delta(diagnosis_result, "latency_guard")
    model_size_regression = _diagnosis_observed_delta(diagnosis_result, "model_size_guard")
    return ASHAObservation(
        stage_id=assignment.stage_id,
        node_id=node.node_id,
        seed_index=assignment.seed_index,
        seed=assignment.seed,
        paired_delta=paired_delta_value,
        target_error_improved_count=improved_count,
        latency_regression=latency_regression,
        model_size_regression=model_size_regression,
        diagnosis_gate_passed=(diagnosis_result.allowed if diagnosis_result is not None else None),
        diagnosis_checks=(
            [check.model_dump(mode="json") for check in diagnosis_result.checks]
            if diagnosis_result is not None
            else []
        ),
        promotion_rejection_reasons=(
            diagnosis_result.rejection_reasons if diagnosis_result is not None else []
        ),
        evidence_complete=evidence_complete,
        failure_reason="",
    )


def _diagnosis_observed_delta(result: Any, check_id: str) -> float | None:
    if result is None:
        return None
    for check in result.checks:
        if check.check_id == check_id:
            return check.observed_delta
    return None


def _matches_target_error_fact(
    delta_item: dict[str, Any],
    targets: list[dict[str, object]],
) -> bool:
    if not targets:
        return False
    identity_fields = ("fact_type", "subject", "class_name", "class_pair", "area", "metric_name")
    for target in targets:
        compared = 0
        matched = True
        for field in identity_fields:
            expected = target.get(field)
            if expected in {None, ""}:
                continue
            compared += 1
            if str(delta_item.get(field) or "") != str(expected):
                matched = False
                break
        if matched and compared >= 2:
            return True
    return False


def _matched_baseline_node(node: ExperimentNode) -> bool:
    return bool(node.command_spec and node.command_spec.metadata.get("matched_baseline_control"))


def _apply_pilot_evidence_gate_to_next_round(
    orchestrator: LoopOrchestrator,
    results: list[PilotEvidenceCompletenessResult],
) -> None:
    """Prevent another training proposal when current-node facts are incomplete."""
    incomplete = [item for item in results if not item.complete]
    if not incomplete:
        return
    path = orchestrator.context.artifact_path("next_round.yaml")
    payload = read_yaml(path) if path.is_file() else {}
    if not isinstance(payload, dict):
        payload = {}
    actions = list(dict.fromkeys(action for item in incomplete for action in item.evidence_actions))
    payload.update(
        {
            "proposal_mode": "evidence_only",
            "training_proposals_allowed": False,
            "full_candidate_proposal_allowed": False,
            "pilot_evidence_complete": False,
            "pilot_evidence_actions": actions,
            "pilot_evidence_incomplete_nodes": [item.node_id for item in incomplete],
            "next_action": actions[0] if actions else "collect_current_node_coco_evidence",
        }
    )
    write_yaml(path, payload)


def assess_candidate_execution(report: LoopPolicyEvaluationReport) -> list[CandidateExecutionAssessment]:
    """Classify guarded policy evaluations by real execution support."""
    assessments: list[CandidateExecutionAssessment] = []
    for evaluation in report.evaluations:
        if evaluation.decision != "accepted" or evaluation.candidate_config is None:
            continue
        candidate = evaluation.candidate_config
        node = evaluation.experiment_node
        command = node.command_spec if node is not None else None
        execution_class: CandidateExecutionClass = "executable"
        reasons: list[str] = []
        required_adapters: list[str] = []

        if command is None:
            execution_class = "recommendation_only"
            reasons.append("accepted policy has no command_spec")
        elif command.command_type != "train":
            execution_class = "recommendation_only"
            reasons.append(f"command_type={command.command_type} is not a pilot training command")

        if candidate.action_domain in NON_TRAINING_DOMAINS:
            execution_class = "recommendation_only"
            reasons.append(f"action_domain={candidate.action_domain} is advisory or evidence-first")

        component_adapters = _required_component_adapters(candidate.components)
        if component_adapters:
            execution_class = "adapter_required"
            required_adapters.extend(component_adapters)
            reasons.append("candidate uses metadata-only model components")

        unsupported_overrides = (
            _unsupported_train_overrides(candidate.train_overrides)
            if command is not None
            and command.command_type == "train"
            and candidate.action_domain not in NON_TRAINING_DOMAINS
            else []
        )
        if unsupported_overrides:
            execution_class = "adapter_required"
            required_adapters.extend(f"ultralytics_override:{key}" for key in unsupported_overrides)
            reasons.append("candidate train_overrides are not mapped to safe Ultralytics CLI options")

        if execution_class == "executable" and not reasons:
            reasons.append("train command uses only currently supported Ultralytics CLI options")

        assessments.append(
            CandidateExecutionAssessment(
                policy_id=evaluation.policy_id,
                candidate_id=candidate.candidate_id,
                node_id=node.node_id if node is not None else None,
                execution_class=execution_class,
                command_type=command.command_type if command is not None else None,
                action_domain=candidate.action_domain,
                action_id=candidate.action_id,
                reasons=list(dict.fromkeys(reasons)),
                required_adapters=list(dict.fromkeys(required_adapters)),
                command=command.display() if command is not None else "",
            )
        )
    return assessments


def _ensure_next_round(orchestrator: LoopOrchestrator) -> dict[str, Any]:
    path = orchestrator.context.artifact_path("next_round.yaml")
    if not path.is_file():
        orchestrator.next_round()
    return read_yaml(path) if path.is_file() else {}


def _fork_or_load_child(parent: LoopOrchestrator, child_run_id: str) -> LoopOrchestrator:
    child_dir = parent.context.run_root / child_run_id
    if child_dir.exists():
        return LoopOrchestrator.from_run_dir(child_dir)
    return parent.fork_next(child_run_id)


def _load_completed_round(
    child: LoopOrchestrator,
    round_index: int,
    parent_run_id: str,
    *,
    execute: bool,
) -> AutoRoundResult | None:
    """Return an existing terminal round so reruns do not repeat training."""
    path = child.context.artifact_path("auto_round_summary.yaml")
    if not path.is_file():
        return None
    try:
        result = AutoRoundResult.model_validate(read_yaml(path))
    except ValueError:
        return None
    if result.round_index != round_index:
        return None
    if result.run_id != child.context.run_id or result.parent_run_id != parent_run_id:
        return None
    if result.status != "completed" or result.stop_reason != "round_completed":
        return None
    if execute and result.training_loop is not None and result.training_loop.executor == "dry-run":
        return None
    return result


def _next_executable_auto_round_index(run_root: Path, base_run_id: str) -> int:
    """Return the next absolute round index after completed executed child rounds."""
    completed = [
        index
        for index, result in _completed_auto_rounds(run_root, base_run_id).items()
        if result.training_loop is not None
        and result.training_loop.executor != "dry-run"
        and result.status == "completed"
        and result.stop_reason == "round_completed"
    ]
    return (max(completed) + 1) if completed else 1


def _latest_completed_auto_child(base: LoopOrchestrator, round_index: int) -> LoopOrchestrator:
    """Return the latest completed child up to round_index, or the base run."""
    if round_index <= 0:
        return base
    child_dir = base.context.run_root / f"{base.context.run_id}-r{round_index}"
    if (child_dir / "run_context.yaml").is_file():
        return LoopOrchestrator.from_run_dir(child_dir)
    completed = [
        index
        for index in _completed_auto_rounds(base.context.run_root, base.context.run_id)
        if index <= round_index
    ]
    if not completed:
        return base
    latest_dir = base.context.run_root / f"{base.context.run_id}-r{max(completed)}"
    if not (latest_dir / "run_context.yaml").is_file():
        return base
    return LoopOrchestrator.from_run_dir(latest_dir)


def _completed_auto_rounds(run_root: Path, base_run_id: str) -> dict[int, AutoRoundResult]:
    """Load completed auto round summaries keyed by absolute round index."""
    import re

    rounds: dict[int, AutoRoundResult] = {}
    pattern = re.compile(rf"^{re.escape(base_run_id)}-r(?P<index>\d+)$")
    child_dirs = run_root.iterdir() if run_root.is_dir() else []
    for child_dir in child_dirs:
        if not child_dir.is_dir():
            continue
        match = pattern.match(child_dir.name)
        if not match:
            continue
        path = child_dir / "artifacts" / "auto_round_summary.yaml"
        if not path.is_file():
            continue
        try:
            result = AutoRoundResult.model_validate(read_yaml(path))
        except ValueError:
            continue
        rounds[int(match.group("index"))] = result
    return rounds


def _prepare_child_training_context(
    child: LoopOrchestrator,
    parent: LoopOrchestrator,
    profile: TrainingBudgetProfileName,
) -> None:
    parent_meta = parent.context.metadata
    child.context.metadata["training_profile"] = profile
    for key in (
        "training_config_path",
        "training_model",
        "research_snapshot_hash",
        "research_snapshot_path",
        "research_snapshot_verified",
    ):
        if key in parent_meta and key not in child.context.metadata:
            child.context.metadata[key] = parent_meta[key]
    inferred_model = _infer_training_model(parent)
    if inferred_model:
        child.context.metadata["training_model"] = inferred_model
    child.context.metadata["auto_optimization_round"] = child.context.metadata.get("auto_optimization_round", "")
    child.context.to_yaml()
    child.context.to_json()


def _inherit_parent_dataset_report(child: LoopOrchestrator, parent: LoopOrchestrator) -> None:
    """Reuse an existing dataset report instead of profiling full COCO every round."""
    child_report = child.context.artifact_path("dataset_report.json")
    if child_report.is_file():
        return
    parent_report = parent.context.artifact_path("dataset_report.json")
    if not parent_report.is_file():
        return
    child_report.parent.mkdir(parents=True, exist_ok=True)
    child_report.write_text(parent_report.read_text(encoding="utf-8-sig"), encoding="utf-8")
    child.artifacts.record("profile_data", {"dataset_report": child_report})
    child.state.mark(
        "profile_data",
        "completed",
        f"Inherited dataset report from parent run {parent.context.run_id}.",
        {"dataset_report": child_report},
    )
    child.state.to_yaml(child.context.run_dir / "loop_state.yaml")
    child.event_log.append(
        run_id=child.context.run_id,
        event_type="stage_completed",
        stage="profile_data",
        status="completed",
        message=f"Inherited dataset report from parent run {parent.context.run_id}.",
        artifacts={"dataset_report": child_report},
    )


def _inherit_parent_annotation_advice(child: LoopOrchestrator, parent: LoopOrchestrator) -> None:
    """Reuse label-quality advice instead of rescanning the dataset every child round."""
    child_json = child.context.artifact_path("annotation_advice.json")
    if child_json.is_file():
        return
    parent_json = parent.context.artifact_path("annotation_advice.json")
    if not parent_json.is_file():
        return
    child_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_json, child_json)
    parent_md = parent.context.artifact_path("annotation_advice.md")
    child_md = child.context.artifact_path("annotation_advice.md")
    artifacts = {"annotation_advice": child_json}
    if parent_md.is_file():
        shutil.copy2(parent_md, child_md)
        artifacts["annotation_advice_md"] = child_md
    child.artifacts.record("advise_labels", artifacts)
    child.state.mark(
        "advise_labels",
        "completed",
        f"Inherited annotation advice from parent run {parent.context.run_id}.",
        artifacts,
    )
    child.state.to_yaml(child.context.run_dir / "loop_state.yaml")
    child.event_log.append(
        run_id=child.context.run_id,
        event_type="stage_completed",
        stage="advise_labels",
        status="completed",
        message=f"Inherited annotation advice from parent run {parent.context.run_id}.",
        artifacts=artifacts,
    )


def _repair_child_proposal_context(child: LoopOrchestrator, parent_facts: list[ErrorFact]) -> None:
    """Overwrite stale fork metadata with current error-fact-driven pilot context."""
    selection = select_coco_error_facts(
        parent_facts,
        baseline_node_ids=list(dict.fromkeys(fact.node_id for fact in parent_facts)),
        max_focus=8,
    )
    focus = _expanded_focus_items(parent_facts, selection.current_round_focus)
    actions = _expanded_actions([str(action) for item in focus for action in item.get("action_candidates", [])])
    if not focus or not actions:
        return
    tried_actions = _tried_action_ids(child.context.run_root, _base_auto_run_id(child.context.run_id))
    child.context.metadata.update(
        {
            "inherited_current_round_focus": focus,
            "inherited_current_round_error_actions": actions,
            "inherited_tried_action_ids": tried_actions,
            "inherited_proposal_mode": "pilot_only",
            "inherited_proposal_budget_profiles_allowed": ["debug", "pilot"],
            "inherited_proposal_budget_profiles_blocked": ["candidate_full"],
            "inherited_proposal_required_bindings": ["target_error_facts", "expected_improvement"],
            "inherited_guardrails": list(
                dict.fromkeys(
                    [
                        *[
                            str(item)
                            for item in child.context.metadata.get("inherited_guardrails", [])
                            if str(item) not in {"proposal_generation_blocked_until_error_facts_exist"}
                        ],
                        "auto_loop_repaired_stale_fork_context_from_parent_error_facts",
                        "pilot_only_proposals",
                        "candidate_full_blocked_until_pilot_promotion",
                    ]
                )
            ),
        }
    )
    child.context.to_yaml()
    child.context.to_json()


def _expanded_focus_items(parent_facts: list[ErrorFact], selected: list[Any]) -> list[dict[str, Any]]:
    """Keep selected focus diverse enough to expose executable policy actions."""
    focus = [item.model_dump(mode="json") for item in selected]
    seen = {_focus_key(item) for item in focus}
    for fact in sorted(parent_facts, key=_fact_rank):
        if fact.severity not in {"high", "medium"}:
            continue
        key = _fact_key(fact)
        if key in seen:
            continue
        if not set(fact.action_candidates).intersection(ACTION_EXPANSIONS):
            continue
        item = {
            "diagnosis_id": ":".join(part for part in [fact.fact_type, fact.subject] if part),
            "diagnosis_kind": "background_fp_class" if fact.fact_type == "background_false_positive_class" else "generic_error_fact",
            "fact_type": fact.fact_type,
            "subject": fact.subject,
            "class_name": fact.class_name,
            "class_pair": fact.class_pair,
            "area": fact.area,
            "metric_name": fact.metric_name,
            "value": fact.value,
            "count": fact.count,
            "severity": fact.severity,
            "priority": 0.0,
            "action_candidates": list(fact.action_candidates),
            "target_error_key": ":".join(part for part in key if part),
            "candidate_id": fact.candidate_id,
            "node_id": fact.node_id,
            "reason": "Added because it unlocks a currently executable pilot action.",
        }
        focus.append({name: value for name, value in item.items() if value is not None})
        seen.add(key)
        if len(focus) >= 8:
            break
    return focus


def _expanded_actions(actions: list[str]) -> list[str]:
    expanded: list[str] = []
    for action in actions:
        expanded.append(action)
        expanded.extend(ACTION_EXPANSIONS.get(action, []))
    return list(dict.fromkeys(item for item in expanded if item))


def _base_auto_run_id(run_id: str) -> str:
    import re

    return re.sub(r"-r\d+$", "", run_id)


def _tried_action_ids(run_root: Path, base_run_id: str) -> list[str]:
    """Return previously executed auto-loop action ids for a base run."""
    tried: list[str] = []
    for path in sorted(run_root.glob(f"{base_run_id}-r*/artifacts/auto_round_summary.yaml")):
        try:
            raw = read_yaml(path)
        except Exception:
            continue
        assessments = raw.get("candidate_assessments", []) if isinstance(raw, dict) else []
        if not isinstance(assessments, list):
            continue
        for item in assessments:
            if not isinstance(item, dict):
                continue
            if item.get("execution_class") != "executable":
                continue
            action_id = item.get("action_id")
            if action_id:
                tried.append(str(action_id))
    return list(dict.fromkeys(tried))


def _ensure_loop_diagnosis_from_error_facts(
    child: LoopOrchestrator,
    parent_facts: list[ErrorFact],
    parent_next_round: dict[str, Any],
) -> Path:
    diagnosis_path = child.context.artifact_path("loop_diagnosis.json")
    dataset_report_path = child.context.artifact_path("dataset_report.json")
    if not dataset_report_path.is_file():
        child.run_stage("profile_data")
    dataset_report = DatasetReport.model_validate(read_json(dataset_report_path))
    task_spec = TaskSpec.from_yaml(child.context.task_path)
    observations = _observations_from_error_facts(parent_facts, parent_next_round)
    training_config = _training_config_from_context(child)
    report = ErrorDrivenLoopEngine().run(
        task_spec=task_spec,
        dataset_report=dataset_report,
        detection_errors=observations,
        evidence_status=_evidence_status_from_parent(child, parent_facts),
        fixed_imgsz=training_config.imgsz if training_config is not None else None,
    )
    write_json(diagnosis_path, report.model_dump(mode="json"))
    child.artifacts.record("diagnose_errors", {"loop_diagnosis": diagnosis_path})
    child.state.mark(
        "diagnose_errors",
        "completed",
        f"Created loop diagnosis from {len(parent_facts)} parent error facts.",
        {"loop_diagnosis": diagnosis_path},
    )
    child.state.to_yaml(child.context.run_dir / "loop_state.yaml")
    child.event_log.append(
        run_id=child.context.run_id,
        event_type="stage_completed",
        stage="diagnose_errors",
        status="completed",
        message="Created loop diagnosis from parent error facts for auto optimization.",
        artifacts={"loop_diagnosis": diagnosis_path},
        details={
            "parent_error_fact_count": len(parent_facts),
            "observation_count": len(observations),
        },
    )
    return diagnosis_path


def _ensure_paper_intelligence(
    child: LoopOrchestrator,
    parent_facts: list[ErrorFact],
    diagnosis_path: Path,
) -> dict[str, Any]:
    """Run paper, recipe, critic, and reproduction bookkeeping before policy stages."""
    plan_path = child.context.artifact_path("paper_recipe_plan.yaml")
    compatibility_path = child.context.artifact_path("component_compatibility.yaml")
    state_paths: list[Path] = []
    try:
        dataset_report_path = child.context.artifact_path("dataset_report.json")
        dataset_report = DatasetReport.model_validate(read_json(dataset_report_path)) if dataset_report_path.is_file() else None
        evidence = child.evidence_store.load_run(child.context.run_id)
        research_root = child.context.run_root.parent / "research"
        bound_snapshot_path = child.context.metadata.get("research_snapshot_path")
        bound_snapshot_hash = child.context.metadata.get("research_snapshot_hash")
        snapshot_ref = (
            load_research_snapshot(research_root, bound_snapshot_path)
            if bound_snapshot_path
            else (load_research_snapshot(research_root) if bound_snapshot_hash is None else None)
        )
        snapshot_hash = str(child.context.metadata.get("research_snapshot_hash") or "none")
        if snapshot_ref is not None:
            snapshot, snapshot_dir = snapshot_ref
            if snapshot_hash not in {"none", snapshot.snapshot_hash}:
                raise ValueError(
                    f"research snapshot changed within loop: expected {snapshot_hash}, got {snapshot.snapshot_hash}"
                )
            snapshot_hash = snapshot.snapshot_hash
            child.context.metadata.update(
                {
                    "research_snapshot_hash": snapshot_hash,
                    "research_snapshot_path": snapshot_dir.resolve().as_posix(),
                    "research_snapshot_verified": True,
                }
            )
            contracts_path = snapshot_dir / "component_contracts.yaml"
            recipes_path = snapshot_dir / "recipes.yaml"
            paper_root = snapshot_dir
        else:
            contracts_path = ResourcePaths.COMPONENT_COMPATIBILITY
            recipes_path = ResourcePaths.RECIPE_BUNDLES
            paper_root = research_root
            child.context.metadata["research_snapshot_verified"] = False
        contracts = load_contracts(contracts_path) if contracts_path.exists() else []
        component_registry = (
            ComponentRegistry(contracts)  # type: ignore[arg-type]
            if snapshot_ref is not None
            else ComponentRegistry.from_path(child.context.component_path)
        )
        paper_registry = PaperRegistry(paper_root)
        recipe_registry = (
            RecipeRegistry.from_path(
                recipes_path,
                component_contracts=contracts if snapshot_ref is not None else (),
            )
            if recipes_path.exists()
            else RecipeRegistry()
        )
        policy_memory = PolicyMemoryStore(child.context.run_root)
        plan = PaperRecipePlanner().plan(
            error_facts=parent_facts,
            dataset_report=dataset_report,
            node_metrics=evidence.metric_records,
            policy_memory=policy_memory,
            paper_registry=paper_registry,
            component_registry=component_registry,
            recipe_registry=recipe_registry,
            tried_actions=_tried_action_ids(child.context.run_root, _base_auto_run_id(child.context.run_id)),
            training_budget={"profile": "pilot", "imgsz": 640},
            optimization_objective=load_optimization_objective(
                child.context.metadata.get("optimization_objective_path")
            ),
        )
        compatibility_snapshot = {
            "schema_version": "component_compatibility_snapshot.v1",
            "imgsz": 640,
            "research_snapshot_hash": snapshot_hash,
            "research_snapshot_verified": bool(child.context.metadata.get("research_snapshot_verified", False)),
            "components": {
                item.component_id: {
                    "maturity": item.maturity,
                    "can_execute": item.can_execute,
                    "implementation_path": item.implementation_path,
                    "adapter_class": item.adapter_class,
                    "fixed_imgsz_compatible": item.fixed_imgsz_compatible,
                }
                for item in contracts
            },
            "paper_registry_count": len(paper_registry.list()),
            "available_recipes": [item.recipe_id for item in recipe_registry.list()],
        }
        compatibility_for_critic = {
            item.component_id: {
                "compatible": item.fixed_imgsz_compatible is not False,
                "blocked_by": [] if item.fixed_imgsz_compatible is not False else ["fixed_imgsz_incompatible"],
            }
            for item in contracts
        }
        memory_records = policy_memory.read()
        recipe_critic_reports = []
        executable_pilot_policies: list[CandidatePolicy] = []
        for planned in [*plan.selected_recipes, *plan.deferred_recipes, *plan.rejected_recipes]:
            recipe = recipe_registry.get(planned.recipe_id, planned.version)
            if recipe is None:
                continue
            report = RecipeCritic().critique(
                recipe,
                error_facts=parent_facts,
                component_contracts=contracts,
                compatibility=compatibility_for_critic,
                local_evidence=memory_records,
            )
            recipe_critic_reports.append(report.model_dump(mode="json"))
            if planned.decision == "selected" and report.accepted and isinstance(recipe, AtomicRecipe):
                executable_pilot_policies.append(
                    _candidate_policy_from_recipe(child, recipe, parent_facts, planned.utility)
                )
        payload = {
            "schema_version": "paper_recipe_plan.v1",
            "research_snapshot_hash": snapshot_hash,
            "research_snapshot_verified": bool(child.context.metadata.get("research_snapshot_verified", False)),
            "research_snapshot_path": child.context.metadata.get("research_snapshot_path"),
            "llm_status": "deferred_to_unified_decision_bundle",
            "llm_proposal": None,
            "rule_plan": plan.model_dump(mode="json"),
            "recipe_critic_reports": recipe_critic_reports,
            "executable_pilot_policies": [
                policy.model_dump(mode="json") for policy in executable_pilot_policies
            ],
            "decision_context_inputs": {
                "paper_candidates": [
                    item.model_dump(mode="json")
                    for item in [*plan.selected_recipes, *plan.deferred_recipes, *plan.rejected_recipes]
                ],
                "executable_adapters": [
                    item.adapter_class
                    for item in contracts
                    if item.can_execute and item.adapter_class
                ],
                "component_maturity": {
                    item.component_id: item.maturity for item in contracts
                },
                "compatibility": compatibility_snapshot["components"],
                "paper_registry_count": len(paper_registry.list()),
            },
            "paper_claims_are_prior_only": True,
        }
        write_yaml(plan_path, payload)
        write_yaml(compatibility_path, compatibility_snapshot)
        component_ids = {
            component_id
            for planned in [*plan.selected_recipes, *plan.deferred_recipes]
            for component_id in _recipe_component_ids(planned.recipe_id, planned.version, recipe_registry)
        }
        for component_id in sorted(component_ids):
            reproduction = ReproductionPipeline(
                child.context.run_dir,
                "recipe_registry",
                component_id,
                policy_path=ResourcePaths.REPRODUCTION_POLICY,
            )
            safe_id = component_id.replace(".", "_").replace("-", "_")
            reproduction.state_path = child.context.artifact_path(f"reproduction_state_{safe_id}.yaml")
            reproduction.initialize(evidence={"paper_record": bool(paper_registry.list())})
            state_paths.append(reproduction.state_path)
        child.context.to_yaml()
        child.context.to_json()
        return {
            "paper_recipe_plan": plan_path,
            "component_compatibility": compatibility_path,
            "reproduction_states": state_paths,
            "research_snapshot_hash": snapshot_hash,
        }
    except Exception as exc:
        write_yaml(plan_path, {
            "schema_version": "paper_recipe_plan.v1",
            "status": "failed_fallback_to_rule_loop",
            "error": str(exc),
            "diagnosis_path": diagnosis_path.as_posix(),
            "research_snapshot_hash": child.context.metadata.get("research_snapshot_hash", "none"),
            "research_snapshot_path": child.context.metadata.get("research_snapshot_path"),
            "research_snapshot_verified": bool(child.context.metadata.get("research_snapshot_verified", False)),
            "llm_status": "deferred_to_unified_decision_bundle",
            "llm_proposal": None,
            "paper_claims_are_prior_only": True,
            "rule_planner_continues": True,
        })
        write_yaml(
            compatibility_path,
            {
                "schema_version": "component_compatibility_snapshot.v1",
                "status": "unavailable",
                "error": str(exc),
                "imgsz": 640,
                "research_snapshot_hash": child.context.metadata.get("research_snapshot_hash", "none"),
            },
        )
        return {"paper_recipe_plan": plan_path, "component_compatibility": compatibility_path, "reproduction_states": state_paths}


def _recipe_component_ids(recipe_id: str, version: str, registry: RecipeRegistry) -> list[str]:
    recipe = registry.get(recipe_id, version)
    return list(recipe.component_ids) if recipe is not None else []


def _candidate_policy_from_recipe(
    child: LoopOrchestrator,
    recipe: AtomicRecipe,
    error_facts: list[ErrorFact],
    utility: float,
) -> CandidatePolicy:
    """Translate an accepted recipe into the existing guarded policy boundary."""
    config = _training_config_from_context(child)
    model = config.model if config is not None else str(child.context.metadata.get("training_model") or "yolo26n.pt")
    expected = {
        key: value
        for key, value in recipe.expected_effects.items()
        if isinstance(value, (int, float))
    }
    target_facts = [
        fact.model_dump(mode="json")
        for fact in error_facts
        if any(
            all(
                getattr(fact, key, None) == value
                for key, value in target.items()
                if key in {"fact_type", "subject", "metric_name", "area", "class_name"} and value is not None
            )
            for target in recipe.target_error_facts
        )
    ]
    action_domain = "model" if recipe.component_ids else ("augmentation" if "augmentation" in recipe.primary_changed_variable else "data")
    return CandidatePolicy(
        policy_id=f"paper_recipe_{recipe.recipe_id}_{recipe.version.replace('.', '_')}",
        source="rule_engine",
        action_domain=action_domain,
        action_id=recipe.recipe_id,
        execution_action="run_training",
        base_model=model,
        scale="n",
        framework="ultralytics",
        components=list(recipe.component_ids),
        train_overrides={**recipe.train_overrides, "imgsz": 640, "target_actions": [recipe.recipe_id]},
        fixed_variables={**recipe.fixed_variables, "imgsz": 640},
        constraints=[
            PolicyConstraint(name="single_variable", value=True, hard=True),
            PolicyConstraint(name="fixed_imgsz", value=640, hard=True),
        ],
        target_error_facts=target_facts,
        expected_improvement={
            "expected_gain": expected or {metric: 0.1 for metric in recipe.target_metrics},
            "paper_prior_only": True,
            "recipe_id": recipe.recipe_id,
        },
        priority_hint=max(0.1, min(float(utility), 10.0)),
        expected_effect=[f"{key}: {value}" for key, value in recipe.expected_effects.items()],
        risk=recipe.implementation_risk if recipe.implementation_risk != "unknown" else "medium",
        rationale="Critic-approved atomic paper recipe; evaluator and pilot gates remain authoritative.",
    )


def _update_reproduction_after_round(
    child: LoopOrchestrator,
    parent: LoopOrchestrator,
    plan_path: Path | None,
    state_paths: list[Path],
    training_loop: TrainingLoopResult | None,
    assessments: list[CandidateExecutionAssessment],
) -> None:
    """Attach imported pilot evidence without overclaiming component maturity."""
    if not plan_path or not plan_path.is_file() or not state_paths or training_loop is None:
        return
    raw_plan = read_yaml(plan_path)
    policies = raw_plan.get("executable_pilot_policies", [])
    if not isinstance(policies, list):
        return
    executed_recipe_ids = {
        str(item.action_id)
        for item in assessments
        if item.execution_class == "executable" and item.action_id
    }
    executed_components = {
        str(component_id)
        for policy in policies
        if isinstance(policy, dict) and str(policy.get("action_id")) in executed_recipe_ids
        for component_id in policy.get("components", [])
    }
    if not executed_components:
        return
    evidence = child.evidence_store.load_run(child.context.run_id)
    objective = load_optimization_objective(child.context.metadata.get("optimization_objective_path"))
    protocol_hash = objective.baseline_protocol_hash if objective is not None else None
    current_node_ids = _executed_round_node_ids(child, executed_recipe_ids)
    if not current_node_ids:
        return
    selected_current = select_metric_evidence(
        evidence.metric_records,
        EvidenceSelector(
            current_run_id=child.context.run_id,
            current_run_only=True,
            current_node_only=sorted(current_node_ids),
            inherited_context=False,
            baseline_reference=False,
            same_protocol_hash=protocol_hash,
            same_dataset_manifest=child.context.dataset_manifest_sha256,
            same_seed=child.context.seed,
            verified=True,
        ),
    ).records
    verified_metrics = {
        item.metric_name: item.value
        for item in selected_current
        if item.verified and item.value is not None
    }
    if not verified_metrics:
        return
    paired_deltas = [
        delta
        for item in selected_current
        for _, delta in [paired_metric_delta(item, evidence.metric_records)]
        if delta is not None
    ]
    local_delta = {item.metric_name: item.paired_delta for item in paired_deltas}
    for state_path in state_paths:
        pipeline = ReproductionPipeline(
            child.context.run_dir,
            "recipe_registry",
            "unknown",
            policy_path=ResourcePaths.REPRODUCTION_POLICY,
        )
        pipeline.state_path = state_path
        state = pipeline.load()
        if state.component_id not in executed_components:
            continue
        state.evidence.update(
            {
                "pilot_evidence_imported": True,
                "pilot_metric_names": sorted(verified_metrics),
                "pilot_training_completed": bool(training_loop.completed),
            }
        )
        state.local_delta.update(local_delta)
        if state.status == "pilot_running" and training_loop.completed:
            state.evidence["pilot_evidence"] = True
            try:
                pipeline.transition("pilot_reproduced", evidence=state.evidence, local_delta=local_delta)
                continue
            except ValueError as exc:
                state.last_error = f"pilot evidence imported but maturity prerequisites remain: {exc}"
        pipeline.save(state)


def _executed_round_node_ids(child: LoopOrchestrator, executed_recipe_ids: set[str]) -> set[str]:
    """Return canonical execution nodes for the recipes completed in this child run."""
    path = child.context.artifact_path("round_execution_plan.yaml")
    if not path.is_file():
        return set()
    plan = RoundExecutionPlan.from_yaml(path)
    policy_path = child.context.artifact_path("policy_evaluation.yaml")
    if not policy_path.is_file():
        return set()
    report = LoopPolicyEvaluationReport.model_validate(read_yaml(policy_path))
    candidate_ids = {
        item.candidate_config.candidate_id
        for item in report.evaluations
        if item.candidate_config is not None
        and any(recipe_id in item.policy_id for recipe_id in executed_recipe_ids)
    }
    return {
        assignment.execution_node_id
        for assignment in plan.assignments
        if assignment.candidate_id in candidate_ids and assignment.status in {"completed", "active"}
    }


def _inherit_parent_metric_evidence(child: LoopOrchestrator, parent: LoopOrchestrator) -> None:
    """Copy parent metric records into the child as inherited context evidence."""
    inherited_from = str(child.context.metadata.get("inherited_metric_evidence_from") or "")
    inheritance_version = int(child.context.metadata.get("inherited_metric_evidence_version") or 0)
    if inherited_from == parent.context.run_id and inheritance_version >= 3:
        return
    parent_metrics_path = parent.context.run_dir / "metrics.json"
    parent_metrics = read_json(parent_metrics_path) if parent_metrics_path.is_file() else {}
    parent_records = _inheritable_lineage_metric_records(parent)
    if parent_metrics:
        child.evidence_store.log_metrics(child.context.run_id, parent_metrics)
    if parent_records:
        child.evidence_store.log_metric_records(
            child.context.run_id,
            [
                record.model_copy(
                    update={
                        "run_id": child.context.run_id,
                        "origin_run_id": record.origin_run_id or record.run_id or parent.context.run_id,
                        "evidence_role": "baseline_reference",
                        "inheritance_depth": max(1, record.inheritance_depth + 1),
                        "source": (
                            record.source
                            if str(record.source).startswith("inherited:")
                            else f"inherited:{parent.context.run_id}:{record.source}"
                        ),
                        "validator": record.validator or "inherited_parent_evidence",
                    }
                )
                for record in parent_records
            ],
        )
    child.context.metadata["inherited_metric_evidence_from"] = parent.context.run_id
    child.context.metadata["inherited_metric_evidence_version"] = 3
    child.context.to_yaml(child.context.run_dir / "run_context.yaml")
    child.event_log.append(
        run_id=child.context.run_id,
        event_type="stage_completed",
        stage="init",
        status="completed",
        message="Inherited parent metric evidence for auto optimization planning.",
        details={
            "parent_run_id": parent.context.run_id,
            "run_metric_count": len(parent_metrics),
            "metric_record_count": len(parent_records),
        },
    )


def _inheritable_parent_metric_records(path: Path, parent_run_id: str) -> list[MetricEvidence]:
    if not path.is_file():
        return []
    selected: dict[tuple[str, str, str, str, str], MetricEvidence] = {}
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            try:
                raw = read_json_line(text)
            except ValueError:
                continue
            if not _is_inheritable_metric_record(raw):
                continue
            try:
                record = MetricEvidence.model_validate(raw)
            except ValueError:
                continue
            key = (
                record.candidate_id,
                record.node_id,
                record.dataset_version,
                record.split,
                record.metric_name,
            )
            selected[key] = record
    return list(selected.values())


def _inheritable_lineage_metric_records(parent: LoopOrchestrator) -> list[MetricEvidence]:
    """Return nearest verified metric evidence across the parent run lineage."""
    selected: dict[tuple[str, str, str], MetricEvidence] = {}
    current: LoopOrchestrator | None = parent
    visited: set[str] = set()
    while current is not None and current.context.run_id not in visited:
        visited.add(current.context.run_id)
        records = _inheritable_parent_metric_records(
            current.context.run_dir / "metrics_by_node.jsonl",
            current.context.run_id,
        )
        for record in records:
            key = (record.dataset_version, record.split, record.metric_name)
            selected.setdefault(key, record)
        parent_dir = current.context.metadata.get("parent_run_dir")
        if not isinstance(parent_dir, str) or not parent_dir:
            break
        path = Path(parent_dir)
        if not (path / "run_context.yaml").is_file():
            break
        current = LoopOrchestrator.from_run_dir(path)
    return list(selected.values())


def read_json_line(text: str) -> dict[str, Any]:
    import json

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("metric record must be a mapping")
    return data


def _is_inheritable_metric_record(raw: dict[str, Any]) -> bool:
    if raw.get("verified") is False:
        return False
    name = str(raw.get("metric_name", ""))
    if not name or raw.get("value") is None:
        return False
    if name.startswith("runtime_stream_"):
        return False
    if name.startswith("batch_tuning_b"):
        return False
    if name.startswith(("per_class_ap/", "per_class_ar/", "coco/")):
        return True
    return name in {
        "ap_small",
        "ap_medium",
        "ap_large",
        "map50_95",
        "map50",
        "precision",
        "recall",
        "model_size_mb",
        "latency_ms",
        "imgsz",
        "epochs",
        "best_epoch",
        "train_box_loss",
        "train_cls_loss",
        "train_dfl_loss",
        "val_box_loss",
        "val_cls_loss",
        "val_dfl_loss",
        "runtime_batch_size",
        "runtime_cache_mode",
        "runtime_dataloader_workers",
        "runtime_avg_it_per_sec",
        "runtime_max_it_per_sec",
        "runtime_epoch_time_seconds",
        "runtime_avg_gpu_util_percent",
        "runtime_max_gpu_memory_used_mb",
        "runtime_dataloader_wait_warning",
        "batch_tuning_applied",
        "batch_tuning_selected_batch",
        "batch_tuning_best_it_per_sec",
        "batch_tuning_trial_count",
        "batch_tuning_oom_trials",
        "data_cache_policy_applied",
        "data_cache_selected_cache",
        "data_cache_selected_workers",
        "data_cache_dataset_size_mb",
        "data_cache_storage_kind",
        "fast_baseline_gate_ok",
        "fast_baseline_gate_profile",
        "fast_baseline_gate_stage",
        "fast_baseline_confirmed_seed_count",
        "fast_baseline_pilot_passed",
        "training_budget_profile",
        "fast_baseline_seed",
        "execution_duration_seconds",
        "execution_return_code",
    }


def _evidence_status_from_parent(child: LoopOrchestrator, parent_facts: list[ErrorFact]) -> dict[str, str]:
    """Build evidence status for the diagnosis engine from inherited parent evidence."""
    evidence = child.evidence_store.load_run(child.context.run_id)
    present = _present_metric_names(evidence)
    status: dict[str, str] = {name: "present" for name in present}
    if child.context.artifact_path("dataset_report.json").is_file():
        status["dataset_report"] = "present"
    if child.context.artifact_path("annotation_advice.json").is_file():
        status["label_quality_report"] = "present"
    if parent_facts:
        status.update(
            {
                "error_facts": "present",
                "localization_error_rate": "present",
                "false_negative_count": "present",
                "false_positive_count": "present",
                "class_confusion_pairs": "present",
                "confusion_matrix": "present",
            }
        )
    for fact in parent_facts:
        if fact.metric_name:
            status[str(fact.metric_name)] = "present"
        if fact.fact_type == "area_metric" and fact.area:
            status[f"ap_{fact.area}"] = "present"
        if fact.metric_name == "per_class_ap":
            status["per_class_ap"] = "present"
        if fact.metric_name == "per_class_ar":
            status["per_class_ar"] = "present"
    return status


def _present_metric_names(evidence: Evidence) -> set[str]:
    names = {name for name, value in evidence.metrics.items() if value is not None}
    names.update(record.metric_name for record in evidence.metric_records if record.value is not None and record.verified)
    return names


def _observations_from_error_facts(
    facts: list[ErrorFact],
    parent_next_round: dict[str, Any],
) -> list[DetectionErrorObservation]:
    focus = parent_next_round.get("current_round_focus", [])
    focus_keys = {
        _focus_key(item)
        for item in focus
        if isinstance(item, dict)
    }
    selected = [
        fact
        for fact in sorted(facts, key=_fact_rank)
        if fact.severity in {"high", "medium"}
        and (not focus_keys or _fact_key(fact) in focus_keys)
    ]
    selected_keys = {_fact_key(fact) for fact in selected}
    for fact in sorted(facts, key=_fact_rank):
        if len(selected) >= 8:
            break
        if fact.severity not in {"high", "medium"} or _fact_key(fact) in selected_keys:
            continue
        if fact.fact_type not in {"background_false_positive_class", "class_confusion_pair"}:
            continue
        selected.append(fact)
        selected_keys.add(_fact_key(fact))
    if not selected:
        selected = [fact for fact in sorted(facts, key=_fact_rank) if fact.severity in {"high", "medium"}]
    observations: list[DetectionErrorObservation] = []
    for fact in selected[:8]:
        observations.append(
            DetectionErrorObservation(
                error_type=_error_type_for_fact(fact),
                count=max(int(fact.count or 1), 1),
                severity=fact.severity,
                notes=[
                    f"fact_type={fact.fact_type}",
                    f"subject={fact.subject}",
                    f"metric={fact.metric_name or 'unknown'}",
                    f"actions={','.join(fact.action_candidates)}",
                ],
            )
        )
    return observations or [
        DetectionErrorObservation(
            error_type="out_of_distribution_miss",
            count=1,
            severity="medium",
            notes=["fallback observation because no medium/high error facts were selected"],
        )
    ]


def _error_type_for_fact(fact: ErrorFact) -> DetectionErrorType:
    if fact.fact_type in {"area_metric", "subset_performance"} and fact.area == "small":
        return "small_object_miss"
    if fact.fact_type == "false_negative_heavy_class":
        return "out_of_distribution_miss"
    if fact.fact_type == "localization_heavy_class":
        return "loose_box"
    if fact.fact_type == "background_false_positive_class":
        return "background_confusion"
    if fact.fact_type == "class_confusion_pair":
        return "class_confusion"
    if fact.fact_type in {"class_low_ap", "per_class_metric"}:
        if fact.metric_name == "per_class_ar":
            return "long_tail_bias"
        return "out_of_distribution_miss"
    return "out_of_distribution_miss"


def _fact_rank(fact: ErrorFact) -> tuple[int, int, float]:
    severity = {"high": 0, "medium": 1, "low": 2}[fact.severity]
    rank = fact.rank if fact.rank is not None else 999
    value = float(fact.value) if isinstance(fact.value, (int, float)) and not isinstance(fact.value, bool) else 999.0
    return (severity, rank, value)


def _focus_key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(item.get("fact_type", "")),
        str(item.get("subject", "")),
        str(item.get("class_name", "")),
        str(item.get("area", "")),
        str(item.get("metric_name", "")),
    )


def _fact_key(fact: ErrorFact) -> tuple[str, str, str, str, str]:
    return (
        fact.fact_type,
        fact.subject,
        fact.class_name or "",
        fact.area or "",
        fact.metric_name or "",
    )


def _infer_training_model(orchestrator: LoopOrchestrator) -> str | None:
    """Infer the real model used by a run so child rounds do not fall back to config defaults."""
    value = orchestrator.context.metadata.get("training_model")
    if isinstance(value, str) and value.strip():
        return value.strip()
    plan_path = orchestrator.context.artifact_path("experiment_plan.yaml")
    if plan_path.is_file():
        try:
            plan = ExperimentPlan.from_yaml(plan_path)
        except Exception:
            plan = None
        if plan is not None:
            for node in plan.nodes:
                model = node.candidate_config.base_model
                if model and model.lower() not in {"yolo11n", "yolo11s"}:
                    return model
                command_spec = node.command_spec
                if command_spec is not None:
                    for arg in command_spec.args:
                        if str(arg).startswith("model="):
                            return str(arg).split("=", 1)[1]
    for args_path in orchestrator.context.run_root.rglob("args.yaml"):
        text = args_path.as_posix().lower()
        if orchestrator.context.run_id.lower() not in text:
            continue
        raw = read_yaml(args_path)
        model = raw.get("model") if isinstance(raw, dict) else None
        if isinstance(model, str) and model.strip():
            return model.strip()
    return None


def _training_config_from_context(child: LoopOrchestrator) -> UltralyticsTrainingConfig | None:
    raw_path = child.context.metadata.get("training_config_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_file():
        return None
    profile = child.context.metadata.get("training_profile")
    config = UltralyticsTrainingConfig.from_yaml(
        path,
        budget_profile=profile if profile in {"debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"} else None,
    )
    model = child.context.metadata.get("training_model")
    if isinstance(model, str) and model.strip():
        config = config.model_copy(update={"model": model.strip()})
    return config


def _assess_policy_evaluation(path: Path) -> list[CandidateExecutionAssessment]:
    if not path.is_file():
        return []
    report = LoopPolicyEvaluationReport.model_validate(read_yaml(path))
    return assess_candidate_execution(report)


def _required_component_adapters(components: list[str]) -> list[str]:
    adapters: list[str] = []
    for component in components:
        if component.startswith(ADAPTER_REQUIRED_COMPONENT_PREFIXES):
            adapters.append(f"component_adapter:{component}")
    return adapters


def _unsupported_train_overrides(overrides: dict[str, Any]) -> list[str]:
    unsupported = []
    for key in overrides:
        if key in HARNESS_ONLY_TRAIN_OVERRIDE_KEYS:
            continue
        if key in SAFE_ULTRALYTICS_OVERRIDE_KEYS:
            continue
        if key == "imgsz":
            continue
        unsupported.append(str(key))
    return unsupported


def _executable_nodes(path: Path, assessments: list[CandidateExecutionAssessment]) -> list[ExperimentNode]:
    if not path.is_file():
        return []
    executable_candidate_ids = {
        item.candidate_id
        for item in assessments
        if item.execution_class == "executable" and item.candidate_id is not None
    }
    plan = ExperimentPlan.from_yaml(path)
    return [
        node
        for node in plan.nodes
        if node.candidate_config.candidate_id in executable_candidate_ids
        or bool(node.command_spec and node.command_spec.metadata.get("matched_baseline_control"))
    ]


def _write_filtered_experiment_plan(
    child: LoopOrchestrator,
    executable_nodes: list[ExperimentNode],
    assessments: list[CandidateExecutionAssessment],
) -> Path:
    source_path = child.context.artifact_path("experiment_plan.yaml")
    round_plan_path = child.context.artifact_path("round_execution_plan.yaml")
    if round_plan_path.is_file():
        round_plan = RoundExecutionPlan.from_yaml(round_plan_path)
        executable_ids = {node.node_id for node in executable_nodes}
        round_plan.execution_nodes = [node for node in round_plan.execution_nodes if node.node_id in executable_ids]
        round_plan.assignments = [
            assignment
            for assignment in round_plan.assignments
            if assignment.status != "active" or assignment.execution_node_id in executable_ids
        ]
        round_plan.critic_results.extend(
            [item.model_dump(mode="json") for item in assessments if item.execution_class != "executable"]
        )
        round_plan.scheduler_mode = "external_asha"
        round_plan.status = "ready" if round_plan.execution_nodes else "blocked"
        round_plan.blocked_reason = "" if round_plan.execution_nodes else "no executable guarded candidates"
        round_plan.to_yaml(round_plan_path)
        round_plan.experiment_projection().to_yaml(source_path)
        write_yaml(child.context.artifact_path("budget_optimization.yaml"), round_plan.budget_projection())
        write_yaml(child.context.artifact_path("ablation_plan.yaml"), round_plan.ablation_projection())
        child.evidence_store.log_artifact_manifest(
            run_id=child.context.run_id,
            name="round_execution_plan",
            artifact_path=round_plan_path,
            producer_stage="auto_optimization_loop",
        )
        return source_path
    original = ExperimentPlan.from_yaml(source_path)
    filtered = ExperimentPlan(
        plan_id=f"{child.context.run_id}_auto_executable_pilot_plan",
        nodes=executable_nodes,
        metadata={
            **original.metadata,
            "source": "AutoOptimizationLoopDriver",
            "original_plan_id": original.plan_id,
            "original_node_count": len(original.nodes),
            "executable_node_count": len(executable_nodes),
            "candidate_execution_assessments": [item.model_dump(mode="json") for item in assessments],
        },
    )
    filtered.metadata["plan_hash"] = filtered.plan_hash()
    filtered.to_yaml(source_path)
    child.evidence_store.log_artifact_manifest(
        run_id=child.context.run_id,
        name="experiment_plan",
        artifact_path=source_path,
        producer_stage="auto_optimization_loop",
    )
    return source_path


def _empty_round(
    *,
    round_index: int,
    parent: LoopOrchestrator,
    status: Literal["completed", "blocked", "failed", "skipped"],
    stop_reason: str,
) -> AutoRoundResult:
    path = parent.context.artifact_path(f"auto_round_{round_index}_blocked.yaml")
    result = AutoRoundResult(
        round_index=round_index,
        run_id=parent.context.run_id,
        run_dir=parent.context.run_dir,
        parent_run_id=parent.context.run_id,
        status=status,
        stop_reason=stop_reason,
        auto_round_summary_path=path,
    )
    write_yaml(path, result.model_dump(mode="json"))
    return result


def _existing_or_none(path: Path) -> Path | None:
    return path if path.exists() else None


def _write_final_outputs(result: AutoOptimizationResult) -> None:
    result.summary_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations = _full_candidate_recommendations(result)
    write_yaml(result.full_candidate_recommendations_path, recommendations)
    result.summary_path.write_text(_summary_markdown(result, recommendations), encoding="utf-8")


def _refresh_objective_status(context: Any, objective: OptimizationObjective) -> OptimizationObjectiveStatus:
    """Evaluate and persist the single objective used by the automatic loop."""
    status = evaluate_optimization_objective(
        objective,
        run_root=context.run_root,
        base_run_id=context.run_id,
    )
    path = context.artifact_path("optimization_objective_status.yaml")
    write_yaml(path, status.model_dump(mode="json"))
    EvidenceStore(context.run_root).log_artifact_manifest(
        run_id=context.run_id,
        name="optimization_objective_status",
        artifact_path=path,
        producer_stage="auto_optimization_loop",
    )
    return status


def _full_candidate_recommendations(result: AutoOptimizationResult) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    screening: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    asha_study = _load_asha_study(result)
    if asha_study is not None:
        for trial in asha_study.trials:
            seen_candidates.add(trial.candidate_id)
            latest = max(trial.observations, key=lambda item: item.created_at) if trial.observations else None
            item = {
                "source_run_id": trial.source_run_id,
                "candidate_id": trial.candidate_id,
                "node_id": trial.source_node.node_id,
                "asha_status": trial.status,
                "latest_stage": latest.stage_id if latest is not None else None,
                "latest_paired_delta": latest.paired_delta if latest is not None else None,
                "target_error_improved_count": latest.target_error_improved_count if latest is not None else 0,
            }
            if trial.status in {"full_pending_confirmation", "confirmation_pending", "confirmed"}:
                items.append(
                    {
                        **item,
                        "next_profile": "candidate_full",
                        "promotion_status": trial.status,
                        "requires": (
                            []
                            if trial.status == "confirmed"
                            else ["explicit --confirm-full-run", "remaining matched full seeds"]
                        ),
                        "command_hint": (
                            "rerun the same yolo-agent train command with "
                            f"--run-id {result.base_run_id} --confirm-full-run"
                        ),
                    }
                )
            else:
                screening.append(
                    {
                        **item,
                        "promotion_status": "screening_only",
                        "reason": trial.eliminated_reason or f"ASHA trial remains {trial.status}",
                    }
                )
    for round_result in result.rounds:
        for assessment in round_result.candidate_assessments:
            if assessment.execution_class != "executable":
                continue
            candidate_key = str(assessment.candidate_id or assessment.policy_id)
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            item = {
                "source_run_id": round_result.run_id,
                "candidate_id": assessment.candidate_id,
                "node_id": assessment.node_id,
                "next_profile": "candidate_full",
            }
            objective = result.objective_status
            objective_selected = bool(
                objective is not None
                and objective.target_reached
                and objective.guardrails_passed
                and assessment.candidate_id == objective.best_candidate_id
            )
            if objective is None or objective_selected:
                items.append(
                    {
                        **item,
                        "promotion_status": (
                            "objective_confirmed"
                            if objective is not None and objective.success
                            else "objective_target_reached_pending_confirmation"
                            if objective_selected
                            else "not_promoted"
                        ),
                        "objective_hash": objective.objective_hash if objective is not None else None,
                        "observed_delta": objective.observed_delta if objective is not None else None,
                        "required_delta": objective.required_delta if objective is not None else None,
                        "requires": [
                            "candidate_promotion_gate_passed",
                            "baseline_trusted",
                            "objective_confirmation_seeds",
                            "objective_confidence_interval",
                            "explicit --confirm-full-run",
                        ],
                        "command_hint": (
                            "rerun the same yolo-agent train command with "
                            f"--run-id {result.base_run_id} --confirm-full-run"
                        ),
                    }
                )
            else:
                screening.append(
                    {
                        **item,
                        "promotion_status": "screening_only",
                        "reason": "candidate has not reached the persisted optimization objective",
                    }
                )
    repeated = _repeated_executable_candidates(result)
    return {
        "schema_version": "full_candidate_recommendations.v2",
        "base_run_id": result.base_run_id,
        "stopped_reason": result.stopped_reason,
        "full_run_started": False,
        "recommendations": items,
        "objective_status": result.objective_status.model_dump(mode="json") if result.objective_status else None,
        "screening_results": screening,
        "asha": _asha_summary(asha_study),
        "not_ready_reason": (
            "ASHA survivors are ready for explicit full confirmation or are already confirmed."
            if items
            else "Executable candidates remain screening-only because the objective or guard metrics are not satisfied."
            if screening
            else "No executable candidate survived the guarded pilot loop."
        ),
        "repeated_executable_candidates": repeated,
        "adapter_required": [
            {
                "round": round_result.round_index,
                "run_id": round_result.run_id,
                **assessment.model_dump(mode="json"),
            }
            for round_result in result.rounds
            for assessment in round_result.candidate_assessments
            if assessment.execution_class == "adapter_required"
        ],
        "recommendation_only": [
            {
                "round": round_result.round_index,
                "run_id": round_result.run_id,
                **assessment.model_dump(mode="json"),
            }
            for round_result in result.rounds
            for assessment in round_result.candidate_assessments
            if assessment.execution_class == "recommendation_only"
        ],
    }


def _load_asha_study(result: AutoOptimizationResult) -> ASHAStudy | None:
    path = result.asha_state_path
    if path is None or not path.is_file():
        return None
    return ASHAStudy.from_yaml(path)


def _asha_summary(study: ASHAStudy | None) -> dict[str, Any] | None:
    if study is None:
        return None
    counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    for trial in study.trials:
        counts[trial.status] = counts.get(trial.status, 0) + 1
        for observation in trial.observations:
            stage_counts[observation.stage_id] = stage_counts.get(observation.stage_id, 0) + 1
    return {
        "study_id": study.study_id,
        "trial_count": len(study.trials),
        "status_counts": counts,
        "observation_counts": stage_counts,
        "reduction_policy": "pilot_3 cohort eta=3; pilot_10 requires target error improvement; full uses 3 matched seeds",
    }


def _repeated_executable_candidates(result: AutoOptimizationResult) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for round_result in result.rounds:
        for assessment in round_result.candidate_assessments:
            if assessment.execution_class != "executable":
                continue
            key = str(assessment.candidate_id or assessment.policy_id)
            item = counts.setdefault(
                key,
                {
                    "candidate_id": assessment.candidate_id,
                    "action_id": assessment.action_id,
                    "count": 0,
                    "rounds": [],
                },
            )
            item["count"] += 1
            item["rounds"].append(round_result.round_index)
    return [item for item in counts.values() if int(item["count"]) > 1]


def _summary_markdown(result: AutoOptimizationResult, recommendations: dict[str, Any]) -> str:
    lines = [
        "# Auto Optimization Summary",
        "",
        f"- Base run: `{result.base_run_id}`",
        f"- Requested rounds: {result.requested_rounds}",
        f"- Executed training: {result.executed}",
        f"- Stop reason: `{result.stopped_reason}`",
        f"- Full run started: false",
        "",
        "## Rounds",
        "",
    ]
    if result.objective_status is not None:
        objective = result.objective_status
        lines[7:7] = [
            f"- Objective metric: `{objective.primary_metric}`",
            f"- Objective progress: baseline={objective.baseline_value} best={objective.best_value} "
            f"delta={objective.observed_delta} required={objective.required_delta}",
            f"- Objective confidence: seeds={objective.candidate_seed_count} "
            f"CI=[{objective.confidence_interval_low}, {objective.confidence_interval_high}]",
            f"- Objective budget: used={objective.gpu_hours_used}h remaining={objective.gpu_budget_remaining}h",
            f"- Objective guards: latency={objective.latency_regression} "
            f"size={objective.model_size_regression} passed={objective.guardrails_passed}",
            "",
        ]
    asha = recommendations.get("asha")
    if isinstance(asha, dict):
        lines.extend(
            [
                "## ASHA Budget",
                "",
                f"- Trials: {asha.get('trial_count', 0)}",
                f"- Status counts: `{asha.get('status_counts', {})}`",
                f"- Rung observations: `{asha.get('observation_counts', {})}`",
                f"- Policy: {asha.get('reduction_policy', '')}",
                "",
            ]
        )
    if not result.rounds:
        lines.append("- No automatic rounds ran.")
    for round_result in result.rounds:
        counts = {
            "executable": round_result.executable_count,
            "adapter_required": sum(1 for item in round_result.candidate_assessments if item.execution_class == "adapter_required"),
            "recommendation_only": sum(1 for item in round_result.candidate_assessments if item.execution_class == "recommendation_only"),
        }
        lines.append(
            f"- Round {round_result.round_index}: `{round_result.run_id}` "
            f"status={round_result.status} stop={round_result.stop_reason} "
            f"executable={counts['executable']} adapter_required={counts['adapter_required']} "
            f"recommendation_only={counts['recommendation_only']}"
        )
    paper_summary = _paper_summary(result)
    lines.extend(["", "## Paper Intelligence", ""])
    lines.append(
        f"- Paper-derived recipes considered: {paper_summary['considered']}; "
        f"critic accepted: {paper_summary['accepted']}; local reproduction snapshots: {paper_summary['states']}."
    )
    if paper_summary["adopted"]:
        lines.append("- Adopted ideas: " + ", ".join(f"`{item}`" for item in paper_summary["adopted"]) + ".")
    else:
        lines.append("- Adopted ideas: none entered the executable pilot path.")
    lines.append("- Paper claims remain priors until local imported metrics support them.")
    if paper_summary["rejected"]:
        lines.append("- Rejected/deferred: " + "; ".join(paper_summary["rejected"][:8]) + ".")
    if paper_summary["reproduced"]:
        lines.append("- Locally reproduced components: " + ", ".join(f"`{item}`" for item in paper_summary["reproduced"]) + ".")
    else:
        lines.append("- Locally reproduced components: none confirmed by imported pilot evidence yet.")
    lines.extend(["", "## Local Contribution And Pareto", ""])
    lines.append(
        "- Recipe contribution remains possible for single-seed pilots; confirmed contribution requires "
        "a single-variable or justified coupled ablation with repeated seeds."
    )
    lines.append(
        "- Accuracy, latency, and model-size evidence stays in node-level metrics and the Pareto report; "
        "missing guard metrics keep a candidate out of full recommendations."
    )
    lines.extend(["", "## Full Candidate Recommendations", ""])
    recs = recommendations.get("recommendations", [])
    if not recs:
        lines.append("- No full candidates are recommended yet. Pilot evidence or adapters are still missing.")
    else:
        for item in recs:
            lines.append(
                f"- `{item.get('candidate_id')}` from `{item.get('source_run_id')}`: "
                f"{item.get('promotion_status', 'not_promoted')}; requires candidate promotion, "
                "trusted baseline, 3 seeds, and explicit full-run confirmation."
            )
    repeated = recommendations.get("repeated_executable_candidates", [])
    if repeated:
        lines.extend(["", "## Repetition Guard", ""])
        for item in repeated:
            lines.append(
                f"- `{item.get('candidate_id')}` repeated {item.get('count')} times; "
                "future rounds should try different pilot actions unless new evidence justifies retry."
            )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "The auto loop only runs pilot-safe executable candidates. Metadata-only components, label/data work, "
            "post-processing policies, and unsupported Ultralytics overrides are recorded as recommendations or "
            "adapter-required items instead of being fake-trained.",
            "",
        ]
    )
    return "\n".join(lines)


def _paper_summary(result: AutoOptimizationResult) -> dict[str, Any]:
    considered = 0
    accepted = 0
    adopted: list[str] = []
    rejected: list[str] = []
    reproduced: list[str] = []
    state_count = 0
    for round_result in result.rounds:
        if round_result.paper_recipe_plan_path and round_result.paper_recipe_plan_path.is_file():
            raw = read_yaml(round_result.paper_recipe_plan_path)
            reports = raw.get("recipe_critic_reports", [])
            if isinstance(reports, list):
                considered += len(reports)
                for report in reports:
                    if not isinstance(report, dict):
                        continue
                    recipe_id = str(report.get("recipe_id") or "unknown")
                    if report.get("accepted"):
                        accepted += 1
                    else:
                        findings = report.get("findings", [])
                        reason = next(
                            (str(item.get("code")) for item in findings if isinstance(item, dict) and item.get("severity") == "error"),
                            str(report.get("decision") or "rejected"),
                        )
                        rejected.append(f"`{recipe_id}` ({reason})")
            policies = raw.get("executable_pilot_policies", [])
            if isinstance(policies, list):
                adopted.extend(
                    str(item.get("action_id") or item.get("policy_id"))
                    for item in policies
                    if isinstance(item, dict)
                )
        for state_path in round_result.reproduction_state_paths:
            if not state_path.is_file():
                continue
            state_count += 1
            state = read_yaml(state_path)
            if state.get("status") in {"pilot_reproduced", "full_reproduced", "confirmed_multi_seed"}:
                reproduced.append(str(state.get("component_id") or state_path.stem))
    return {
        "considered": considered,
        "accepted": accepted,
        "adopted": sorted(set(adopted)),
        "rejected": list(dict.fromkeys(rejected)),
        "reproduced": sorted(set(reproduced)),
        "states": state_count,
    }


__all__ = [
    "AutoOptimizationLoopDriver",
    "AutoOptimizationResult",
    "AutoRoundResult",
    "CandidateExecutionAssessment",
    "assess_candidate_execution",
]
