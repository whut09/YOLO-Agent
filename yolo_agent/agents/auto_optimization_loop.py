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
from yolo_agent.adapters.ultralytics.baseline_acceptance import BaselineAcceptanceResult
from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopEngine
from yolo_agent.agents.error_to_action import DetectionErrorObservation, DetectionErrorType
from yolo_agent.agents.exploration_diversity import (
    DiversityStopDecision,
    ExplorationDiversityPolicy,
    ExplorationHistoryEntry,
    ExplorationHistoryStore,
    evaluate_diversity_stop,
)
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
from yolo_agent.agents.loop_policy_evaluator import BudgetPolicy, LoopPolicyEvaluationReport
from yolo_agent.agents.orchestrator import LoopOrchestrator, TrainingLoopResult
from yolo_agent.agents.paper_recipe_planner import PaperRecipePlanner
from yolo_agent.agents.recipe_critic import RecipeCritic
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint
from yolo_agent.core.coco_error_selection import select_coco_error_facts
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueItem, ExecutionQueueStore
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.experiment_graph import Evidence, ExperimentNode, ExperimentPlan, MetricEvidence
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.full_run_consent import FullRunConsentDriver
from yolo_agent.core.evidence_selector import EvidenceSelector, select_metric_evidence
from yolo_agent.core.matched_baseline import paired_metric_delta
from yolo_agent.core.paired_experiment import build_paired_experiment_result
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.adapters import ComponentAdapterRegistry
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
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
from yolo_agent.core.run_protocol import RunProtocolVersion, build_run_protocol_version
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.recipes.schemas import AtomicRecipe, RecipeSpec
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.reproduction_pipeline import ReproductionPipeline
from yolo_agent.research.snapshot import load_research_snapshot
from yolo_agent.resources import ResourcePaths
from yolo_agent.tools.dataset_stats import DatasetReport


CandidateExecutionClass = Literal["executable", "recommendation_only", "adapter_required"]


def _trusted_full_run_authorization(
    context: Any,
    objective: OptimizationObjective | None,
    objective_status: OptimizationObjectiveStatus | None,
) -> tuple[bool, str]:
    """Require scoped consent plus a persisted trusted three-seed baseline."""
    if objective is None:
        return False, "full_run_objective_missing"
    consent = FullRunConsentDriver(context.run_dir).validate(
        run_id=context.run_id,
        objective=objective,
        dataset_manifest_sha256=context.dataset_manifest_sha256,
        objective_status=objective_status,
    )
    if not consent.allowed:
        return False, consent.reason
    acceptance_path = context.artifact_path("baseline_acceptance.json")
    if not acceptance_path.is_file():
        return False, "baseline_acceptance_missing"
    try:
        acceptance = BaselineAcceptanceResult.model_validate(read_json(acceptance_path))
    except (OSError, ValueError, TypeError):
        return False, "baseline_acceptance_invalid"
    if not acceptance.baseline_trusted or acceptance.accepted_seed_count < objective.confirmation_seeds:
        return False, "baseline_not_trusted"
    if acceptance.actual_dataset_manifest_sha256 != context.dataset_manifest_sha256:
        return False, "baseline_manifest_mismatch"
    return True, "trusted_full_run_authorized"

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
    diversity_outcomes: list[ExplorationHistoryEntry] = Field(default_factory=list)
    diversity_stop: DiversityStopDecision | None = None

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


def _diversity_policy(
    orchestrator: LoopOrchestrator,
    objective: OptimizationObjective | None,
) -> ExplorationDiversityPolicy:
    budget = BudgetPolicy.model_validate(orchestrator.policy.policy_budget)
    return ExplorationDiversityPolicy(
        component_family_cooldown_rounds=budget.component_family_cooldown_rounds,
        minimum_semantic_distance=budget.minimum_semantic_distance,
        no_improvement_patience=(
            objective.no_improvement_patience if objective is not None else budget.no_improvement_patience
        ),
        family_exhaustion_attempts=budget.family_exhaustion_attempts,
        minimum_improvement=budget.minimum_improvement,
        minimum_families_for_exhaustion_stop=budget.minimum_families_for_exhaustion_stop,
    )


def _record_exploration_outcomes(
    *,
    child: LoopOrchestrator,
    round_result: AutoRoundResult,
    history_store: ExplorationHistoryStore,
    policy: ExplorationDiversityPolicy,
) -> list[ExplorationHistoryEntry]:
    """Persist only actually executed candidate recipes and paired outcomes."""
    if (
        round_result.training_loop is None
        or not round_result.training_loop.completed
        or round_result.training_loop.executor == "dry-run"
        or round_result.policy_evaluation_path is None
        or not round_result.policy_evaluation_path.is_file()
    ):
        return []
    report = LoopPolicyEvaluationReport.model_validate(read_yaml(round_result.policy_evaluation_path))
    by_policy = {item.policy_id: item for item in report.evaluations}
    evidence = child.evidence_store.load_run(child.context.run_id)
    entries: list[ExplorationHistoryEntry] = []
    for assessment in round_result.candidate_assessments:
        if assessment.execution_class != "executable":
            continue
        evaluation = by_policy.get(assessment.policy_id)
        if (
            evaluation is None
            or evaluation.candidate_config is None
            or evaluation.experiment_node is None
            or not evaluation.recipe_fingerprint
            or not evaluation.component_family
        ):
            continue
        effect_delta = _executed_candidate_effect_delta(
            evidence,
            candidate_id=evaluation.candidate_config.candidate_id,
            node_id=evaluation.experiment_node.node_id,
        )
        entries.append(
            ExplorationHistoryEntry(
                run_id=child.context.run_id,
                round_index=round_result.round_index,
                policy_id=evaluation.policy_id,
                candidate_id=evaluation.candidate_config.candidate_id,
                recipe_fingerprint=evaluation.recipe_fingerprint,
                component_family=evaluation.component_family,
                changed_values=evaluation.changed_variables,
                semantic_tokens=sorted(
                    {evaluation.component_family, *evaluation.changed_variables.keys(), *evaluation.candidate_config.components}
                ),
                bucket=evaluation.budget_bucket or "exploration",
                effect_delta=effect_delta,
                improved=effect_delta is not None and effect_delta > policy.minimum_improvement,
                completed=True,
            )
        )
    additions = history_store.append(entries)
    if additions:
        base_run_id = history_store.path.parent.parent.name
        child.evidence_store.log_artifact_manifest(
            run_id=base_run_id,
            name="exploration_history",
            artifact_path=history_store.path,
            producer_stage="auto_optimization_loop",
        )
    return additions


def _executed_candidate_effect_delta(
    evidence: Evidence,
    *,
    candidate_id: str,
    node_id: str,
) -> float | None:
    for metric_name in ("coco_ap50_95", "map50_95"):
        candidates = [
            item for item in evidence.metric_records
            if item.candidate_id == candidate_id and item.node_id == node_id
            and item.metric_name == metric_name and item.evidence_role == "current_observation"
            and item.inheritance_depth == 0 and item.verified
            and isinstance(item.value, (int, float)) and not isinstance(item.value, bool)
        ]
        for candidate in sorted(candidates, key=lambda item: item.created_at, reverse=True):
            _, delta = paired_metric_delta(candidate, evidence.metric_records)
            if delta is not None:
                return delta.effect_delta
    return None


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
        diversity_policy = _diversity_policy(base_orchestrator, objective)
        diversity_store = ExplorationHistoryStore(base_context.artifact_path("exploration_history.jsonl"))
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
        full_run_authorized = False
        if confirm_full_run:
            full_run_authorized, authorization_reason = _trusted_full_run_authorization(
                base_context,
                objective,
                result.objective_status,
            )
            if not full_run_authorized:
                result.stopped_reason = authorization_reason
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
            asha_assignment = asha_scheduler.next_assignment(confirm_full_run=full_run_authorized)
            existing_round = _load_completed_round(child, round_index, parent.context.run_id, execute=execute)
            if existing_round is not None and asha_assignment is None:
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
                    asha_store=asha_store,
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
                    scheduler=asha_scheduler,
                )
            asha_store.save(asha_scheduler)
            if execute and asha_assignment is None:
                outcomes = _record_exploration_outcomes(
                    child=child,
                    round_result=round_result,
                    history_store=diversity_store,
                    policy=diversity_policy,
                )
                stop_decision = evaluate_diversity_stop(diversity_store.read(), diversity_policy)
                round_result = round_result.model_copy(
                    update={"diversity_outcomes": outcomes, "diversity_stop": stop_decision}
                )
                write_yaml(round_result.auto_round_summary_path, round_result.model_dump(mode="json"))
                child.evidence_store.log_artifact_manifest(
                    run_id=child.context.run_id,
                    name="auto_round_summary",
                    artifact_path=round_result.auto_round_summary_path,
                    producer_stage="auto_optimization_diversity",
                )
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
            if round_result.diversity_stop is not None and round_result.diversity_stop.should_stop:
                result.stopped_reason = round_result.diversity_stop.reason
                break
            if round_result.status != "completed" or round_result.stop_reason in {
                "no_guarded_candidates",
                "no_executable_candidates",
                "queue_blocked",
                "training_failed",
            }:
                result.stopped_reason = round_result.stop_reason or round_result.status
                break
            if round_result.stop_reason != "diversity_deferred":
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
        asha_store: ASHAStudyStore,
        execute: bool,
        executor: str,
        max_steps: int,
        auto_import: bool,
        total_rounds: int,
    ) -> AutoRoundResult:
        """Execute one cross-round ASHA promotion without generating a new recipe."""
        trial = scheduler.study.trial(assignment.trial_id)
        diagnosis_path = _ensure_loop_diagnosis_from_error_facts(child, parent_facts, parent_next_round)
        child.context.metadata["asha_budget_authority"] = True
        child.context.to_yaml()
        child.context.to_json()
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
            assignment_id=assignment.assignment_id,
        )
        _bind_child_run_protocol(
            child,
            round_plan,
            profile=("pilot" if assignment.stage_id.startswith("pilot") else "candidate_full"),
        )
        candidate_node = next(node for node in round_plan.execution_nodes if not _matched_baseline_node(node))
        if execute:
            scheduler.mark_running(
                assignment,
                run_id=child.context.run_id,
                node_id=candidate_node.node_id,
            )
            asha_store.save(scheduler)
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
            EventLog(child.context.events_path).append(
                run_id=child.context.run_id,
                event_type="auto_round_decision",
                status="completed" if observation.diagnosis_gate_passed is not False else "blocked",
                message=(
                    f"Diagnosis promotion gate {'passed' if observation.diagnosis_gate_passed else 'rejected'} "
                    f"{assignment.candidate_id} at {assignment.stage_id}."
                ),
                details={
                    "candidate_id": assignment.candidate_id,
                    "stage_id": assignment.stage_id,
                    "diagnosis_gate_passed": observation.diagnosis_gate_passed,
                    "diagnosis_checks": observation.diagnosis_checks,
                    "rejection_reasons": observation.promotion_rejection_reasons,
                    "paired_delta": observation.paired_delta,
                    "latency_regression": observation.latency_regression,
                    "model_size_regression": observation.model_size_regression,
                },
            )
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
        scheduler: ASHAScheduler,
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

        assessments = _assess_policy_evaluation(child)
        _log_candidate_decisions(child, round_index=round_index, total_rounds=total_rounds, assessments=assessments)
        if status == "completed":
            if not assessments:
                diversity_reason = _empty_diversity_round_reason(
                    child.context.artifact_path("policy_evaluation.yaml")
                )
                if diversity_reason == "family_exhaustion":
                    status = "blocked"
                    stop_reason = "family_exhaustion"
                elif diversity_reason == "diversity_deferred":
                    status = "completed"
                    stop_reason = "diversity_deferred"
                else:
                    status = "blocked"
                    stop_reason = "no_guarded_candidates"
            else:
                executable_nodes = _executable_nodes(child.context.artifact_path("experiment_plan.yaml"), assessments)
                if not executable_nodes:
                    status = "blocked"
                    stop_reason = "no_executable_candidates"
                else:
                    child.context.metadata["asha_budget_authority"] = True
                    child.context.to_yaml()
                    child.context.to_json()
                    if not execute:
                        stop_reason = "asha_registration_dry_run"
                    else:
                        registered = _register_guarded_pilot_trials(
                            scheduler,
                            child,
                            executable_nodes,
                        )
                        if registered:
                            stop_reason = "asha_candidates_registered"
                        else:
                            status = "blocked"
                            stop_reason = "no_new_asha_trials"

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
            protocol_hash=str(
                (node.command_spec.metadata if node.command_spec is not None else {}).get("run_protocol_hash")
                or orchestrator.context.run_protocol_hash
                or ""
            ),
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


def _enqueue_coco_evidence_recovery(
    orchestrator: LoopOrchestrator,
    nodes: list[ExperimentNode],
    results: list[PilotEvidenceCompletenessResult],
) -> ExecutionQueue:
    """Replace the queue with recovery-only actions for incomplete current-node evidence."""
    incomplete_ids = {item.node_id for item in results if not item.complete}
    items: list[ExecutionQueueItem] = []
    for node in nodes:
        if node.node_id not in incomplete_ids or node.command_spec is None:
            continue
        source_spec = node.command_spec
        best_pt = source_spec.expected_artifacts.get("best_pt")
        training_run_dir = best_pt.parent.parent if best_pt is not None else None
        recovery_spec = CommandSpec(
            command_type="benchmark",
            command=source_spec.command,
            args=[],
            argv=[source_spec.command],
            shell=False,
            timeout_seconds=source_spec.timeout_seconds,
            expected_artifacts={
                "predictions_json": (
                    training_run_dir / "coco_post_eval" / "predictions.json"
                    if training_run_dir is not None
                    else orchestrator.context.artifact_path(f"{node.node_id}_predictions.json")
                ),
                "coco_eval_json": (
                    training_run_dir / "coco_post_eval" / "coco_eval.json"
                    if training_run_dir is not None
                    else orchestrator.context.artifact_path(f"{node.node_id}_coco_eval.json")
                ),
            },
            expected_metrics=[
                "ap_small",
                "ap_medium",
                "ap_large",
                "per_class_ap/*",
                "per_class_ar/*",
                "fn_heavy_classes",
                "background_fp_classes",
                "localization_heavy_classes",
                "confusion_summary",
            ],
            resource_requirements=source_spec.resource_requirements.model_copy(
                update={"requires_batch_tuning": False, "full_run": False, "high_risk": False}
            ),
            metadata={
                **source_spec.metadata,
                "evidence_recovery_action": "coco_post_eval",
                "training_run_dir": training_run_dir.as_posix() if training_run_dir is not None else "",
                "data_yaml": orchestrator.context.data_yaml.as_posix(),
                "source_training_node_id": node.node_id,
            },
        )
        recovery_node = node.model_copy(
            update={
                "command_spec": recovery_spec,
                "command": recovery_spec.display(),
                "status": "planned",
            }
        )
        items.append(ExecutionQueueItem.from_node(orchestrator.context.run_id, recovery_node))
    queue = ExecutionQueue(
        run_id=orchestrator.context.run_id,
        items=items,
        metadata={
            "source_authority": "PilotEvidenceCompletenessGate",
            "evidence_recovery_only": True,
            "source_node_count": len(items),
        },
    )
    store = ExecutionQueueStore(orchestrator.context.run_dir)
    store.save(queue)
    path = orchestrator.context.run_dir / "execution_queue.yaml"
    orchestrator.evidence_store.log_artifact_manifest(
        run_id=orchestrator.context.run_id,
        name="execution_queue",
        artifact_path=path,
        producer_stage="pilot_evidence_recovery",
    )
    EventLog(orchestrator.context.events_path).append(
        run_id=orchestrator.context.run_id,
        event_type="queue_enqueued",
        status="completed",
        message=f"Enqueued {len(items)} COCO evidence recovery actions; no training actions are eligible.",
        artifacts={"execution_queue": path},
        details={"node_ids": sorted(incomplete_ids), "evidence_recovery_only": True},
    )
    return queue


def _register_guarded_pilot_trials(
    scheduler: ASHAScheduler,
    child: LoopOrchestrator,
    executable_nodes: list[ExperimentNode],
) -> int:
    """Register guarded recipes without granting them training budget directly."""
    plan_path = child.context.artifact_path("round_execution_plan.yaml")
    if not plan_path.is_file():
        return 0
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
    registered = 0
    existing_trial_ids = {trial.trial_id for trial in scheduler.study.trials}
    for node in executable_nodes:
        if _matched_baseline_node(node):
            continue
        source = source_by_candidate.get(node.candidate_config.candidate_id)
        if source is None:
            continue
        if (
            source.command_spec is not None
            and source.command_spec.metadata.get("matched_pilot_required") is True
            and baseline_control is None
        ):
            continue
        trial_id = f"{scheduler.study.base_run_id}:{node.candidate_config.candidate_id}"
        raw_targets = source.candidate_config.train_overrides.get("target_error_facts", [])
        target_error_facts = [
            dict(item)
            for item in raw_targets
            if isinstance(raw_targets, list) and isinstance(item, dict)
        ]
        trial = scheduler.register_trial(
            trial_id=trial_id,
            candidate_id=node.candidate_config.candidate_id,
            source_run_id=child.context.run_id,
            source_node=source,
            baseline_control_node=baseline_control,
            target_error_facts=target_error_facts,
        )
        if trial.trial_id not in existing_trial_ids:
            registered += 1
            existing_trial_ids.add(trial.trial_id)
    return registered


def _asha_observation(
    child: LoopOrchestrator,
    *,
    node: ExperimentNode,
    assignment: ASHAAssignment,
    target_error_facts: list[dict[str, object]],
) -> ASHAObservation:
    """Build one strict paired ASHA observation from imported local evidence."""
    evidence = child.evidence_store.load_run(child.context.run_id)
    facts = ErrorFactStore(child.context.run_root).read(child.context.run_id)
    paired_result = build_paired_experiment_result(
        run_id=child.context.run_id,
        candidate_id=node.candidate_config.candidate_id,
        candidate_node_id=node.node_id,
        metric_records=evidence.metric_records,
        error_facts=facts,
        primary_metric="map50_95",
        target_error_facts=[dict(item) for item in target_error_facts],
    )
    paired_result_path = paired_result.to_json(
        child.context.artifact_path(f"{node.node_id}_paired_experiment_result.json")
    )
    child.evidence_store.log_artifact_manifest(
        child.context.run_id,
        name=f"{node.node_id}_paired_experiment_result",
        artifact_path=paired_result_path,
        producer_stage="asha_observation",
        candidate_id=node.candidate_config.candidate_id,
        node_id=node.node_id,
        protocol_hash=(
            paired_result.matched_control.match_key.protocol_hash
            if paired_result.matched_control.match_key is not None
            else None
        ),
    )
    primary_delta = paired_result.metric_deltas.get("map50_95")
    paired_delta_value = primary_delta.effect_delta if primary_delta is not None else None
    improved_count = sum(1 for item in paired_result.target_error_fact_deltas if item.improved)
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
        paired_result.verified
        and paired_delta_value is not None
        and (not requires_target_facts or (bool(target_error_facts) and bool(facts)))
        and not missing_diagnosis_evidence
    )
    latency_regression = _paired_regression_ratio(paired_result.latency_delta)
    model_size_regression = _paired_regression_ratio(paired_result.model_size_delta)
    return ASHAObservation(
        stage_id=assignment.stage_id,
        node_id=node.node_id,
        seed_index=assignment.seed_index,
        seed=assignment.seed,
        paired_delta=paired_delta_value,
        paired_result_verified=paired_result.verified,
        paired_result_hash=paired_result.result_hash,
        protocol_match_status=paired_result.protocol_match_status,
        paired_experiment_result=paired_result,
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


def _paired_regression_ratio(delta: Any) -> float | None:
    if delta is None or delta.baseline_value == 0:
        return None
    return delta.candidate_value / delta.baseline_value - 1.0


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


def assess_candidate_execution(
    report: LoopPolicyEvaluationReport,
    *,
    component_contracts: list[ComponentContract] | None = None,
    adapter_registry: ComponentAdapterRegistry | None = None,
    workspace: Path | None = None,
    evidence_store: EvidenceStore | None = None,
    run_id: str | None = None,
    protocol_hash: str | None = None,
) -> list[CandidateExecutionAssessment]:
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

        bridge_result = None
        if candidate.components:
            contracts = {item.component_id: item for item in (component_contracts or [])}
            missing_contracts = [item for item in candidate.components if item not in contracts]
            immature = [
                item for item in candidate.components
                if item in contracts and not contracts[item].can_execute
            ]
            if missing_contracts or immature:
                execution_class = "adapter_required"
                required_adapters.extend(
                    f"component_adapter:{item}" for item in [*missing_contracts, *immature]
                )
                reasons.extend(f"missing_component_contract:{item}" for item in missing_contracts)
                reasons.extend(
                    f"component_maturity_below_smoke_passed:{item}:{contracts[item].maturity}"
                    for item in immature
                )
            elif node is not None and command is not None and command.command_type == "train":
                recipe = RecipeSpec(
                    recipe_id=candidate.action_id or evaluation.policy_id,
                    version="execution-bridge.v1",
                    target_error_facts=[],
                    target_metrics=[],
                    component_ids=list(candidate.components),
                    train_overrides={"imgsz": 640, **candidate.train_overrides},
                    fixed_variables={"imgsz": 640, **evaluation.fixed_variables},
                    primary_changed_variable=(
                        next(iter(evaluation.changed_variables), candidate.action_id or candidate.components[0])
                    ),
                    coupled_variables=(
                        list(evaluation.changed_variables) if len(candidate.components) > 1 else []
                    ),
                    stop_conditions=["pilot_no_gain"],
                    maturity="smoke_passed",
                )
                bridge_result = ComponentExecutionBridge(
                    adapter_registry=adapter_registry or ComponentAdapterRegistry()
                ).prepare(
                    recipe=recipe,
                    node=node,
                    contracts=contracts,
                    model_config={"model": candidate.base_model},
                    training_config=dict(candidate.train_overrides),
                    workspace=(workspace or Path("artifacts/component_execution")) / node.node_id,
                    evidence_store=evidence_store,
                    run_id=run_id,
                    protocol_hash=protocol_hash,
                )
                evaluation.experiment_node = bridge_result.node
                node = bridge_result.node
                command = node.command_spec
                if bridge_result.status != "executable":
                    execution_class = (
                        "adapter_required" if bridge_result.status == "adapter_required" else "recommendation_only"
                    )
                    required_adapters.extend(
                        f"component_adapter:{item}" for item in candidate.components
                    )
                    reasons.extend(bridge_result.blocked_by)
                else:
                    reasons.append(
                        f"component adapters passed smoke gate; patch={bridge_result.aggregate_patch_hash}"
                    )

        unsupported_overrides = (
            _unsupported_train_overrides(candidate.train_overrides)
            if command is not None
            and command.command_type == "train"
            and candidate.action_domain not in NON_TRAINING_DOMAINS
            else []
        )
        adapted_training_fields = {
            key.split(".", 1)[1]
            for key in (bridge_result.changed_variables if bridge_result is not None else {})
            if key.startswith("training_config.")
        }
        unsupported_overrides = [key for key in unsupported_overrides if key not in adapted_training_fields]
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
    reusable_reasons = {
        "round_completed",
        "diversity_deferred",
        "asha_registration_dry_run",
        "asha_assignment_completed",
    }
    if result.status != "completed" or result.stop_reason not in reusable_reasons:
        return None
    if execute and result.stop_reason == "asha_registration_dry_run":
        return None
    if execute and result.training_loop is not None and result.training_loop.executor == "dry-run":
        return None
    return result


def _next_executable_auto_round_index(run_root: Path, base_run_id: str) -> int:
    """Return the next absolute round index after completed executed child rounds."""
    completed = [
        index
        for index, result in _completed_auto_rounds(run_root, base_run_id).items()
        if result.status == "completed"
        and (
            result.stop_reason == "diversity_deferred"
            or (
                result.stop_reason in {"round_completed", "asha_assignment_completed"}
                and (
                    result.stop_reason == "asha_assignment_completed"
                    or (
                        result.training_loop is not None
                        and result.training_loop.executor != "dry-run"
                    )
                )
            )
        )
    ]
    return (max(completed) + 1) if completed else 1


def _latest_completed_auto_child(base: LoopOrchestrator, round_index: int) -> LoopOrchestrator:
    """Return the latest completed child up to round_index, or the base run."""
    if round_index <= 0:
        return base
    completed = [
        index for index, result in _completed_auto_rounds(
            base.context.run_root, base.context.run_id
        ).items()
        if index <= round_index
        and result.status == "completed"
        and result.stop_reason in {"round_completed", "asha_assignment_completed"}
        and (
            result.stop_reason == "asha_assignment_completed"
            or (result.training_loop is not None and result.training_loop.executor != "dry-run")
        )
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
        "asha_state_path",
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
    """Return executed or diversity-screened action ids for a base run."""
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
        evaluation_path = path.parent / "policy_evaluation.yaml"
        if evaluation_path.is_file():
            try:
                evaluation = LoopPolicyEvaluationReport.model_validate(read_yaml(evaluation_path))
            except ValueError:
                evaluation = None
            if evaluation is not None:
                for item in evaluation.evaluations:
                    if not item.diversity_reason or item.candidate_config is None:
                        continue
                    action_id = item.candidate_config.action_id
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
                    "paper_intelligence": snapshot.paper_intelligence,
                    "unavailable_reason": snapshot.unavailable_reason,
                    "research_network_allowed": False,
                    "maturity_summary": snapshot.maturity_summary.model_dump(mode="json"),
                }
            )
            contracts_path = snapshot_dir / "component_contracts.yaml"
            recipes_path = snapshot_dir / "recipes.yaml"
            paper_root = snapshot_dir
        else:
            contracts_path = ResourcePaths.COMPONENT_COMPATIBILITY
            recipes_path = None
            # Never read the live research registry during training. A missing
            # frozen snapshot is an explicit unavailable state, not permission
            # to perform an implicit online/live lookup.
            paper_root = child.context.run_dir / ".paper_intelligence_unavailable"
            paper_root.mkdir(parents=True, exist_ok=True)
            child.context.metadata["research_snapshot_verified"] = False
            child.context.metadata["paper_intelligence"] = "unavailable"
            child.context.metadata["unavailable_reason"] = "snapshot_missing"
            child.context.metadata["research_network_allowed"] = False
        contracts = load_contracts(contracts_path) if contracts_path.exists() else []
        if snapshot_ref is None:
            contracts = _merge_local_component_contracts(contracts)
        component_registry = ComponentRegistry(contracts)  # type: ignore[arg-type]
        paper_registry = PaperRegistry(paper_root)
        recipe_registry = (
            RecipeRegistry.from_path(
                recipes_path,
                component_contracts=contracts if snapshot_ref is not None else (),
            )
            if recipes_path is not None and recipes_path.exists()
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
            training_budget={
                "profile": "pilot",
                "fidelity": "pilot_3",
                "imgsz": 640,
                "seed": child.context.metadata.get("seed", 1),
                "dataset_signature": child.context.dataset_manifest_sha256 or child.context.dataset_version,
                "protocol_hash": child.context.metadata.get("baseline_protocol_hash", "unknown"),
            },
            optimization_objective=load_optimization_objective(
                child.context.metadata.get("optimization_objective_path")
            ),
        )
        compatibility_snapshot = {
            "schema_version": "component_compatibility_snapshot.v1",
            "imgsz": 640,
            "research_snapshot_hash": snapshot_hash,
            "research_snapshot_verified": bool(child.context.metadata.get("research_snapshot_verified", False)),
            "paper_intelligence": child.context.metadata.get("paper_intelligence", "unavailable"),
            "paper_intelligence_reason": child.context.metadata.get("unavailable_reason"),
            "research_network_allowed": False,
            "maturity_summary": snapshot.maturity_summary.model_dump(mode="json") if snapshot_ref is not None else {},
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
            "paper_intelligence": child.context.metadata.get("paper_intelligence", "unavailable"),
            "paper_intelligence_reason": child.context.metadata.get("unavailable_reason"),
            "research_network_allowed": False,
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


def _assess_policy_evaluation(child: LoopOrchestrator) -> list[CandidateExecutionAssessment]:
    path = child.context.artifact_path("policy_evaluation.yaml")
    if not path.is_file():
        return []
    report = LoopPolicyEvaluationReport.model_validate(read_yaml(path))
    contracts = _load_execution_contracts(child)
    assessments = assess_candidate_execution(
        report,
        component_contracts=contracts,
        workspace=child.context.artifact_path("component_execution"),
        evidence_store=child.evidence_store,
        run_id=child.context.run_id,
        protocol_hash=child.context.run_protocol_hash,
    )
    write_yaml(path, report.model_dump(mode="json"))
    plan_path = child.context.artifact_path("experiment_plan.yaml")
    if plan_path.is_file():
        plan = ExperimentPlan.from_yaml(plan_path)
        updated = {
            item.experiment_node.node_id: item.experiment_node
            for item in report.evaluations
            if item.experiment_node is not None
        }
        plan.nodes = [updated.get(node.node_id, node) for node in plan.nodes]
        plan.to_yaml(plan_path)
    round_plan_path = child.context.artifact_path("round_execution_plan.yaml")
    if round_plan_path.is_file():
        round_plan = RoundExecutionPlan.from_yaml(round_plan_path)
        patched_by_candidate = {
            item.experiment_node.candidate_config.candidate_id: item.experiment_node
            for item in report.evaluations
            if item.experiment_node is not None
        }
        round_plan.execution_nodes = [
            _merge_adapter_node(node, patched_by_candidate.get(node.candidate_config.candidate_id))
            for node in round_plan.execution_nodes
        ]
        round_plan.deferred_nodes = [
            _merge_adapter_node(node, patched_by_candidate.get(node.candidate_config.candidate_id))
            for node in round_plan.deferred_nodes
        ]
        round_plan.to_yaml(round_plan_path)
    return assessments


def _merge_adapter_node(original: ExperimentNode, patched: ExperimentNode | None) -> ExperimentNode:
    if patched is None or _matched_baseline_node(original):
        return original
    return patched.model_copy(
        update={
            "node_id": original.node_id,
            "seed": original.seed,
            "parent_id": original.parent_id,
            "status": original.status,
        }
    )


def _load_execution_contracts(child: LoopOrchestrator) -> list[ComponentContract]:
    """Load frozen snapshot contracts plus locally implemented contract files."""
    paths: list[Path] = []
    snapshot_path = child.context.metadata.get("research_snapshot_path")
    if isinstance(snapshot_path, str) and snapshot_path:
        paths.append(Path(snapshot_path) / "component_contracts.yaml")
    paths.append(ResourcePaths.COMPONENT_COMPATIBILITY)
    paths.extend(sorted(ResourcePaths.COMPONENTS_DIR.rglob("*.yaml")))
    contracts: dict[str, ComponentContract] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            loaded = load_contracts(path)
        except (ValueError, KeyError, TypeError):
            continue
        for contract in loaded:
            contracts[contract.component_id] = contract
    return list(contracts.values())


def _merge_local_component_contracts(
    initial: list[ComponentContract],
) -> list[ComponentContract]:
    contracts = {item.component_id: item for item in initial}
    for path in sorted(ResourcePaths.COMPONENTS_DIR.rglob("*.yaml")):
        try:
            loaded = load_contracts(path)
        except (ValueError, KeyError, TypeError):
            continue
        for contract in loaded:
            contracts[contract.component_id] = contract
    return list(contracts.values())


def _empty_diversity_round_reason(path: Path) -> str | None:
    """Distinguish temporary diversity deferral from terminal family exhaustion."""
    if not path.is_file():
        return None
    report = LoopPolicyEvaluationReport.model_validate(read_yaml(path))
    decisions = [
        item for item in report.evaluations
        if item.diversity_reason and item.decision == "deferred"
    ]
    if not decisions:
        return None
    if all(item.diversity_reason == "component_family_exhausted" for item in decisions):
        return "family_exhaustion"
    if all(
        item.diversity_reason == "duplicate_recipe_fingerprint"
        or item.diversity_reason.startswith("component_family_cooldown:")
        or item.diversity_reason.startswith("minimum_semantic_distance:")
        for item in decisions
    ):
        return "diversity_deferred"
    return None


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
        profile = str(child.context.metadata.get("training_profile") or "pilot")
        _bind_child_run_protocol(child, round_plan, profile=profile)
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
        run_protocol_hash=round_plan.run_protocol_hash or original.run_protocol_hash,
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


def _bind_child_run_protocol(
    child: LoopOrchestrator,
    round_plan: RoundExecutionPlan,
    *,
    profile: str,
) -> RunProtocolVersion | None:
    """Attach a stage-specific protocol identity to a child run and all executable nodes."""
    config = _training_config_from_context(child)
    node = next((item for item in round_plan.execution_nodes if not _matched_baseline_node(item)), None)
    if config is None or node is None or profile not in config.budget_profiles:
        return None
    epochs = _command_numeric_arg(node.command_spec, "epochs")
    fraction = _command_float_arg(node.command_spec, "fraction")
    protocol = build_run_protocol_version(
        model=node.candidate_config.base_model,
        context=child.context,
        training_config=config,
        profile=profile,
        seed=node.seed,
        epochs=int(epochs) if epochs is not None else None,
        fraction=fraction,
    )
    for execution in round_plan.execution_nodes:
        if execution.command_spec is None:
            continue
        execution.command_spec = execution.command_spec.model_copy(
            update={
                "metadata": {
                    **execution.command_spec.metadata,
                    "run_protocol_hash": protocol.protocol_hash,
                    "dataset_manifest_sha256": protocol.dataset_manifest_sha256,
                    "subset_manifest_sha256": protocol.subset_manifest_sha256,
                    "batch_policy_hash": protocol.batch_policy_hash,
                    "eval_protocol_hash": protocol.eval_protocol_hash,
                    "ultralytics_version": protocol.ultralytics_version,
                    "code_version": protocol.code_version,
                }
            }
        )
        execution.command = execution.command_spec.display()
    round_plan.run_protocol_hash = protocol.protocol_hash
    path = child.context.artifact_path("run_protocol.yaml")
    protocol.to_yaml(path)
    child.context.run_protocol_path = path
    child.context.run_protocol_hash = protocol.protocol_hash
    child.context.legacy_run = False
    child.context.metadata.update(
        {
            "run_protocol_hash": protocol.protocol_hash,
            "post_eval_protocol_hash": protocol.eval_protocol_hash,
            "subset_manifest_sha256": protocol.subset_manifest_sha256,
            "batch_policy_hash": protocol.batch_policy_hash,
            "ultralytics_version": protocol.ultralytics_version,
            "code_version": protocol.code_version,
        }
    )
    child.context.to_yaml()
    child.context.to_json()
    child.evidence_store.log_artifact_manifest(
        run_id=child.context.run_id,
        name="run_protocol",
        artifact_path=path,
        producer_stage="auto_optimization_loop",
    )
    return protocol


def _command_numeric_arg(spec: CommandSpec | None, name: str) -> int | None:
    value = _command_arg(spec, name)
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _command_float_arg(spec: CommandSpec | None, name: str) -> float | None:
    value = _command_arg(spec, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _command_arg(spec: CommandSpec | None, name: str) -> str | None:
    if spec is None:
        return None
    prefix = f"{name}="
    for arg in spec.args:
        text = str(arg)
        if text.startswith(prefix):
            return text.split("=", 1)[1]
    return None


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
                "diagnosis_gate_passed": latest.diagnosis_gate_passed if latest is not None else None,
                "diagnosis_checks": latest.diagnosis_checks if latest is not None else [],
                "promotion_rejection_reasons": (
                    latest.promotion_rejection_reasons if latest is not None else []
                ),
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
        if round_result.diversity_outcomes:
            outcomes = round_result.diversity_outcomes
            lines.append(
                "  - Diversity: "
                + ", ".join(
                    f"{item.component_family}:{item.bucket}:delta={item.effect_delta}"
                    for item in outcomes
                )
            )
        if round_result.diversity_stop is not None:
            lines.append(
                f"  - Search boundary: stagnant_rounds={round_result.diversity_stop.no_improvement_rounds} "
                f"exhausted_families={round_result.diversity_stop.exhausted_families}"
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
