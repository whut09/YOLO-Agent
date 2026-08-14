"""Automatic pilot optimization loop driver.

The driver connects completed pilot evidence to the guarded loop machinery:

PilotResult -> LLMAnalysis -> PolicyProposal -> GuardedCandidate -> PilotRun -> DeltaAnalysis

It intentionally does not pretend that every metadata proposal can be trained.
Each accepted candidate is classified before queue materialization so the loop
only executes candidates backed by real adapter support.
"""

from __future__ import annotations

import json
import hashlib
import re
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
    ASHATrial,
)
from yolo_agent.agents.diagnosis_promotion import (
    DiagnosisPromotionGate,
    DiagnosisPromotionPolicy,
)
from yolo_agent.agents.loop_io import read_json, read_yaml, write_json, write_yaml
from yolo_agent.agents.loop_policy_evaluator import BudgetPolicy, LoopPolicyEvaluationReport
from yolo_agent.agents.orchestrator import LoopOrchestrator, TrainingLoopResult
from yolo_agent.agents.paper_recipe_materialization.runtime_identity import (
    validate_certified_runtime_node,
)
from yolo_agent.agents.paper_recipe_materialization.maturity import (
    EffectiveComponentMaturity,
    EffectiveMaturityResolver,
)
from yolo_agent.agents.paper_recipe_planner import PaperRecipePlanner
from yolo_agent.agents.paper_proposal_ledger import (
    PaperCandidateCoverageLedger,
    ProposalDisposition,
    planned_recipe_disposition,
)
from yolo_agent.agents.recipe_critic import RecipeCritic
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint
from yolo_agent.core.coco_error_selection import select_coco_error_facts
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.decision_ledger import DecisionLedger, DecisionLedgerRecord
from yolo_agent.core.error_facts import (
    ErrorFact,
    ErrorFactStore,
    build_error_facts_from_coco_error_report,
    build_error_facts_from_coco_metrics,
)
from yolo_agent.core.execution_failure import ExecutionFailure, classify_execution_failure
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueItem, ExecutionQueueStore
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.experiment_graph import Evidence, ExperimentNode, ExperimentPlan, MetricEvidence
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.full_run_consent import FullRunConsentDriver
from yolo_agent.core.evidence_selector import EvidenceSelector, select_metric_evidence
from yolo_agent.core.matched_baseline import paired_metric_delta
from yolo_agent.core.paired_experiment import build_paired_experiment_result
from yolo_agent.core.process_probe import probe_command_process
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.adapters import AdapterRuntimePayload, ComponentAdapterRegistry
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
from yolo_agent.components.adapters.assigners.yolo26_assignment import ASSIGNMENT_SPECS
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.certification.assignment_pilot_gate import (
    AssignmentActivePilotMaterializer,
)
from yolo_agent.certification.component_queue_gate import (
    ComponentQueueCertificationGate,
)
from yolo_agent.certification.runner import RealGpuAcceptanceSuite
from yolo_agent.core.policy_memory import PolicyMemoryStore
from yolo_agent.core.pilot_evidence import PilotEvidenceCompletenessGate, PilotEvidenceCompletenessResult
from yolo_agent.core.optimization_objective import (
    OptimizationObjective,
    OptimizationObjectiveStatus,
    evaluate_optimization_objective,
    load_optimization_objective,
)
from yolo_agent.core.optimization_readiness import (
    OptimizationReadinessGate,
    OptimizationReadinessResult,
)
from yolo_agent.core.round_execution_plan import RoundExecutionPlan, build_asha_assignment_plan
from yolo_agent.core.run_protocol import RunProtocolVersion, build_run_protocol_version
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.recipes.schemas import AtomicRecipe, CoupledRecipe, RecipeSpec
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.maturity_snapshot import (
    EffectiveComponentMaturityManifest,
    FrozenComponentMaturity,
)
from yolo_agent.research.reproduction_pipeline import ReproductionPipeline
from yolo_agent.research.snapshot import load_research_snapshot
from yolo_agent.resources import ResourcePaths
from yolo_agent.tools.dataset_stats import DatasetReport


CandidateExecutionClass = Literal["executable", "recommendation_only", "adapter_required"]


def _assignment_shadow_evidence_only(node: ExperimentNode) -> bool:
    """Return whether a runtime node observes assignment without changing training."""
    metadata = node.command_spec.metadata if node.command_spec is not None else {}
    if metadata.get("assignment_execution_mode") == "shadow":
        return True
    payload_path = metadata.get("adapter_runtime_payload_path")
    if not isinstance(payload_path, str) or not payload_path:
        return False
    try:
        payload = AdapterRuntimePayload.read(payload_path, verify_imports=False)
    except (OSError, TypeError, ValueError):
        return False
    return bool(payload.assigner_plugin) and all(
        str(plugin.options.get("mode") or "shadow") == "shadow"
        for plugin in payload.assigner_plugin
    )


def _activate_completed_assignment_shadows(
    orchestrator: LoopOrchestrator,
    scheduler: ASHAScheduler,
) -> bool:
    """Migrate completed or stale legacy shadows into active ASHA trials.

    Older loop versions could promote a shadow trial to ``pilot_10`` before
    assignment activation was wired in.  Valid evidence is enough to migrate
    that state, but only after the child run's actual command is confirmed to
    have no live process.  Unknown process state is left untouched.
    """
    changed = False
    for trial in list(scheduler.study.trials):
        if not _assignment_shadow_evidence_only(trial.source_node):
            continue
        has_completed_evidence = any(
            item.trial_id == trial.trial_id and item.status == "completed"
            for item in scheduler.study.assignments
        )
        outstanding = [
            item
            for item in scheduler.study.assignments
            if item.trial_id == trial.trial_id and item.status in {"issued", "running"}
        ]
        if outstanding:
            live_state = _assignment_shadow_process_state(orchestrator, scheduler, trial)
            if live_state is not False:
                continue
        elif not has_completed_evidence and trial.status not in {"eliminated", "failed"}:
            continue
        if trial.status not in {
            "waiting",
            "running",
            "promotion_pending",
            "needs_evidence",
            "full_pending_confirmation",
            "eliminated",
            "failed",
        }:
            continue
        active_exists = any(
            item.candidate_id == f"{trial.candidate_id}_active"
            for item in scheduler.study.trials
        )
        if active_exists:
            needs_cleanup = bool(outstanding) or trial.status != "eliminated"
            if needs_cleanup:
                scheduler.complete_evidence_only_trial(
                    trial.trial_id,
                    node_id=trial.source_node.node_id,
                    reason="assignment_shadow_evidence_collected_not_ranked",
                    succeeded=True,
                )
                changed = True
            continue
        activated, blockers = _activate_assignment_shadow_trial(
            orchestrator,
            scheduler,
            trial_id=trial.trial_id,
            completed_node_id=trial.source_node.node_id,
        )
        if not activated:
            continue
        scheduler.complete_evidence_only_trial(
            trial.trial_id,
            node_id=trial.source_node.node_id,
            reason="assignment_shadow_evidence_collected_not_ranked",
            succeeded=True,
        )
        changed = True
    return changed


def _assignment_shadow_process_state(
    orchestrator: LoopOrchestrator,
    scheduler: ASHAScheduler,
    trial: ASHATrial,
) -> bool | None:
    """Return whether a legacy shadow child is live, or ``None`` if unknown."""
    context = orchestrator.context
    assignments = [
        item
        for item in scheduler.study.assignments
        if item.trial_id == trial.trial_id and item.status in {"issued", "running"}
    ]
    run_id = next(
        (item.assigned_run_id for item in reversed(assignments) if item.assigned_run_id),
        None,
    )
    run_root = getattr(context, "run_root", None)
    if not run_id or run_root is None:
        return None
    queue_path = Path(run_root) / run_id / "execution_queue.yaml"
    if not queue_path.is_file():
        return None
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
    except Exception:
        return None
    if not queue.items:
        return None
    states = [probe_command_process(item.command).status for item in queue.items]
    if "found" in states:
        return True
    if all(state == "not_found" for state in states):
        return False
    return None


def _activate_assignment_shadow_trial(
    orchestrator: LoopOrchestrator,
    scheduler: ASHAScheduler,
    *,
    trial_id: str,
    completed_node_id: str,
) -> tuple[bool, list[str]]:
    trial = scheduler.study.trial(trial_id)
    prepared, blockers = _materialize_active_assignment_node(
        orchestrator,
        trial=trial,
    )
    if prepared is None:
        return False, blockers
    active_node, active_recipe = prepared
    active_candidate_id = active_node.candidate_config.candidate_id
    scheduler.register_trial(
        trial_id=f"{scheduler.study.base_run_id}:{active_candidate_id}",
        candidate_id=active_candidate_id,
        source_run_id=orchestrator.context.run_id,
        source_node=active_node,
        baseline_control_node=trial.baseline_control_node,
        target_error_facts=trial.target_error_facts,
    )
    EventLog(orchestrator.context.events_path).append(
        run_id=orchestrator.context.run_id,
        event_type="auto_round_decision",
        status="completed",
        message=(
            f"Assignment shadow evidence passed; queued active candidate "
            f"{active_candidate_id} for matched mAP evaluation."
        ),
        details={
            "shadow_trial_id": trial_id,
            "shadow_node_id": completed_node_id,
            "active_recipe_id": active_recipe.recipe_id,
            "active_candidate_id": active_candidate_id,
        },
    )
    return True, []


def _materialize_active_assignment_node(
    orchestrator: LoopOrchestrator,
    *,
    trial: ASHATrial,
) -> tuple[tuple[ExperimentNode, AtomicRecipe] | None, list[str]]:
    source = trial.source_node
    command = source.command_spec
    if command is None:
        return None, ["assignment_shadow_command_missing"]
    payload_path = command.metadata.get("adapter_runtime_payload_path")
    if not isinstance(payload_path, str) or not payload_path:
        return None, ["assignment_shadow_payload_missing"]
    try:
        payload = AdapterRuntimePayload.read(payload_path, verify_imports=False)
    except (OSError, TypeError, ValueError) as exc:
        return None, [f"assignment_shadow_payload_invalid:{exc}"]
    if len(payload.assigner_plugin) != 1 or len(payload.component_ids) != 1:
        return None, ["assignment_shadow_payload_scope_invalid"]
    plugin = payload.assigner_plugin[0]
    component_id = payload.component_ids[0]
    spec = ASSIGNMENT_SPECS.get(component_id)
    if spec is None or str(plugin.options.get("mode") or "shadow") != "shadow":
        return None, ["assignment_shadow_payload_not_supported"]
    evidence = _assignment_shadow_evidence_path(command, payload, spec.method)
    try:
        evidence_payload = json.loads(evidence.read_text(encoding="utf-8-sig"))
    except (OSError, TypeError, ValueError) as exc:
        return None, [f"assignment_shadow_evidence_invalid:{exc}"]
    if evidence_payload.get("runtime_payload_hash") != payload.payload_hash:
        return None, ["assignment_shadow_source_payload_mismatch"]
    control = trial.baseline_control_node
    control_protocol = _node_protocol_hash(control) if control is not None else ""
    shadow_recipe = AtomicRecipe(
        recipe_id=str(command.metadata.get("component_recipe_id") or trial.candidate_id),
        version="assignment-shadow-runtime.v1",
        target_error_facts=list(trial.target_error_facts),
        target_metrics=["assignment_positive_ratio", "assignment_conflict_rate"],
        component_ids=[component_id],
        train_overrides={
            "imgsz": 640,
            spec.changed_variable: "shadow",
        },
        fixed_variables={
            "imgsz": 640,
            "assignment_path": str(plugin.options.get("assignment_path") or "one_to_many"),
        },
        primary_changed_variable=spec.changed_variable,
        compatibility_requirements=[
            "point_based_candidates",
            "shadow_evidence_gate",
            "matched_control",
        ],
        promotion_requirements=["explicit_active_pilot", "matched_control", "ASHA_only"],
        maturity="smoke_passed",
    )
    decision = AssignmentActivePilotMaterializer().materialize(
        shadow_recipe=shadow_recipe,
        shadow_evidence_path=evidence,
        candidate_protocol_hash=payload.protocol_hash,
        control_protocol_hash=control_protocol,
        matched_control_available=control is not None,
        minimum_shadow_batches=int(plugin.options.get("minimum_shadow_batches") or 1),
        maximum_conflict_rate=float(plugin.options.get("maximum_conflict_rate") or 1.0),
    )
    if not decision.allowed or decision.active_recipe is None:
        return None, list(decision.blocked_by)
    active_recipe = decision.active_recipe
    base_command = command.model_copy(
        update={
            "command": payload.base_command[0],
            "args": payload.base_command[1:],
            "argv": list(payload.base_command),
            "expected_artifacts": {
                key: value
                for key, value in command.expected_artifacts.items()
                if not key.startswith(("assignment_", "adapter_", "component_execution"))
                and key != "plugin_runtime_evidence"
            },
            "metadata": {
                key: value
                for key, value in command.metadata.items()
                if not key.startswith(("adapter_", "component_", "assignment_"))
                and key not in {"evidence_only", "optimization_metric_eligible"}
            },
        }
    )
    active_candidate_id = f"{trial.candidate_id}_active"
    active_candidate = source.candidate_config.model_copy(
        update={
            "candidate_id": active_candidate_id,
            "train_overrides": dict(active_recipe.train_overrides),
            "action_id": f"{source.candidate_config.action_id or component_id}.active",
        }
    )
    active_source = source.model_copy(
        update={
            "node_id": f"{source.node_id}_active",
            "candidate_config": active_candidate,
            "command_spec": base_command,
            "command": base_command.display(),
            "effective_overrides": dict(active_recipe.train_overrides),
            "changed_variables": {},
        }
    )
    contracts = {
        item.component_id: item for item in _load_execution_contracts(orchestrator)
    }
    contract = contracts.get(component_id)
    if contract is None:
        return None, [f"active_assignment_contract_unavailable:{component_id}"]
    runtime = ComponentExecutionBridge().prepare(
        recipe=active_recipe,
        node=active_source,
        contracts={component_id: contract},
        training_config=dict(active_recipe.train_overrides),
        workspace=(
            orchestrator.context.artifact_path("assignment_active")
            / re.sub(r"[^A-Za-z0-9_.-]+", "_", active_candidate_id)
        ),
        evidence_store=orchestrator.evidence_store,
        run_id=orchestrator.context.run_id,
        protocol_hash=payload.protocol_hash,
        dry_run=True,
    )
    if runtime.status != "executable":
        return None, list(runtime.blocked_by)
    metadata = runtime.node.command_spec.metadata if runtime.node.command_spec else {}
    if metadata.get("assignment_execution_mode") != "active":
        return None, ["active_assignment_payload_not_active"]
    return (runtime.node, active_recipe), []


def _assignment_shadow_evidence_path(
    command: CommandSpec,
    payload: AdapterRuntimePayload,
    method: str,
) -> Path:
    artifact = command.expected_artifacts.get(f"assignment_{method}_shadow_evidence")
    if artifact is not None:
        return Path(artifact)
    return Path(str(command.metadata.get("adapter_runtime_payload_path") or "")).parent / (
        f"assignment_{method}_shadow_evidence.json"
    )


def _node_protocol_hash(node: ExperimentNode | None) -> str:
    if node is None or node.command_spec is None:
        return ""
    metadata = node.command_spec.metadata
    return str(
        metadata.get("run_protocol_hash")
        or metadata.get("baseline_protocol_hash")
        or metadata.get("adapter_runtime_protocol_hash")
        or ""
    )


def _evidence_only_assignment_plan(plan: RoundExecutionPlan) -> RoundExecutionPlan:
    """Remove the matched control and mAP post-eval from a shadow evidence plan."""
    candidate_nodes = [
        node for node in plan.execution_nodes if not _matched_baseline_node(node)
    ]
    candidate_ids = {node.node_id for node in candidate_nodes}
    assignments = [
        item
        for item in plan.assignments
        if item.execution_node_id in candidate_ids and item.role == "candidate"
    ]
    for assignment in assignments:
        # Shadow assignment runs collect runtime evidence only; they are not
        # ranked candidates and therefore must not require a paired control.
        assignment.role = "evidence_recovery"
        assignment.matched_control_execution_node_id = None
        assignment.reason = "assignment_shadow_evidence_only"
    for node in candidate_nodes:
        if node.command_spec is None:
            continue
        metadata = {
            **node.command_spec.metadata,
            "evidence_only": True,
            "optimization_metric_eligible": False,
            "matched_pilot_required": False,
            "coco_post_eval_required": False,
        }
        node.command_spec = node.command_spec.model_copy(
            update={
                "expected_metrics": [],
                "metadata": metadata,
            }
        )
        node.command = node.command_spec.display()
    return plan.model_copy(
        update={
            "execution_nodes": candidate_nodes,
            "assignments": assignments,
            "primary_metric": "assignment_shadow_evidence",
            "require_complete_post_eval": False,
            "evidence_requirements": {
                node.node_id: ["assignment_shadow_evidence"] for node in candidate_nodes
            },
        }
    )


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
    adapter_ids: list[str] = Field(default_factory=list)
    adapter_patch_hash: str | None = None
    adapter_runtime_payload_hash: str | None = None
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
    readiness: OptimizationReadinessResult | None = None
    readiness_path: Path | None = None
    certification_attempted: bool = False
    certification_status: str | None = None
    certification_report_path: Path | None = None
    certification_failure: str | None = None

    @field_serializer(
        "base_run_dir",
        "summary_path",
        "full_candidate_recommendations_path",
        "asha_state_path",
        "readiness_path",
        "certification_report_path",
    )
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
    remaining = sum(1 for item in assessments if item.execution_class == "executable")
    for assessment in assessments[:8]:
        paper_context = _paper_progress_context(
            orchestrator.context.artifact_path("paper_recipe_plan.yaml"),
            assessment=assessment,
        )
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
                "adapter_ids": assessment.adapter_ids,
                "adapter_patch_hash": assessment.adapter_patch_hash,
                "adapter_runtime_payload_hash": assessment.adapter_runtime_payload_hash,
            },
        )


def _paper_progress_context(
    path: Path,
    *,
    assessment: CandidateExecutionAssessment | None = None,
) -> dict[str, str]:
    if not path.is_file():
        return {}
    raw = read_yaml(path)
    rule_plan = raw.get("rule_plan", {})
    selected = rule_plan.get("selected_recipes", []) if isinstance(rule_plan, dict) else []
    llm = raw.get("llm_proposal") if isinstance(raw.get("llm_proposal"), dict) else {}
    policies = raw.get("executable_pilot_policies", [])
    matched_policy: dict[str, Any] = {}
    if assessment is not None and isinstance(policies, list):
        matched_policy = next(
            (
                item for item in policies
                if isinstance(item, dict)
                and (
                    str(item.get("policy_id") or "") == assessment.policy_id
                    or str(item.get("action_id") or "") == str(assessment.action_id or "")
                    or str(item.get("action_id") or "") == str(assessment.candidate_id or "")
                )
            ),
            {},
        )
    recipe_id = str(matched_policy.get("action_id") or "")
    matched_recipe: dict[str, Any] = {}
    if recipe_id and isinstance(selected, list):
        matched_recipe = next(
            (
                item for item in selected
                if isinstance(item, dict) and str(item.get("recipe_id") or "") == recipe_id
            ),
            {},
        )
    return {
        "diagnosis": str(llm.get("primary_problem") or ""),
        "recipe": recipe_id,
        "changed_variable": str(matched_recipe.get("primary_changed_variable") or ""),
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

    def __init__(
        self,
        *,
        auto_certify_gpu: bool = False,
        certification_suite: RealGpuAcceptanceSuite | None = None,
        certification_model: str = "yolo26n.pt",
        certification_device: str = "0",
        certification_recipe: str = "reduce_mosaic",
    ) -> None:
        self.auto_certify_gpu = auto_certify_gpu
        self.certification_suite = certification_suite
        self.certification_model = certification_model
        self.certification_device = certification_device
        self.certification_recipe = certification_recipe

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
        require_gpu_certification: bool = True,
        certification_report_path: Path | str | None = None,
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
        if _activate_completed_assignment_shadows(base_orchestrator, asha_scheduler):
            asha_store.save(asha_scheduler)
        if execute and _reopen_retryable_resource_assignments(base_context, asha_scheduler):
            asha_store.save(asha_scheduler)
        if objective is not None:
            result.objective_status = _refresh_objective_status(base_context, objective)
        if auto_rounds <= 0:
            result.stopped_reason = "auto_rounds_zero"
            _write_final_outputs(result)
            return result
        readiness_path = base_context.artifact_path("optimization_readiness.yaml")
        configured_report = certification_report_path or base_context.metadata.get(
            "gpu_certification_report"
        )
        readiness = OptimizationReadinessGate().evaluate(
            run_root=base_context.run_root,
            execute=execute,
            require_certification=require_gpu_certification,
            report_path=configured_report,
        )
        if (
            execute
            and require_gpu_certification
            and self.auto_certify_gpu
            and not readiness.ready
        ):
            report_path = (
                Path(configured_report)
                if configured_report is not None
                else base_context.run_root
                / "certification"
                / "mini-gpu"
                / "certification_report.yaml"
            )
            result.certification_attempted = True
            result.certification_report_path = report_path
            certification_error = self._auto_certify(
                context=base_context,
                report_path=report_path,
            )
            configured_report = report_path
            readiness = OptimizationReadinessGate().evaluate(
                run_root=base_context.run_root,
                execute=execute,
                require_certification=require_gpu_certification,
                report_path=report_path,
            )
            if certification_error is not None:
                readiness.blockers.insert(
                    0,
                    f"gpu_certification_auto_run_failed:{certification_error}",
                )
                result.certification_status = "failed"
                result.certification_failure = certification_error
            elif readiness.ready:
                result.certification_status = "passed"
            else:
                result.certification_status = "failed"
                result.certification_failure = "; ".join(readiness.blockers)
        readiness.to_yaml(readiness_path, exclude_none=True, sort_keys=False)
        base_orchestrator.evidence_store.log_artifact_manifest(
            run_id=base_context.run_id,
            name="optimization_readiness",
            artifact_path=readiness_path,
            producer_stage="optimization_readiness_gate",
        )
        result.readiness = readiness
        result.readiness_path = readiness_path
        if not readiness.ready:
            result.stopped_reason = "optimization_readiness_blocked"
            EventLog(base_context.run_dir / "events.jsonl").append(
                run_id=base_context.run_id,
                event_type="contract_blocked",
                status="blocked",
                message="Automatic candidate exploration requires a matching passed GPU certification.",
                artifacts={"optimization_readiness": readiness_path},
                details={"blockers": readiness.blockers},
            )
            _write_final_outputs(result)
            return result
        if readiness.certification_report is not None:
            base_context.metadata["gpu_certification_report"] = (
                readiness.certification_report.resolve().as_posix()
            )
            base_context.metadata["gpu_certification_report_hash"] = (
                readiness.certification_report_hash
            )
            base_context.to_yaml()
            base_context.to_json()
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

        outstanding_assignment = asha_scheduler.next_assignment(
            confirm_full_run=full_run_authorized
        )
        # A persisted patience stop can predate a newly recovered active ASHA
        # assignment. Let that bounded method trial run before stopping; the
        # stop remains authoritative when no assignment is pending.
        if result.objective_status is not None and result.objective_status.should_stop:
            patience_has_pending_method = (
                result.objective_status.stop_reason == "no_improvement_patience_reached"
                and outstanding_assignment is not None
            )
            if not patience_has_pending_method:
                result.stopped_reason = result.objective_status.stop_reason
                _write_final_outputs(result)
                return result
        bound_round_index = _assigned_auto_round_index(
            outstanding_assignment,
            base_context.run_id,
        )
        start_round_index = (
            bound_round_index
            if execute and bound_round_index is not None
            else _next_executable_auto_round_index(base_context.run_root, base_context.run_id)
            if execute
            else 1
        )
        if execute and outstanding_assignment is not None and bound_round_index is None:
            start_round_index = _next_round_without_conflicting_queue(
                base_context.run_root,
                base_context.run_id,
                start_round_index,
                outstanding_assignment,
            )
        end_round_index = start_round_index + auto_rounds - 1
        parent = _parent_for_auto_round(
            base_orchestrator,
            start_round_index,
            bound_assignment=outstanding_assignment,
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
            parent_facts, error_fact_source_run_id = _planning_error_facts(parent)
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
            if error_fact_source_run_id != parent.context.run_id:
                _log_auto_round_event(
                    base_context,
                    event_type="auto_round_decision",
                    round_index=round_index,
                    total_rounds=end_round_index,
                    status="completed",
                    message=(
                        "Using inherited error facts as planning context; "
                        "they are not current child evidence."
                    ),
                    details={
                        "parent_run_id": parent.context.run_id,
                        "error_fact_source_run_id": error_fact_source_run_id,
                        "error_fact_count": len(parent_facts),
                        "evidence_role": "inherited_context",
                    },
                )

            child_run_id = (
                outstanding_assignment.assigned_run_id
                if outstanding_assignment is not None
                and _assigned_auto_round_index(
                    outstanding_assignment,
                    base_context.run_id,
                )
                == round_index
                and outstanding_assignment.assigned_run_id
                else f"{base_context.run_id}-r{round_index}"
            )
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
            recovered_round = self._recover_blocked_asha_evidence_round(
                round_index=round_index,
                parent=parent,
                child=child,
                scheduler=asha_scheduler,
                execute=execute,
                executor=executor,
                auto_import=auto_import,
            )
            asha_assignment = (
                None
                if recovered_round is not None
                else asha_scheduler.next_assignment(confirm_full_run=full_run_authorized)
            )
            outstanding_assignment = asha_assignment
            existing_round = (
                None
                if recovered_round is not None
                else _load_completed_round(child, round_index, parent.context.run_id, execute=execute)
            )
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
            if recovered_round is not None:
                round_result = recovered_round
            else:
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
                    if not _objective_stop_requires_method_replan(
                        result.objective_status,
                        asha_assignment=asha_assignment,
                        round_result=round_result,
                    ):
                        result.stopped_reason = result.objective_status.stop_reason
                        break
            if round_result.diversity_stop is not None and round_result.diversity_stop.should_stop:
                result.stopped_reason = round_result.diversity_stop.reason
                break
            if round_result.status != "completed" or round_result.stop_reason in {
                "no_guarded_candidates",
                "no_executable_candidates",
                "no_certified_paper_components",
                "method_candidates_exhausted",
                "paper_adapter_implementation_required",
                "queue_blocked",
                "resource_recovery_pending",
                "training_failed",
            }:
                result.stopped_reason = (
                    "method_candidates_exhausted"
                    if round_result.stop_reason == "no_new_asha_trials"
                    and result.objective_status is not None
                    and result.objective_status.stop_reason
                    == "no_improvement_patience_reached"
                    else round_result.stop_reason or round_result.status
                )
                break
            if round_result.stop_reason != "diversity_deferred":
                parent = child
        if not result.stopped_reason:
            result.stopped_reason = "requested_rounds_completed"
        _write_final_outputs(result)
        return result

    def _auto_certify(self, *, context: Any, report_path: Path) -> str | None:
        """Run one bounded mini-GPU certification attempt for a train command."""
        event_log = EventLog(context.events_path)
        event_log.append(
            run_id=context.run_id,
            event_type="gpu_certification_started",
            status="running",
            message=(
                "Running automatic mini GPU certification before candidate optimization."
            ),
            details={
                "model": self.certification_model,
                "device": self.certification_device,
                "recipe": self.certification_recipe,
            },
            artifacts={"certification_report": report_path},
        )
        try:
            suite = self.certification_suite or RealGpuAcceptanceSuite()
            report = suite.run(
                workdir=report_path.parent,
                model=self.certification_model,
                device=self.certification_device,
                recipe_id=self.certification_recipe,
                execute_real_gpu=True,
            )
            if not report_path.is_file():
                report.to_yaml(report_path, exclude_none=True, sort_keys=False)
            if report.status != "passed":
                reason = "; ".join(report.failures) or f"status={report.status}"
                event_log.append(
                    run_id=context.run_id,
                    event_type="gpu_certification_failed",
                    status="failed",
                    message=f"Automatic mini GPU certification failed: {reason}",
                    details={"status": report.status, "failures": report.failures},
                    artifacts={"certification_report": report_path},
                )
                return reason
        except Exception as exc:  # real backends must become a user-visible blocker
            reason = str(exc) or type(exc).__name__
            event_log.append(
                run_id=context.run_id,
                event_type="gpu_certification_failed",
                status="failed",
                message=f"Automatic mini GPU certification failed: {reason}",
                details={"exception_type": type(exc).__name__},
                artifacts={"certification_report": report_path},
            )
            return reason
        event_log.append(
            run_id=context.run_id,
            event_type="gpu_certification_completed",
            status="completed",
            message="Automatic mini GPU certification passed; candidate optimization may start.",
            details={"status": "passed"},
            artifacts={"certification_report": report_path},
        )
        return None

    def _recover_blocked_asha_evidence_round(
        self,
        *,
        round_index: int,
        parent: LoopOrchestrator,
        child: LoopOrchestrator,
        scheduler: ASHAScheduler,
        execute: bool,
        executor: str,
        auto_import: bool,
    ) -> AutoRoundResult | None:
        """Resume a blocked ASHA round by collecting evidence from completed checkpoints."""
        if not execute:
            return None
        summary_path = child.context.artifact_path("auto_round_summary.yaml")
        round_plan_path = child.context.artifact_path("round_execution_plan.yaml")
        if not summary_path.is_file() or not round_plan_path.is_file():
            return None
        try:
            previous = AutoRoundResult.model_validate(read_yaml(summary_path))
            round_plan = RoundExecutionPlan.from_yaml(round_plan_path)
        except ValueError:
            return None
        if (
            previous.round_index != round_index
            or previous.parent_run_id != parent.context.run_id
            or previous.status != "blocked"
            or previous.stop_reason != "asha_evidence_incomplete"
            or not round_plan.asha_assignment_id
        ):
            return None
        assignment = next(
            (
                item
                for item in scheduler.study.assignments
                if item.assignment_id == round_plan.asha_assignment_id
            ),
            None,
        )
        if assignment is None:
            return None
        trial = scheduler.study.trial(assignment.trial_id)
        candidate_node = next(
            (node for node in round_plan.execution_nodes if not _matched_baseline_node(node)),
            None,
        )
        if candidate_node is None:
            return None

        completeness = _persist_pilot_evidence_completeness(child, round_plan.execution_nodes)
        recovery_loop: TrainingLoopResult | None = None
        if any(not item.complete for item in completeness):
            _import_existing_coco_error_facts(child, round_plan.execution_nodes, completeness)
            completeness = _persist_pilot_evidence_completeness(child, round_plan.execution_nodes)
            if any(not item.complete for item in completeness):
                _enqueue_coco_evidence_recovery(child, round_plan.execution_nodes, completeness)
                recovery_loop = child.run_training_loop(
                    profile=("pilot" if assignment.stage_id.startswith("pilot") else "candidate_full"),
                    executor=executor,
                    max_steps=1,
                    auto_import=auto_import,
                )
                completeness = _persist_pilot_evidence_completeness(child, round_plan.execution_nodes)

        evidence_complete = bool(completeness) and all(item.complete for item in completeness)
        observation = _asha_observation(
            child,
            node=candidate_node,
            assignment=assignment,
            target_error_facts=trial.target_error_facts,
        )
        scheduler.report(assignment.trial_id, observation)
        child.next_round()
        training_loop = _merge_evidence_recovery_loop(
            previous.training_loop,
            recovery_loop,
            evidence_complete=evidence_complete and observation.evidence_complete,
        )
        result = previous.model_copy(
            update={
                "status": "completed" if observation.evidence_complete else "blocked",
                "stop_reason": (
                    "asha_assignment_completed"
                    if observation.evidence_complete
                    else "asha_evidence_incomplete"
                ),
                "training_loop": training_loop,
            }
        )
        write_yaml(summary_path, result.model_dump(mode="json"))
        EventLog(child.context.events_path).append(
            run_id=child.context.run_id,
            event_type="auto_round_decision",
            status="completed" if observation.evidence_complete else "blocked",
            message=(
                "Recovered fixed COCO evidence from completed checkpoints; ASHA observation was rebuilt."
                if observation.evidence_complete
                else "COCO evidence recovery remains incomplete; no ASHA promotion decision was made."
            ),
            details={
                "assignment_id": assignment.assignment_id,
                "candidate_id": candidate_node.candidate_config.candidate_id,
                "evidence_complete": observation.evidence_complete,
                "paired_result_verified": observation.paired_result_verified,
            },
        )
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
        shadow_evidence_only = _assignment_shadow_evidence_only(trial.source_node)
        round_plan = build_asha_assignment_plan(
            run_id=child.context.run_id,
            source_node=trial.source_node,
            stage_id=assignment.stage_id,
            epochs=1 if shadow_evidence_only else assignment.epochs,
            fraction=min(0.01, assignment.fraction) if shadow_evidence_only else assignment.fraction,
            seed=int(assignment.seed),
            seed_index=assignment.seed_index,
            run_name=run_name,
            baseline_control_node=trial.baseline_control_node,
            assignment_id=assignment.assignment_id,
        )
        if shadow_evidence_only:
            round_plan = _evidence_only_assignment_plan(round_plan)
        retry_queue = _load_frozen_assignment_retry_queue(
            child.context.run_dir,
            assignment,
            round_plan,
        )
        if retry_queue is None:
            _bind_child_run_protocol(
                child,
                round_plan,
                profile=("pilot" if assignment.stage_id.startswith("pilot") else "candidate_full"),
            )
        else:
            round_plan = _adopt_frozen_retry_nodes(round_plan, retry_queue)
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
        experiment_projection = round_plan.experiment_projection()
        round_plan.to_yaml(round_plan_path)
        experiment_projection.to_yaml(experiment_plan_path)
        if retry_queue is not None:
            retry_queue.metadata.update(
                {
                    "source_round_plan_hash": round_plan.plan_hash(),
                    "queue_source_plan_hash": experiment_projection.plan_hash(),
                    "frozen_resource_retry": True,
                    "frozen_run_protocol_hash": round_plan.run_protocol_hash,
                }
            )
            ExecutionQueueStore(child.context.run_dir).save(retry_queue)
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
        if execute and training_loop.completed and not shadow_evidence_only:
            completeness = _persist_pilot_evidence_completeness(
                child,
                round_plan.execution_nodes,
            )
            if any(not item.complete for item in completeness):
                _import_existing_coco_error_facts(child, round_plan.execution_nodes, completeness)
                completeness = _persist_pilot_evidence_completeness(child, round_plan.execution_nodes)
            if any(not item.complete for item in completeness):
                _enqueue_coco_evidence_recovery(child, round_plan.execution_nodes, completeness)
                recovery_loop = child.run_training_loop(
                    profile=("pilot" if assignment.stage_id.startswith("pilot") else "candidate_full"),
                    executor=executor,
                    max_steps=1,
                    auto_import=auto_import,
                )
                completeness = _persist_pilot_evidence_completeness(child, round_plan.execution_nodes)
                training_loop = _merge_evidence_recovery_loop(
                    training_loop,
                    recovery_loop,
                    evidence_complete=bool(completeness)
                    and all(item.complete for item in completeness),
                )
        status: Literal["completed", "blocked", "failed", "skipped"] = "completed"
        stop_reason = "asha_assignment_completed"
        if (
            execute
            and not training_loop.completed
            and training_loop.stopped_reason != "evidence_recovery_incomplete"
        ):
            status = "blocked"
            recovery_failure = _retryable_resource_failure(child.context.run_dir)
            if recovery_failure is not None:
                stop_reason = "resource_recovery_pending"
            else:
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
                if _candidate_training_failure_isolated(
                    child.context.run_dir,
                    candidate_id=assignment.candidate_id,
                ):
                    _mark_paper_candidate_disposition(
                        child,
                        trial.source_node,
                        disposition="blocked_runtime",
                        reasons=_candidate_training_failure_reason_codes(
                            child.context.run_dir,
                            candidate_id=assignment.candidate_id,
                        ),
                        source_stage="asha_execution",
                    )
                    status = "completed"
                    stop_reason = "asha_candidate_failed_isolated"
        elif execute:
            if _assignment_shadow_evidence_only(trial.source_node):
                activated, blockers = _activate_assignment_shadow_trial(
                    child,
                    scheduler,
                    trial_id=assignment.trial_id,
                    completed_node_id=candidate_node.node_id,
                )
                scheduler.complete_evidence_only_trial(
                    assignment.trial_id,
                    node_id=candidate_node.node_id,
                    reason=(
                        "assignment_shadow_evidence_collected_not_ranked"
                        if activated
                        else "assignment_shadow_activation_blocked:" + ";".join(blockers)
                    ),
                    succeeded=activated,
                )
                status = "completed" if activated else "blocked"
                stop_reason = (
                    "assignment_shadow_promoted_to_active"
                    if activated
                    else "assignment_shadow_evidence_invalid"
                )
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
                recipe_reason = _empty_recipe_round_reason(
                    child.context.artifact_path("loop_plan.yaml")
                )
                paper_stop_reason = child.context.metadata.get(
                    "paper_training_stop_reason"
                )
                if paper_stop_reason == "no_certified_paper_components":
                    status = "blocked"
                    stop_reason = paper_stop_reason
                elif recipe_reason:
                    status = "blocked"
                    stop_reason = recipe_reason
                elif diversity_reason == "family_exhaustion":
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
                    stop_reason = (
                        "no_certified_paper_components"
                        if child.context.metadata.get("paper_training_blocked") is True
                        else (
                            "paper_adapter_implementation_required"
                            if any(
                                item.execution_class == "adapter_required"
                                for item in assessments
                            )
                            else "no_executable_candidates"
                        )
                    )
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
                            stop_reason = (
                                "method_candidates_exhausted"
                                if child.context.metadata.get(
                                    "asha_registration_terminal_exhaustion"
                                )
                                is True
                                else "no_new_asha_trials"
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
            protocol_hash=str(
                (node.command_spec.metadata if node.command_spec is not None else {}).get("run_protocol_hash")
                or orchestrator.context.run_protocol_hash
                or ""
            ),
            evidence_role=(
                "baseline_reference"
                if _matched_baseline_node(node)
                else "current_observation"
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


def _import_existing_coco_error_facts(
    orchestrator: LoopOrchestrator,
    nodes: list[ExperimentNode],
    results: list[PilotEvidenceCompletenessResult],
) -> list[PilotEvidenceCompletenessResult]:
    """Import missing facts from completed COCO artifacts without rerunning eval."""
    incomplete = {item.node_id: item for item in results if not item.complete}
    fact_store = ErrorFactStore(orchestrator.evidence_store.root)
    imported: list[str] = []
    for node in nodes:
        result = incomplete.get(node.node_id)
        if result is None or not result.evidence_actions:
            continue
        recovery_actions = set(result.evidence_actions)
        if not recovery_actions.issubset({"import_current_node_error_facts"}):
            continue
        evidence = orchestrator.evidence_store.load_run(orchestrator.context.run_id)
        role = "baseline_reference" if _matched_baseline_node(node) else "current_observation"
        report_entry = next(
            (
                entry for entry in evidence.artifact_manifest
                if entry.run_id == orchestrator.context.run_id
                and entry.candidate_id == node.candidate_config.candidate_id
                and entry.node_id == node.node_id
                and entry.protocol_hash == result.protocol_hash
                and entry.name.endswith("coco_error_report")
                and entry.verify()
            ),
            None,
        )
        if report_entry is None:
            continue
        try:
            report = json.loads(report_entry.path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        metric_records = [
            record for record in evidence.metric_records
            if record.run_id == orchestrator.context.run_id
            and record.candidate_id == node.candidate_config.candidate_id
            and record.node_id == node.node_id
            and record.protocol_hash == result.protocol_hash
            and record.evidence_role == role
            and record.inheritance_depth == 0
            and record.verified
        ]
        metrics = {record.metric_name: record.value for record in metric_records}
        facts = [
            fact.model_copy(update={"evidence_role": role})
            for fact in build_error_facts_from_coco_error_report(
                report=report,
                run_id=orchestrator.context.run_id,
                candidate_id=node.candidate_config.candidate_id,
                node_id=node.node_id,
                dataset_version=node.data_version,
                split="val2017",
                source="existing_coco_error_report_import",
                source_artifact=report_entry.path,
            )
        ]
        eval_entry = next(
            (
                entry for entry in evidence.artifact_manifest
                if entry.run_id == orchestrator.context.run_id
                and entry.candidate_id == node.candidate_config.candidate_id
                and entry.node_id == node.node_id
                and entry.protocol_hash == result.protocol_hash
                and entry.name.endswith("coco_eval")
                and entry.verify()
            ),
            None,
        )
        facts.extend(
            fact.model_copy(update={"evidence_role": role})
            for fact in build_error_facts_from_coco_metrics(
                metrics=metrics,
                run_id=orchestrator.context.run_id,
                candidate_id=node.candidate_config.candidate_id,
                node_id=node.node_id,
                dataset_version=node.data_version,
                split="val2017",
                source="existing_coco_metrics_import",
                source_artifact=eval_entry.path if eval_entry is not None else report_entry.path,
            )
        )
        if not facts:
            continue
        if not metric_records:
            continue
        identity = {
            "protocol_hash": result.protocol_hash,
            "dataset_manifest_sha256": metric_records[0].dataset_manifest_sha256,
            "subset_manifest_sha256": metric_records[0].subset_manifest_sha256,
            "eval_protocol_hash": metric_records[0].eval_protocol_hash,
            "seed": metric_records[0].seed,
            "fidelity": metric_records[0].fidelity,
            "epochs": metric_records[0].epochs,
            "batch_policy_hash": metric_records[0].batch_policy_hash,
            "ultralytics_version": metric_records[0].ultralytics_version,
            "imgsz": metric_records[0].imgsz,
            "evidence_role": role,
        }
        facts = [fact.model_copy(update=identity) for fact in facts]
        fact_store.replace_current_node(
            orchestrator.context.run_id,
            node.candidate_config.candidate_id,
            node.node_id,
            result.protocol_hash,
            facts,
            evidence_role=role,
        )
        imported.append(node.node_id)
    if imported:
        EventLog(orchestrator.context.events_path).append(
            run_id=orchestrator.context.run_id,
            event_type="auto_round_decision",
            status="completed",
            message="Imported missing structured COCO facts from existing evaluation artifacts; no evaluation was rerun.",
            details={"node_ids": imported, "action": "import_current_node_error_facts"},
        )
    return _persist_pilot_evidence_completeness(orchestrator, nodes) if imported else results


def _enqueue_coco_evidence_recovery(
    orchestrator: LoopOrchestrator,
    nodes: list[ExperimentNode],
    results: list[PilotEvidenceCompletenessResult],
) -> ExecutionQueue:
    """Replace the queue with recovery-only actions for incomplete current-node evidence."""
    incomplete_ids = {
        item.node_id
        for item in results
        if not item.complete
        and any(action != "import_current_node_error_facts" for action in item.evidence_actions)
    }
    results_by_node = {item.node_id: item for item in results}
    items: list[ExecutionQueueItem] = []
    for node in nodes:
        if node.node_id not in incomplete_ids or node.command_spec is None:
            continue
        source_spec = node.command_spec
        completeness = results_by_node[node.node_id]
        evidence = orchestrator.evidence_store.load_run(orchestrator.context.run_id)
        best_pt = next(
            (
                entry.path
                for entry in reversed(evidence.artifact_manifest)
                if entry.run_id == orchestrator.context.run_id
                and entry.candidate_id == node.candidate_config.candidate_id
                and entry.node_id == node.node_id
                and entry.protocol_hash == completeness.protocol_hash
                and entry.name.endswith("best_pt")
                and entry.verify()
            ),
            source_spec.expected_artifacts.get("best_pt"),
        )
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
                "source_training_argv": json.dumps(list(source_spec.argv)),
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


def _merge_evidence_recovery_loop(
    original: TrainingLoopResult | None,
    recovery: TrainingLoopResult | None,
    *,
    evidence_complete: bool,
) -> TrainingLoopResult | None:
    if recovery is None:
        return original
    if original is None:
        return recovery.model_copy(
            update={
                "completed": evidence_complete,
                "stopped_reason": "complete" if evidence_complete else "evidence_recovery_incomplete",
            }
        )
    return recovery.model_copy(
        update={
            "steps": [*original.steps, *recovery.steps],
            "completed": evidence_complete,
            "stopped_reason": "complete" if evidence_complete else "evidence_recovery_incomplete",
        }
    )


def _mark_paper_candidate_disposition(
    child: LoopOrchestrator,
    node: ExperimentNode,
    *,
    disposition: ProposalDisposition,
    reasons: list[str],
    source_stage: str,
) -> None:
    """Persist a downstream candidate decision, recovering omitted upstream records."""
    objective = load_optimization_objective(
        child.context.metadata.get("optimization_objective_path")
    )
    ledger = PaperCandidateCoverageLedger(
        child.context.artifact_path("paper_candidate_coverage.yaml"),
        run_id=child.context.run_id,
        protocol_hash=(objective.baseline_protocol_hash if objective is not None else "unknown"),
    )
    candidate = node.candidate_config
    updated = ledger.update_candidate_disposition(
        candidate_id=candidate.candidate_id,
        disposition=disposition,
        reason_codes=reasons,
        source_stage=source_stage,
        node_id=node.node_id,
    )
    if updated is not None or not candidate.components:
        return
    metadata = node.command_spec.metadata if node.command_spec is not None else {}
    recipe_id = str(
        metadata.get("component_recipe_id")
        or candidate.action_id
        or candidate.candidate_id
    )
    recipe_version = str(
        metadata.get("component_recipe_version")
        or candidate.train_overrides.get("recipe_version")
        or "unknown"
    )
    payload = {
        "recipe_id": recipe_id,
        "recipe_version": recipe_version,
        "candidate_id": candidate.candidate_id,
        "components": sorted(candidate.components),
        "protocol_hash": ledger.protocol_hash,
        "dataset": metadata.get("dataset_manifest_sha256") or node.data_version,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    ledger.ensure_runtime_candidate(
        candidate_id=candidate.candidate_id,
        recipe_id=recipe_id,
        recipe_version=recipe_version,
        component_ids=list(candidate.components),
        execution_fingerprint=fingerprint,
        disposition=disposition,
        reason_codes=reasons,
        source_stage=source_stage,
        node_id=node.node_id,
    )


def _register_guarded_pilot_trials(
    scheduler: ASHAScheduler,
    child: LoopOrchestrator,
    executable_nodes: list[ExperimentNode],
) -> int:
    """Register guarded recipes without granting them training budget directly."""
    considered = 0
    terminal_rejections = 0
    retryable_rejections = 0
    plan_path = child.context.artifact_path("round_execution_plan.yaml")
    if not plan_path.is_file() or not executable_nodes:
        return 0
    plan = RoundExecutionPlan.from_yaml(plan_path)
    source_by_candidate = {
        node.candidate_config.candidate_id: node
        for node in plan.deferred_nodes
        if not _matched_baseline_node(node)
    }
    def mark(node: ExperimentNode, disposition: str, reasons: list[str]) -> None:
        _mark_paper_candidate_disposition(
            child,
            node,
            disposition=disposition,  # type: ignore[arg-type]
            reasons=reasons,
            source_stage="asha_registration",
        )

    objective = load_optimization_objective(
        child.context.metadata.get("optimization_objective_path")
    )

    overall_map_goal = _is_overall_map_goal(objective)
    eligible_sources = [
        source_by_candidate.get(node.candidate_config.candidate_id, node)
        for node in executable_nodes
        if not _matched_baseline_node(node)
    ]
    if overall_map_goal:
        eligible_sources = [
            source
            for source in eligible_sources
            if not _small_object_specific_node(source)
        ]
    adapter_candidates_available = any(
        _adapter_backed_node(source) for source in eligible_sources
    )
    baseline_control = next(
        (node for node in plan.deferred_nodes if _matched_baseline_node(node)),
        None,
    )
    registered = 0
    policy_budget = getattr(getattr(child, "policy", None), "policy_budget", {})
    scalar_hpo_allowed = BudgetPolicy.model_validate(policy_budget).allow_scalar_hpo
    existing_trial_ids = {trial.trial_id for trial in scheduler.study.trials}
    effective_contracts: dict[str, ComponentContract] | None = None
    for node in executable_nodes:
        if _matched_baseline_node(node):
            continue
        considered += 1
        source = source_by_candidate.get(node.candidate_config.candidate_id, node)
        if overall_map_goal and _small_object_specific_node(source):
            terminal_rejections += 1
            mark(source, "incompatible", ["small_object_method_out_of_scope_for_overall_map"])
            EventLog(child.context.events_path).append(
                run_id=child.context.run_id,
                event_type="auto_round_decision",
                status="completed",
                message=(
                    f"Deferred {source.candidate_config.candidate_id}: the objective targets "
                    "overall mAP, not a small-object-only metric."
                ),
                details={
                    "candidate_id": source.candidate_config.candidate_id,
                    "adapter_ids": source.candidate_config.components,
                    "reason": "small_object_method_out_of_scope_for_overall_map",
                    "budget_authority": "ASHA",
                },
            )
            continue
        if adapter_candidates_available and not _adapter_backed_node(source):
            retryable_rejections += 1
            mark(source, "deferred_budget", ["native_fallback_deferred_for_adapter_methods"])
            EventLog(child.context.events_path).append(
                run_id=child.context.run_id,
                event_type="auto_round_decision",
                status="completed",
                message=(
                    f"Deferred {source.candidate_config.candidate_id}: executable adapter-backed "
                    "methods take precedence over native YOLO tuning."
                ),
                details={
                    "candidate_id": source.candidate_config.candidate_id,
                    "reason": "native_fallback_deferred_for_adapter_methods",
                    "budget_authority": "ASHA",
                },
            )
            continue
        if (
            source.command_spec is not None
            and source.command_spec.metadata.get("matched_pilot_required") is True
            and baseline_control is None
        ):
            retryable_rejections += 1
            mark(source, "blocked_runtime", ["matched_baseline_control_missing"])
            continue
        if source.candidate_config.search_tier == "scalar_hpo" and not scalar_hpo_allowed:
            terminal_rejections += 1
            mark(source, "deferred_budget", ["scalar_hpo_disabled"])
            EventLog(child.context.events_path).append(
                run_id=child.context.run_id,
                event_type="auto_round_decision",
                status="blocked",
                message=(
                    f"Skipped {source.candidate_config.candidate_id}: scalar HPO is disabled."
                ),
                details={
                    "candidate_id": source.candidate_config.candidate_id,
                    "scalar_hpo_enabled": False,
                    "budget_authority": "ASHA",
                },
            )
            continue
        if source.candidate_config.components:
            if effective_contracts is None:
                effective_contracts = {
                    item.component_id: item
                    for item in _load_execution_contracts(child)
                }
            component_certification = ComponentQueueCertificationGate().evaluate(
                component_ids=list(source.candidate_config.components),
                report_path=child.context.metadata.get("gpu_certification_report"),
                component_contracts=effective_contracts,
            )
            if not component_certification.allowed:
                retryable_rejections += 1
                mark(source, "blocked_runtime", component_certification.blockers)
                EventLog(child.context.events_path).append(
                    run_id=child.context.run_id,
                    event_type="auto_round_decision",
                    status="blocked",
                    message=(
                        f"Skipped {source.candidate_config.candidate_id}: component "
                        "end-to-end certification is incomplete."
                    ),
                    details={
                        "candidate_id": source.candidate_config.candidate_id,
                        "adapter_ids": source.candidate_config.components,
                        "blocked_by": component_certification.blockers,
                        "certification_report": (
                            component_certification.report_path.as_posix()
                            if component_certification.report_path is not None
                            else None
                        ),
                        "budget_authority": "ASHA",
                    },
                )
                continue
            runtime_errors = validate_certified_runtime_node(source)
            if runtime_errors:
                retryable_rejections += 1
                mark(source, "blocked_runtime", runtime_errors)
                EventLog(child.context.events_path).append(
                    run_id=child.context.run_id,
                    event_type="auto_round_decision",
                    status="blocked",
                    message=(
                        f"Skipped {source.candidate_config.candidate_id}: certified adapter "
                        "runtime is incomplete."
                    ),
                    details={
                        "candidate_id": source.candidate_config.candidate_id,
                        "adapter_ids": source.candidate_config.components,
                        "blocked_by": runtime_errors,
                        "budget_authority": "ASHA",
                    },
                )
                continue
        trial_id = f"{scheduler.study.base_run_id}:{node.candidate_config.candidate_id}"
        raw_targets = source.candidate_config.target_error_facts
        target_error_facts = [
            dict(item)
            for item in raw_targets
            if isinstance(item, dict)
        ]
        if not target_error_facts:
            retryable_rejections += 1
            mark(source, "evidence_recovery", ["target_error_facts_missing"])
            continue
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
            mark(source, "queued", ["asha_trial_registered"])
            metadata = source.command_spec.metadata if source.command_spec is not None else {}
            DecisionLedger(
                child.context.artifact_path("decision_ledger.jsonl")
            ).append(DecisionLedgerRecord(
                run_id=child.context.run_id,
                policy_id=source.candidate_config.action_id or trial.trial_id,
                decision_type="paper_recipe_asha_registration",
                proposal={
                    "candidate_id": source.candidate_config.candidate_id,
                    "adapter_ids": source.candidate_config.components,
                    "adapter_patch_hash": metadata.get("adapter_patch_hash"),
                    "adapter_runtime_payload_hash": metadata.get(
                        "adapter_runtime_payload_hash"
                    ),
                    "component_certification_report_hash": (
                        component_certification.report_hash
                        if source.candidate_config.components
                        else None
                    ),
                    "queue_authority": "ASHA/RoundExecutionPlan",
                    "scalar_hpo_enabled": scalar_hpo_allowed,
                },
                decision="registered",
                created_candidate_id=source.candidate_config.candidate_id,
                created_node_id=source.node_id,
                rationale="Certified runtime recipe registered; ASHA owns pilot budget.",
                policy_version="paper_recipe_materialization_gate.v1",
            ))
        else:
            terminal_rejections += 1
            mark(source, "already_tested", ["asha_trial_already_registered"])
    metadata = getattr(child.context, "metadata", None)
    if isinstance(metadata, dict):
        metadata["asha_registration_summary"] = {
            "considered": considered,
            "registered": registered,
            "terminal_rejections": terminal_rejections,
            "retryable_rejections": retryable_rejections,
        }
        metadata["asha_registration_terminal_exhaustion"] = bool(
            considered > 0
            and registered == 0
            and terminal_rejections == considered
            and retryable_rejections == 0
        )
    return registered


def _adapter_backed_node(node: ExperimentNode) -> bool:
    command = node.command_spec
    metadata = command.metadata if command is not None else {}
    return bool(
        node.candidate_config.components
        and metadata.get("adapter_runtime_entrypoint")
    )


def _small_object_specific_node(node: ExperimentNode) -> bool:
    config = node.candidate_config
    tokens = {
        config.candidate_id.lower(),
        str(config.action_id or "").lower(),
        *(component.lower() for component in config.components),
    }
    return any(
        "small_object" in token
        or token in {"head.p2_small_object", "sampling.small_object"}
        for token in tokens
    )


def _is_overall_map_goal(objective: OptimizationObjective | None) -> bool:
    if objective is None or objective.primary_metric != "map50_95":
        return False
    description = str(objective.goal_description or "").lower()
    return "overall" in description or "整体" in description


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
    objective = load_optimization_objective(
        child.context.metadata.get("optimization_objective_path")
    )
    primary_metric = objective.primary_metric if objective is not None else "map50_95"
    paired_result = build_paired_experiment_result(
        run_id=child.context.run_id,
        candidate_id=node.candidate_config.candidate_id,
        candidate_node_id=node.node_id,
        metric_records=evidence.metric_records,
        error_facts=facts,
        primary_metric=primary_metric,
        target_error_facts=[dict(item) for item in target_error_facts],
        additional_metrics=(
            ["map50_95"] if primary_metric != "map50_95" else None
        ),
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
    primary_delta = paired_result.metric_deltas.get(primary_metric)
    paired_delta_value = primary_delta.effect_delta if primary_delta is not None else None
    improved_count = sum(1 for item in paired_result.target_error_fact_deltas if item.improved)
    requires_target_facts = assignment.stage_id in {
        "pilot_10",
        "candidate_full_seed_1",
        "candidate_full_confirmation",
    }
    diagnosis_result = None
    if assignment.stage_id != "pilot_3":
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
    evidence_complete = (
        paired_result.verified
        and paired_delta_value is not None
        and (
            not requires_target_facts
            or (
                bool(target_error_facts)
                and bool(paired_result.target_error_fact_deltas)
                and all(item.verified for item in paired_result.target_error_fact_deltas)
            )
        )
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


def _candidate_training_failure_isolated(
    run_dir: Path,
    *,
    candidate_id: str,
) -> bool:
    """Return true only when the candidate failed and its control did not."""
    try:
        queue = ExecutionQueueStore(run_dir).load()
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
    failed = [item for item in queue.items if item.status == "failed"]
    if not failed:
        return False
    control_failed = any(
        item.command.metadata.get("matched_baseline_control") is True
        for item in failed
    )
    candidate_failed = any(item.candidate_id == candidate_id for item in failed)
    return candidate_failed and not control_failed


def _candidate_training_failure_reason_codes(
    run_dir: Path,
    *,
    candidate_id: str,
) -> list[str]:
    """Return stable runtime blocker codes for one failed candidate queue item."""
    try:
        queue = ExecutionQueueStore(run_dir).load()
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return ["candidate_training_failed"]
    item = next(
        (
            candidate
            for candidate in queue.items
            if candidate.candidate_id == candidate_id and candidate.status == "failed"
        ),
        None,
    )
    if item is None or item.last_result is None:
        return ["candidate_training_failed"]
    failure = item.last_result.failure or classify_execution_failure(
        stdout=item.last_result.stdout,
        stderr=item.last_result.stderr,
        command=item.last_result.command,
    )
    return [failure.kind if failure is not None else "candidate_training_failed"]


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
                    version=str(
                        candidate.train_overrides.get("recipe_version")
                        or "execution-bridge.v1"
                    ),
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
                adapter_ids=(
                    [
                        item.strip()
                        for item in str(command.metadata.get("component_ids") or "").split(",")
                        if item.strip()
                    ]
                    if command is not None
                    else []
                ),
                adapter_patch_hash=(
                    str(command.metadata.get("adapter_patch_hash"))
                    if command is not None and command.metadata.get("adapter_patch_hash")
                    else None
                ),
                adapter_runtime_payload_hash=(
                    str(command.metadata.get("adapter_runtime_payload_hash"))
                    if command is not None
                    and command.metadata.get("adapter_runtime_payload_hash")
                    else None
                ),
                command=command.display() if command is not None else "",
            )
        )
    return assessments


def _ensure_next_round(orchestrator: LoopOrchestrator) -> dict[str, Any]:
    path = orchestrator.context.artifact_path("next_round.yaml")
    if not path.is_file():
        orchestrator.next_round()
    return read_yaml(path) if path.is_file() else {}


def _planning_error_facts(orchestrator: LoopOrchestrator) -> tuple[list[ErrorFact], str | None]:
    """Return current facts or nearest same-dataset ancestor facts for planning only.

    A child that merely registered ASHA trials has no current observation yet.  It may
    use its parent's diagnosis as context to issue the first pilot assignment, but
    those inherited facts are never copied into the child's evidence store and are
    therefore unavailable to paired deltas, promotion, or contribution accounting.
    """
    store = ErrorFactStore(orchestrator.context.run_root)
    current: LoopOrchestrator | None = orchestrator
    visited: set[str] = set()
    expected_manifest = orchestrator.context.dataset_manifest_sha256
    while current is not None and current.context.run_id not in visited:
        visited.add(current.context.run_id)
        facts = store.read(current.context.run_id)
        if facts:
            compatible = [
                fact
                for fact in facts
                if _error_fact_matches_dataset(
                    fact,
                    dataset_version=orchestrator.context.dataset_version,
                    dataset_manifest_sha256=expected_manifest,
                )
            ]
            if compatible:
                return compatible, current.context.run_id
        parent_dir = current.context.metadata.get("parent_run_dir")
        if not isinstance(parent_dir, str) or not parent_dir:
            break
        parent_path = Path(parent_dir)
        if not (parent_path / "run_context.yaml").is_file():
            break
        current = LoopOrchestrator.from_run_dir(parent_path)
    return [], None


def _error_fact_matches_dataset(
    fact: ErrorFact,
    *,
    dataset_version: str,
    dataset_manifest_sha256: str | None,
) -> bool:
    """Accept legacy facts without a manifest only within the same dataset version.

    COCO error import predates manifest propagation, so those facts have a null
    manifest even though their dataset and evaluation protocol are recorded. A
    missing manifest is weaker evidence than an exact match, but rejecting it
    unconditionally turns real post-eval artifacts into a false blocker.
    """
    if fact.dataset_version != dataset_version:
        return False
    if fact.dataset_manifest_sha256 is None:
        return True
    return dataset_manifest_sha256 is None or fact.dataset_manifest_sha256 == dataset_manifest_sha256


def _fork_or_load_child(parent: LoopOrchestrator, child_run_id: str) -> LoopOrchestrator:
    child_dir = parent.context.run_root / child_run_id
    if child_dir.exists():
        return LoopOrchestrator.from_run_dir(child_dir)
    return parent.fork_next(child_run_id)


def _assigned_auto_round_index(
    assignment: ASHAAssignment | None,
    base_run_id: str,
) -> int | None:
    """Return the exact child round already claimed by an ASHA assignment."""
    if assignment is None or not assignment.assigned_run_id:
        return None
    match = re.fullmatch(
        rf"{re.escape(base_run_id)}-r(?P<index>[1-9]\d*)",
        assignment.assigned_run_id,
    )
    return int(match.group("index")) if match is not None else None


def _next_round_without_conflicting_queue(
    run_root: Path,
    base_run_id: str,
    start_index: int,
    assignment: ASHAAssignment,
) -> int:
    """Avoid reusing a child whose active queue belongs to another ASHA plan."""
    index = start_index
    while True:
        queue_path = run_root / f"{base_run_id}-r{index}" / "execution_queue.yaml"
        if not queue_path.is_file():
            return index
        try:
            queue = ExecutionQueue.from_yaml(queue_path)
        except (OSError, ValueError, TypeError):
            return index
        counts = queue.counts()
        has_active_items = any(
            counts.get(status, 0) > 0
            for status in ("queued", "running", "paused", "blocked_by_resource", "needs_resume", "needs_evidence")
        )
        if not has_active_items:
            return index
        queue_assignment = str(queue.metadata.get("asha_assignment_id") or "")
        queue_stage = str(queue.metadata.get("source_round_stage") or "")
        if queue_assignment == assignment.assignment_id and queue_stage == assignment.stage_id:
            return index
        index += 1


def _parent_for_auto_round(
    base: LoopOrchestrator,
    round_index: int,
    *,
    bound_assignment: ASHAAssignment | None,
) -> LoopOrchestrator:
    """Restore the persisted parent when resuming an assignment-bound child."""
    bound_index = _assigned_auto_round_index(bound_assignment, base.context.run_id)
    if bound_index == round_index and bound_assignment is not None:
        child_dir = base.context.run_root / str(bound_assignment.assigned_run_id)
        summary_path = child_dir / "artifacts" / "auto_round_summary.yaml"
        if summary_path.is_file():
            try:
                summary = AutoRoundResult.model_validate(read_yaml(summary_path))
            except (OSError, ValueError, TypeError):
                summary = None
            if summary is not None:
                parent_dir = base.context.run_root / summary.parent_run_id
                if (parent_dir / "run_context.yaml").is_file():
                    return LoopOrchestrator.from_run_dir(parent_dir)
    return (
        _latest_completed_auto_child(base, round_index - 1)
        if round_index > 1
        else base
    )


def _retryable_resource_failure(run_dir: Path) -> ExecutionFailure | None:
    """Return a bounded infrastructure failure from a child execution queue."""
    queue_path = run_dir / "execution_queue.yaml"
    if not queue_path.is_file():
        return None
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
    except (OSError, ValueError):
        return None
    return _queue_retryable_resource_failure(queue)


def _queue_retryable_resource_failure(queue: ExecutionQueue) -> ExecutionFailure | None:
    """Return a recoverable failure while honoring the queue's latest command state."""
    for item in queue.items:
        result = item.last_result
        if result is None or result.status != "failed":
            continue
        failure = classify_execution_failure(
            stdout=result.stdout,
            stderr=result.stderr,
            command=item.command,
        ) or result.failure
        if failure is not None and failure.recoverable:
            return failure
    return None


def _load_frozen_assignment_retry_queue(
    run_dir: Path,
    assignment: ASHAAssignment,
    round_plan: RoundExecutionPlan,
) -> ExecutionQueue | None:
    """Load an interrupted paired cohort without rebinding it to the current code protocol."""
    queue_path = run_dir / "execution_queue.yaml"
    if not queue_path.is_file():
        return None
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
    except (OSError, ValueError):
        return None
    if queue.metadata.get("asha_assignment_id") != assignment.assignment_id:
        return None
    if _queue_retryable_resource_failure(queue) is None:
        return None
    expected_nodes = {node.node_id for node in round_plan.execution_nodes}
    if {item.node_id for item in queue.items} != expected_nodes:
        return None
    protocols = {
        str(item.command.metadata.get("run_protocol_hash") or "")
        for item in queue.items
    }
    protocols.discard("")
    if len(protocols) != 1:
        return None
    return queue


def _adopt_frozen_retry_nodes(
    round_plan: RoundExecutionPlan,
    queue: ExecutionQueue,
) -> RoundExecutionPlan:
    """Keep candidate/control commands protocol-identical across an infrastructure retry."""
    by_node = {item.node_id: item.experiment_node for item in queue.items}
    nodes = [by_node[node.node_id] for node in round_plan.execution_nodes]
    protocol_hash = str(next(iter({item.command.metadata["run_protocol_hash"] for item in queue.items})))
    return round_plan.model_copy(
        update={
            "execution_nodes": nodes,
            "run_protocol_hash": protocol_hash,
        }
    )


def _reopen_retryable_resource_assignments(context: Any, scheduler: ASHAScheduler) -> int:
    """Undo ASHA elimination caused solely by a recoverable execution failure."""
    reopened = 0
    for assignment in scheduler.study.assignments:
        if assignment.status != "failed" or not assignment.assigned_run_id:
            continue
        failure = _retryable_resource_failure(context.run_root / assignment.assigned_run_id)
        stale_queue_conflict = _stale_assignment_queue_conflict(
            context.run_root,
            assignment,
        )
        if failure is None and not stale_queue_conflict:
            continue
        trial = scheduler.study.trial(assignment.trial_id)
        observation = trial.observation(assignment.stage_id, assignment.seed_index)
        if observation is None or observation.failure_reason not in {
            "queue_blocked",
            "resource_recovery_pending",
            "training_failed",
        }:
            continue
        trial.observations = [
            item
            for item in trial.observations
            if not (
                item.stage_id == assignment.stage_id
                and item.seed_index == assignment.seed_index
            )
        ]
        trial.status = _retry_trial_status(assignment.stage_id)
        trial.pending_stage = assignment.stage_id
        trial.eliminated_reason = ""
        assignment.status = "issued"
        assignment.started_at = None
        assignment.completed_at = None
        if stale_queue_conflict:
            # The old child still owns a different active round plan. Keep its
            # artifacts intact, but let the scheduler allocate a fresh child.
            assignment.assigned_run_id = None
            assignment.assigned_node_id = None
        reopened += 1
    return reopened


def _stale_assignment_queue_conflict(
    run_root: Path,
    assignment: ASHAAssignment,
) -> bool:
    """Detect a failed assignment bound to a child whose active queue is stale."""
    if not assignment.assigned_run_id:
        return False
    child_dir = run_root / assignment.assigned_run_id
    queue_path = child_dir / "execution_queue.yaml"
    plan_path = child_dir / "artifacts" / "round_execution_plan.yaml"
    if not queue_path.is_file() or not plan_path.is_file():
        return False
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
        plan = RoundExecutionPlan.from_yaml(plan_path)
    except (OSError, ValueError, TypeError):
        return False
    counts = queue.counts()
    has_active_items = any(
        counts.get(status, 0) > 0
        for status in ("queued", "running", "paused", "blocked_by_resource", "needs_resume", "needs_evidence")
    )
    if not has_active_items:
        return False
    queue_hash = str(queue.metadata.get("source_round_plan_hash") or "")
    return bool(queue_hash) and queue_hash != plan.plan_hash()


def _retry_trial_status(stage_id: str) -> str:
    if stage_id == "pilot_3":
        return "waiting"
    if stage_id == "pilot_10":
        return "promotion_pending"
    if stage_id == "candidate_full_seed_1":
        return "full_pending_confirmation"
    return "confirmation_pending"


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
        "snapshot_status",
        "stale_reasons",
        "paper_method_coverage_version",
        "effective_maturity_version",
        "asha_state_path",
        "gpu_certification_report",
        "gpu_certification_report_hash",
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
            # This is planning context only. Do not write these facts into the
            # child store, where they could be mistaken for candidate evidence.
            "inherited_planning_error_facts": [fact.model_dump(mode="json") for fact in parent_facts][-100:],
            "inherited_planning_error_fact_source_run_ids": sorted(
                {str(fact.origin_run_id or fact.run_id) for fact in parent_facts}
            ),
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
                    if (
                        item.candidate_config is None
                        or not item.diversity_reason
                        or item.diversity_reason == "diversity_guards_passed"
                    ):
                        continue
                    action_id = item.candidate_config.action_id
                    if action_id:
                        tried.append(str(action_id))
    return list(dict.fromkeys(tried))


def _objective_stop_requires_method_replan(
    status: OptimizationObjectiveStatus,
    *,
    asha_assignment: ASHAAssignment | None,
    round_result: AutoRoundResult,
) -> bool:
    """Let bounded method planning exhaust its queue before patience stops the run."""
    if status.stop_reason != "no_improvement_patience_reached":
        return False
    return (
        asha_assignment is not None
        or round_result.stop_reason == "asha_candidates_registered"
    )


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
    portfolio_path = child.context.artifact_path("executable_portfolio.yaml")
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
                    "snapshot_status": snapshot.snapshot_status,
                    "stale_reasons": snapshot.stale_reasons,
                    "paper_intelligence": snapshot.paper_intelligence,
                    "unavailable_reason": snapshot.unavailable_reason,
                    "research_network_allowed": False,
                    "maturity_summary": snapshot.maturity_summary.model_dump(mode="json"),
                    "paper_method_coverage_version": snapshot.paper_method_coverage_version,
                    "effective_maturity_version": snapshot.effective_maturity_version,
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
        contracts, effective_maturity = _resolve_effective_component_contracts(
            child,
            contracts,
        )
        component_registry = ComponentRegistry(contracts)  # type: ignore[arg-type]
        paper_registry = PaperRegistry(paper_root)
        # Training is bound to the frozen snapshot. Local reviewed recipes are
        # merged by ResearchProductionPipeline before the snapshot is hashed.
        recipe_sources = [
            recipes_path
        ] if recipes_path is not None and recipes_path.exists() else []
        recipe_registry = RecipeRegistry.from_paths(
            recipe_sources,
            component_contracts=contracts if snapshot_ref is not None else (),
            strict=False,
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
        method_coverage_path = paper_root / "paper_method_coverage.yaml"
        plan, method_profile_bindings = _apply_paper_method_profile_gate(
            plan,
            recipe_registry=recipe_registry,
            coverage_path=method_coverage_path,
            require_frozen_coverage=snapshot_ref is not None,
        )
        paper_recipe_candidates = [
            item
            for item in [
                *plan.selected_recipes,
                *plan.deferred_recipes,
                *plan.rejected_recipes,
            ]
            if item.related_papers or method_profile_bindings.get(item.recipe_id)
        ]
        paper_training_blocked = bool(
            paper_recipe_candidates and not plan.selected_recipes
        )
        child.context.metadata.update(
            {
                "paper_training_blocked": paper_training_blocked,
                "paper_training_stop_reason": (
                    "no_certified_paper_components"
                    if paper_training_blocked
                    else None
                ),
            }
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
                    "adapter_hash": effective_maturity[item.component_id].adapter_hash,
                    "maturity_evidence_source": effective_maturity[
                        item.component_id
                    ].evidence_source,
                    "maturity_overlay_status": effective_maturity[
                        item.component_id
                    ].overlay_status,
                    "maturity_rejection_reasons": effective_maturity[
                        item.component_id
                    ].rejection_reasons,
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
        paper_component_decisions = _paper_component_decision_rows(
            plan=plan,
            recipe_registry=recipe_registry,
            coverage_path=method_coverage_path,
            effective_maturity=effective_maturity,
            method_profile_bindings=method_profile_bindings,
        )
        compatibility_snapshot["paper_component_decisions"] = (
            paper_component_decisions
        )
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
            if (
                planned.decision in {"selected", "deferred"}
                and report.accepted
                and isinstance(recipe, (AtomicRecipe, CoupledRecipe))
            ):
                executable_pilot_policies.extend(
                    _candidate_policies_from_recipe(
                        child,
                        recipe,
                        parent_facts,
                        planned.utility,
                        paper_ids=method_profile_bindings.get(recipe.recipe_id, []),
                    )
                )
        _write_paper_candidate_coverage(
            child=child,
            plan=plan,
            recipe_registry=recipe_registry,
            method_profile_bindings=method_profile_bindings,
            critic_reports=recipe_critic_reports,
        )
        planned_recipes = [
            *plan.selected_recipes,
            *plan.deferred_recipes,
            *plan.rejected_recipes,
        ]
        coverage_payload = (
            read_yaml(method_coverage_path)
            if method_coverage_path.is_file()
            else {}
        )
        coverage_profiles = (
            coverage_payload.get("profiles", [])
            if isinstance(coverage_payload, dict)
            else []
        )
        runtime_component_ids = {
            component_id
            for component_id, maturity in effective_maturity.items()
            if maturity.valid_for_training
        }
        runtime_recipes = [
            recipe
            for recipe in recipe_registry.list()
            if recipe.component_ids
            and set(recipe.component_ids).issubset(runtime_component_ids)
        ]
        contract_categories = {
            contract.component_id: contract.category for contract in contracts
        }
        executable_portfolio = {
            "schema_version": "executable_portfolio.v1",
            "research_snapshot_hash": snapshot_hash,
            "catalog_papers": len(paper_registry.list()),
            "method_profiles": len(coverage_profiles)
            if isinstance(coverage_profiles, list)
            else 0,
            "recipe_definitions": len(recipe_registry.list()),
            "runtime_eligible_components": len(runtime_component_ids),
            "runtime_eligible_recipes": len(runtime_recipes),
            "diagnosis_matched_recipes": len(planned_recipes),
            "planner_selected_recipes": len(plan.selected_recipes),
            "critic_accepted_recipes": sum(
                1 for report in recipe_critic_reports if report.get("accepted") is True
            ),
            "executable_policies": len(executable_pilot_policies),
            "paper_bound_executable_policies": sum(
                1
                for policy in executable_pilot_policies
                if method_profile_bindings.get(str(policy.action_id or ""))
            ),
            "candidate_families": sorted(
                {
                    contract_categories.get(component_id, component_id.split(".", 1)[0])
                    for policy in executable_pilot_policies
                    for component_id in policy.components
                }
            ),
            "recipe_sources": [path.resolve().as_posix() for path in recipe_sources],
            "recipe_load_errors": list(recipe_registry.load_errors),
            "authority": (
                "frozen_runtime_identity_and_effective_maturity; "
                "paper counts and metadata do not authorize training"
            ),
        }
        payload = {
            "schema_version": "paper_recipe_plan.v1",
            "research_snapshot_hash": snapshot_hash,
            "research_snapshot_verified": bool(child.context.metadata.get("research_snapshot_verified", False)),
            "research_snapshot_path": child.context.metadata.get("research_snapshot_path"),
            "llm_status": "deferred_to_unified_decision_bundle",
            "llm_proposal": None,
            "rule_plan": plan.model_dump(mode="json"),
            "recipe_critic_reports": recipe_critic_reports,
            "paper_method_coverage_path": (
                method_coverage_path.as_posix()
                if method_coverage_path.is_file()
                else None
            ),
            "paper_method_profile_bindings": method_profile_bindings,
            "paper_component_decisions": paper_component_decisions,
            "paper_training_blocked": paper_training_blocked,
            "paper_training_stop_reason": child.context.metadata.get(
                "paper_training_stop_reason"
            ),
            "executable_pilot_policies": [
                policy.model_dump(mode="json") for policy in executable_pilot_policies
            ],
            "executable_portfolio": executable_portfolio,
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
        write_yaml(portfolio_path, executable_portfolio)
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
            "executable_portfolio": portfolio_path,
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


def _write_paper_candidate_coverage(
    *,
    child: LoopOrchestrator,
    plan: Any,
    recipe_registry: RecipeRegistry,
    method_profile_bindings: dict[str, list[str]],
    critic_reports: list[dict[str, Any]],
) -> None:
    """Persist every planner-visible recipe before downstream filtering."""
    objective = load_optimization_objective(
        child.context.metadata.get("optimization_objective_path")
    )
    protocol_hash = objective.baseline_protocol_hash if objective is not None else "unknown"
    ledger = PaperCandidateCoverageLedger(
        child.context.artifact_path("paper_candidate_coverage.yaml"),
        run_id=child.context.run_id,
        protocol_hash=protocol_hash,
    )
    reports = {str(item.get("recipe_id")): item for item in critic_reports}
    records = []
    planned_by_identity = {
        (planned.recipe_id, planned.version): planned
        for planned in getattr(plan, "candidate_inventory", [])
    }
    planned_by_identity.update({
        (planned.recipe_id, planned.version): planned
        for planned in [
            *plan.selected_recipes,
            *plan.deferred_recipes,
            *plan.rejected_recipes,
        ]
    })
    planned_items = list(planned_by_identity.values())
    for budget_rank, planned in enumerate(planned_items, start=1):
        recipe = recipe_registry.get(planned.recipe_id, planned.version)
        if recipe is None:
            unresolved_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "recipe_id": planned.recipe_id,
                        "version": planned.version,
                        "protocol_hash": protocol_hash,
                        "status": "unresolved_recipe",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            records.append(
                planned_recipe_disposition(
                    run_id=child.context.run_id,
                    round_index=int(
                        child.context.metadata.get("auto_round_index") or 0
                    ),
                    recipe_id=planned.recipe_id,
                    recipe_version=planned.version,
                    component_ids=[f"unresolved.recipe:{planned.recipe_id}"],
                    decision="implementation_proposal",
                    reasons=["recipe_registry_entry_missing"],
                    related_papers=planned.related_papers,
                    method_profile_ids=planned.related_method_profile_ids,
                    required_adapters=[f"recipe_contract:{planned.recipe_id}"],
                    execution_fingerprint=unresolved_fingerprint,
                    budget_rank=budget_rank,
                )
            )
            continue
        critic = reports.get(planned.recipe_id, {})
        reasons = list(planned.reasons)
        if critic.get("accepted") is False:
            reasons.extend(
                str(item.get("code"))
                for item in critic.get("findings", [])
                if isinstance(item, dict) and item.get("code")
            )
        effective_decision = planned.decision
        if critic.get("accepted") is False and planned.decision in {"selected", "deferred"}:
            effective_decision = "rejected"
        arms = (
            _coupled_recipe_arms(recipe)
            if isinstance(recipe, CoupledRecipe)
            else [
                {
                    "combination_id": None,
                    "component_ids": list(recipe.component_ids),
                    "changed_variables": {},
                }
            ]
        )
        for arm in arms:
            combination_id = (
                str(arm["combination_id"])
                if arm.get("combination_id") is not None
                else None
            )
            component_ids = list(arm["component_ids"])
            fingerprint = _paper_recipe_execution_fingerprint(
                recipe,
                protocol_hash=protocol_hash,
                dataset_signature=(
                    child.context.dataset_manifest_sha256
                    or child.context.dataset_version
                    or "unknown"
                ),
                combination_id=combination_id,
                component_ids=component_ids,
                arm_overrides=dict(arm["changed_variables"]),
            )
            common = {
                "run_id": child.context.run_id,
                "round_index": int(
                    child.context.metadata.get("auto_round_index") or 0
                ),
                "recipe_id": recipe.recipe_id,
                "recipe_version": recipe.version,
                "component_ids": component_ids,
                "related_papers": sorted(
                    set(planned.related_papers)
                    | set(method_profile_bindings.get(recipe.recipe_id, []))
                ),
                "method_profile_ids": planned.related_method_profile_ids,
                "required_evidence": (
                    [item for item in reasons if item.startswith("missing_")]
                    if planned.decision == "needs_evidence"
                    else []
                ),
                "required_adapters": planned.required_adapters,
                "execution_fingerprint": fingerprint,
                "candidate_id": _paper_candidate_id(recipe, combination_id),
                "combination_id": combination_id,
                "budget_rank": budget_rank,
            }
            records.append(
                planned_recipe_disposition(
                    **common,
                    decision=planned.decision,
                    reasons=list(planned.reasons),
                )
            )
            if critic:
                records.append(
                    planned_recipe_disposition(
                        **common,
                        decision=effective_decision,
                        reasons=list(dict.fromkeys(reasons)),
                        source_stage="recipe_critic",
                    )
                )
    if records:
        coverage = ledger.upsert_many(records)
        ledger.reconcile(
            record.execution_fingerprint
            for record in records
            if record.execution_fingerprint is not None
        )
        child.context.metadata["paper_candidate_coverage_path"] = coverage_path = ledger.path.as_posix()
        child.context.metadata["paper_candidate_disposition_counts"] = coverage.disposition_counts
        child.evidence_store.log_artifact_manifest(
            run_id=child.context.run_id,
            name="paper_candidate_coverage",
            artifact_path=coverage_path,
            producer_stage="paper_recipe_planner",
        )


def _paper_recipe_execution_fingerprint(
    recipe: RecipeSpec,
    *,
    protocol_hash: str,
    dataset_signature: str,
    combination_id: str | None = None,
    component_ids: list[str] | None = None,
    arm_overrides: dict[str, Any] | None = None,
) -> str:
    payload = {
        "recipe_id": recipe.recipe_id,
        "version": recipe.version,
        "combination_id": combination_id,
        "component_ids": sorted(component_ids or recipe.component_ids),
        "train_overrides": {**recipe.train_overrides, **(arm_overrides or {})},
        "fixed_variables": recipe.fixed_variables,
        "coupled_variables": recipe.coupled_variables,
        "protocol_hash": protocol_hash,
        "dataset_signature": dataset_signature,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _paper_candidate_id(
    recipe: RecipeSpec,
    combination_id: str | None = None,
) -> str:
    base = f"paper_recipe_{recipe.recipe_id}_{recipe.version.replace('.', '_')}"
    if not combination_id:
        return base
    suffix = re.sub(r"[^a-z0-9]+", "_", combination_id.lower()).strip("_")
    return f"{base}__{suffix or 'combination'}"


def _coupled_recipe_arms(recipe: CoupledRecipe) -> list[dict[str, Any]]:
    """Return every declared non-baseline arm in stable declaration order."""
    arms: list[dict[str, Any]] = []
    for index, raw in enumerate(recipe.internal_ablation_plan, start=1):
        components = raw.get("components") if isinstance(raw, dict) else None
        if not isinstance(components, list) or not components:
            continue
        component_ids = [str(item) for item in components]
        if not set(component_ids).issubset(set(recipe.component_ids)):
            continue
        name = str(raw.get("name") or raw.get("variant") or f"arm_{index}")
        changed = raw.get("changed_variables", {})
        arms.append(
            {
                "combination_id": name,
                "component_ids": component_ids,
                "changed_variables": dict(changed) if isinstance(changed, dict) else {},
            }
        )
    return arms


def _candidate_policies_from_recipe(
    child: LoopOrchestrator,
    recipe: RecipeSpec,
    error_facts: list[ErrorFact],
    utility: float,
    *,
    paper_ids: list[str] | None = None,
) -> list[CandidatePolicy]:
    if not isinstance(recipe, CoupledRecipe):
        return [
            _candidate_policy_from_recipe(
                child,
                recipe,
                error_facts,
                utility,
                paper_ids=paper_ids,
            )
        ]
    return [
        _candidate_policy_from_recipe(
            child,
            recipe,
            error_facts,
            utility,
            paper_ids=paper_ids,
            combination_id=str(arm["combination_id"]),
            component_ids=list(arm["component_ids"]),
            arm_overrides=dict(arm["changed_variables"]),
        )
        for arm in _coupled_recipe_arms(recipe)
    ]


def _candidate_policy_from_recipe(
    child: LoopOrchestrator,
    recipe: RecipeSpec,
    error_facts: list[ErrorFact],
    utility: float,
    *,
    paper_ids: list[str] | None = None,
    combination_id: str | None = None,
    component_ids: list[str] | None = None,
    arm_overrides: dict[str, Any] | None = None,
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
    active_components = list(component_ids or recipe.component_ids)
    active_overrides = dict(arm_overrides or {})
    coupled_arm = isinstance(recipe, CoupledRecipe) and len(active_components) > 1
    action_domain = "model" if active_components else ("augmentation" if "augmentation" in recipe.primary_changed_variable else "data")
    return CandidatePolicy(
        policy_id=_paper_candidate_id(recipe, combination_id),
        source="rule_engine",
        action_domain=action_domain,
        action_id=recipe.recipe_id,
        execution_action="run_training",
        base_model=model,
        scale="n",
        framework="ultralytics",
        components=active_components,
        train_overrides={
            **recipe.train_overrides,
            **active_overrides,
            "imgsz": 640,
            "target_actions": [recipe.recipe_id],
            "recipe_version": recipe.version,
        },
        fixed_variables={**recipe.fixed_variables, "imgsz": 640},
        constraints=[
            PolicyConstraint(
                name="coupled_recipe",
                value=coupled_arm,
                hard=True,
            ),
            *([] if coupled_arm else [
                PolicyConstraint(name="single_variable", value=True, hard=True),
            ]),
            *(
                [
                    PolicyConstraint(
                        name="coupling_reason",
                        value=recipe.coupling_reason,
                        hard=True,
                    ),
                    PolicyConstraint(
                        name="internal_ablation_plan",
                        value=recipe.internal_ablation_plan,
                        hard=True,
                    ),
                    PolicyConstraint(
                        name="ablation_combination_id",
                        value=combination_id,
                        hard=True,
                    ),
                ]
                if isinstance(recipe, CoupledRecipe)
                else []
            ),
            PolicyConstraint(name="fixed_imgsz", value=640, hard=True),
        ],
        target_error_facts=target_facts,
        expected_improvement={
            "expected_gain": expected or {metric: 0.1 for metric in recipe.target_metrics},
            "paper_prior_only": True,
            "recipe_id": recipe.recipe_id,
            "recipe_version": recipe.version,
            "combination_id": combination_id,
            "paper_ids": sorted(set(paper_ids or recipe.coupling_source_papers)),
            "component_ids": active_components,
            "implementation_status": "smoke_passed",
        },
        priority_hint=max(8.0, min(float(utility), 10.0)),
        expected_effect=[f"{key}: {value}" for key, value in recipe.expected_effects.items()],
        risk=recipe.implementation_risk if recipe.implementation_risk != "unknown" else "medium",
        rationale="Critic-approved atomic paper recipe; evaluator and pilot gates remain authoritative.",
    )


def _apply_paper_method_profile_gate(
    plan: Any,
    *,
    recipe_registry: RecipeRegistry,
    coverage_path: Path,
    require_frozen_coverage: bool,
) -> tuple[Any, dict[str, list[str]]]:
    """Allow only snapshot-frozen profiles with a trainable implementation route."""
    def implementation_request(planned: Any, reason: str) -> Any:
        recipe = recipe_registry.get(planned.recipe_id, planned.version)
        component_ids = list(recipe.component_ids) if recipe is not None else []
        required_adapters = list(planned.required_adapters) or [
            f"adapter_for:{component_id}" for component_id in component_ids
        ]
        return planned.model_copy(update={
            "decision": "implementation_proposal",
            "reasons": [*planned.reasons, reason],
            "required_adapters": required_adapters,
        })

    if not coverage_path.is_file():
        if require_frozen_coverage:
            rejected = [
                implementation_request(item, "paper_method_coverage_missing")
                for item in [*plan.selected_recipes, *plan.deferred_recipes]
                if _requires_paper_method_profile(
                    recipe_registry.get(item.recipe_id, item.version)
                )
            ]
            retained = [
                item
                for item in [*plan.selected_recipes, *plan.deferred_recipes]
                if not _requires_paper_method_profile(
                    recipe_registry.get(item.recipe_id, item.version)
                )
            ]
            return (
                plan.model_copy(
                    update={
                        "selected_recipes": [
                            item for item in retained if item.decision == "selected"
                        ],
                        "deferred_recipes": [
                            item for item in retained if item.decision == "deferred"
                        ],
                        "rejected_recipes": [*plan.rejected_recipes, *rejected],
                    }
                ),
                {},
            )
        return plan, {}
    try:
        raw = read_yaml(coverage_path)
    except (OSError, TypeError, ValueError):
        raw = {}
    profiles = {
        str(item.get("profile_id")): item
        for item in raw.get("profiles", [])
        if isinstance(item, dict) and item.get("profile_id")
    }
    decisions = {
        str(item.get("profile_id")): item
        for item in raw.get("decisions", [])
        if isinstance(item, dict) and item.get("profile_id")
    }
    bindings: dict[str, list[str]] = {}
    profile_bindings: dict[str, list[str]] = {}
    for planned in [*plan.selected_recipes, *plan.deferred_recipes]:
        recipe = recipe_registry.get(planned.recipe_id, planned.version)
        if recipe is None:
            continue
        for profile_id, profile in profiles.items():
            decision = decisions.get(profile_id, {})
            canonical = set(profile.get("canonical_component_ids", []))
            decision_components = set(decision.get("canonical_component_ids", []))
            if (
                decision.get("decision") in {"reuse_existing_adapter", "coupled_recipe"}
                and set(recipe.component_ids).issubset(canonical)
                and set(recipe.component_ids).issubset(decision_components)
            ):
                bindings.setdefault(planned.recipe_id, []).append(
                    str(profile.get("paper_id"))
                )
                profile_bindings.setdefault(planned.recipe_id, []).append(profile_id)

    def with_provenance(planned: Any) -> Any:
        return planned.model_copy(
            update={
                "related_papers": sorted(
                    set(planned.related_papers)
                    | set(bindings.get(planned.recipe_id, []))
                ),
                "related_method_profile_ids": sorted(
                    set(planned.related_method_profile_ids)
                    | set(profile_bindings.get(planned.recipe_id, []))
                ),
            }
        )
    rejected: list[Any] = []
    deferred: list[Any] = []
    selected: list[Any] = []
    for planned in plan.selected_recipes:
        recipe = recipe_registry.get(planned.recipe_id, planned.version)
        if not _requires_paper_method_profile(recipe):
            selected.append(planned)
        elif bindings.get(planned.recipe_id):
            selected.append(with_provenance(planned))
        else:
            rejected.append(implementation_request(planned, "paper_method_profile_not_trainable"))
    for planned in plan.deferred_recipes:
        recipe = recipe_registry.get(planned.recipe_id, planned.version)
        if not _requires_paper_method_profile(recipe) or bindings.get(planned.recipe_id):
            deferred.append(
                with_provenance(planned)
                if bindings.get(planned.recipe_id)
                else planned
            )
        else:
            rejected.append(implementation_request(planned, "paper_method_profile_not_trainable"))
    return (
        plan.model_copy(
            update={
                "selected_recipes": selected,
                "deferred_recipes": deferred,
                "rejected_recipes": [*plan.rejected_recipes, *rejected],
            }
        ),
        {key: sorted(set(value)) for key, value in bindings.items()},
    )


def _requires_paper_method_profile(recipe: Any | None) -> bool:
    """Keep paper provenance gates scoped to recipes that claim paper identity."""
    if recipe is None:
        return True
    return recipe.recipe_id.startswith(("paper.", "paper-", "paper_"))


def _paper_component_decision_rows(
    *,
    plan: Any,
    recipe_registry: RecipeRegistry,
    coverage_path: Path,
    effective_maturity: dict[str, EffectiveComponentMaturity],
    method_profile_bindings: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Build concise paper/component identity rows for terminal and ledger output."""
    target_components: set[str] = set()
    papers_by_component: dict[str, set[str]] = {}
    planned = [
        *plan.selected_recipes,
        *plan.deferred_recipes,
        *plan.rejected_recipes,
    ]
    for item in planned:
        recipe = recipe_registry.get(item.recipe_id, item.version)
        if recipe is None:
            continue
        target_components.update(recipe.component_ids)
        for component_id in recipe.component_ids:
            papers_by_component.setdefault(component_id, set()).update(
                method_profile_bindings.get(item.recipe_id, [])
            )
    if coverage_path.is_file():
        try:
            coverage = read_yaml(coverage_path)
        except (OSError, TypeError, ValueError):
            coverage = {}
        for profile in coverage.get("profiles", []):
            if not isinstance(profile, dict):
                continue
            paper_id = str(profile.get("paper_id") or "")
            for component_id in profile.get("canonical_component_ids", []):
                if component_id in target_components and paper_id:
                    papers_by_component.setdefault(component_id, set()).add(paper_id)
    rows: list[dict[str, Any]] = []
    for component_id in sorted(target_components):
        effective = effective_maturity.get(component_id)
        reasons = list(effective.rejection_reasons) if effective is not None else [
            "effective_maturity_contract_missing"
        ]
        paper_ids = sorted(papers_by_component.get(component_id, set()))
        if not paper_ids:
            reasons.append("paper_method_profile_not_trainable")
        eligible = bool(
            effective is not None
            and effective.valid_for_training
            and paper_ids
        )
        rows.append({
            "paper_ids": paper_ids,
            "component_id": component_id,
            "adapter_hash": effective.adapter_hash if effective is not None else None,
            "maturity": (
                effective.effective_maturity if effective is not None else "missing"
            ),
            "maturity_evidence_source": (
                effective.evidence_source if effective is not None else "none"
            ),
            "eligible": eligible,
            "rejection_reasons": [] if eligible else list(dict.fromkeys(reasons)),
        })
    return rows


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
    """Load the run-bound contracts without consulting live maturity state."""
    paths: list[Path] = []
    snapshot_path = child.context.metadata.get("research_snapshot_path")
    snapshot_verified = bool(
        child.context.metadata.get("research_snapshot_verified", False)
    )
    if snapshot_verified and isinstance(snapshot_path, str) and snapshot_path:
        paths.append(Path(snapshot_path) / "component_contracts.yaml")
    else:
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
    effective, resolutions = _resolve_effective_component_contracts(
        child,
        list(contracts.values()),
    )
    return [
        contract
        for contract in effective
        if resolutions[contract.component_id].valid_for_training
    ]


def _resolve_effective_component_contracts(
    child: LoopOrchestrator,
    contracts: list[ComponentContract],
) -> tuple[list[ComponentContract], dict[str, EffectiveComponentMaturity]]:
    """Resolve only run-bound maturity; live overlays are build-time inputs."""
    frozen_identities = _load_frozen_maturity_identities(child)
    resolved = EffectiveMaturityResolver(
        frozen_identities=frozen_identities,
    ).resolve(
        {item.component_id: item for item in contracts}
    )
    child.context.metadata["effective_component_maturity"] = {
        component_id: {
            "source_maturity": item.source_maturity,
            "effective_maturity": item.effective_maturity,
            "adapter_hash": item.adapter_hash,
            "frozen_protocol_hash": (
                frozen_identities[item.component_id].protocol_hash
                if item.component_id in frozen_identities
                else None
            ),
            "evidence_source": item.evidence_source,
            "overlay_status": item.overlay_status,
            "valid_for_training": item.valid_for_training,
            "rejection_reasons": item.rejection_reasons,
        }
        for component_id, item in resolved.items()
    }
    return (
        [resolved[key].contract for key in sorted(resolved)],
        resolved,
    )


def _load_frozen_maturity_identities(
    child: LoopOrchestrator,
) -> dict[str, FrozenComponentMaturity]:
    if not child.context.metadata.get("research_snapshot_verified", False):
        return {}
    snapshot_path = child.context.metadata.get("research_snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path:
        return {}
    manifest_path = Path(snapshot_path) / "effective_component_maturity.yaml"
    if not manifest_path.is_file():
        return {}
    return EffectiveComponentMaturityManifest.from_yaml(manifest_path).by_component()


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


def _empty_recipe_round_reason(path: Path) -> str | None:
    """Explain when method recipes are exhausted and scalar HPO is intentionally disabled."""
    if not path.is_file():
        return None
    raw = read_yaml(path)
    plan = raw.get("training_recipe_plan", {})
    if not isinstance(plan, dict) or plan.get("policies"):
        return None
    decisions = plan.get("family_decisions", [])
    if not isinstance(decisions, list):
        return None
    scalar_disabled = any(
        isinstance(item, dict) and "Scalar HPO is disabled" in str(item.get("reason") or "")
        for item in decisions
    )
    method_terminal = any(
        isinstance(item, dict)
        and item.get("decision") in {"exhausted", "rejected_by_evidence"}
        and "Scalar HPO" not in str(item.get("reason") or "")
        for item in decisions
    )
    return "method_candidates_exhausted" if scalar_disabled and method_terminal else None


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
        if round_result.training_loop is None or not round_result.training_loop.completed:
            continue
        for assessment in round_result.candidate_assessments:
            if assessment.execution_class != "executable":
                continue
            stage = str(assessment.action_id or "direct")
            candidate_id = str(assessment.candidate_id or assessment.policy_id)
            key = f"{candidate_id}:{stage}"
            item = counts.setdefault(
                key,
                {
                    "candidate_id": assessment.candidate_id,
                    "action_id": assessment.action_id,
                    "stage": stage,
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
        "- Full run started: false",
        "",
        "## Rounds",
        "",
    ]
    if result.readiness is not None:
        readiness_lines = [
            f"- Exploration readiness: `{result.readiness.mode}`",
            f"- Readiness blockers: `{result.readiness.blockers}`",
        ]
        if result.certification_attempted:
            readiness_lines.extend(
                [
                    f"- Automatic GPU certification: `{result.certification_status}`",
                    f"- Certification report: `{result.certification_report_path}`",
                ]
            )
        lines[7:7] = [*readiness_lines, ""]
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
        lines.append(
            "- Paper recipes admitted to the executable path: "
            + ", ".join(f"`{item}`" for item in paper_summary["adopted"])
            + "."
        )
    else:
        lines.append("- Paper recipes admitted to the executable path: none.")
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
                f"- `{item.get('candidate_id')}` executed {item.get('stage')} "
                f"{item.get('count')} times in rounds {item.get('rounds')}; "
                "the same candidate and ASHA rung should normally execute once."
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
                executable_assessments = [
                    item for item in round_result.candidate_assessments
                    if item.execution_class == "executable"
                ]
                adopted.extend(
                    str(item.get("action_id") or item.get("policy_id"))
                    for item in policies
                    if isinstance(item, dict)
                    and any(
                        assessment.policy_id == str(item.get("policy_id") or "")
                        or str(assessment.action_id or "") == str(item.get("action_id") or "")
                        or str(assessment.candidate_id or "") == str(item.get("action_id") or "")
                        for assessment in executable_assessments
                    )
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
