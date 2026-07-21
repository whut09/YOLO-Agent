"""Canonical paper-candidate bridge into ASHA and RoundExecutionPlan."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.agents.asha_scheduler import (
    ASHAAssignment,
    ASHAObservation,
    ASHAScheduler,
    ASHAStudyStore,
)
from yolo_agent.agents.decision_bundle import DecisionContext
from yolo_agent.agents.paper_component_gate import PaperComponentGateResult
from yolo_agent.agents.recipe_critic import RecipeCriticReport
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.decision_ledger import DecisionLedger, DecisionLedgerRecord
from yolo_agent.core.execution_queue import ExecutionQueue
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.paired_experiment import PairedExperimentResult
from yolo_agent.core.policy_memory import (
    ActionFingerprint,
    PolicyMemoryRecord,
    PolicyMemoryStore,
)
from yolo_agent.core.round_execution_plan import (
    RoundAssignment,
    RoundExecutionPlan,
    RoundStageSpec,
    build_asha_assignment_plan,
)
from yolo_agent.agents.loop_io import write_yaml
from yolo_agent.recipes.paper_priors import RecipePrior
from yolo_agent.recipes.schemas import RecipeSpec
from yolo_agent.research.reproduction_state import ReproductionState
from yolo_agent.research.snapshot import ResearchSnapshot


CandidateBucket = Literal["exploration", "exploitation"]
OrchestratorAction = Literal[
    "queue_assignment",
    "evidence_recovery",
    "awaiting_pilot_3_cohort",
    "full_candidate_recommendation",
    "idle",
]


class PaperCandidateOrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_pilot_3_cohort: int = Field(default=3, ge=3)
    max_registered_candidates: int = Field(default=6, ge=3)
    exploitation_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    family_cooldown_rounds: int = Field(default=2, ge=0)


class PaperCandidateSubmission(BaseModel):
    """One fully guarded materialized paper candidate, never a raw prior."""

    model_config = ConfigDict(extra="forbid")
    decision_context: DecisionContext
    research_snapshot: ResearchSnapshot
    recipe_prior: RecipePrior
    recipe: RecipeSpec
    eligibility: PaperComponentGateResult
    critic: RecipeCriticReport
    source_node: ExperimentNode
    matched_control_node: ExperimentNode | None = None
    component_family: str
    bucket: CandidateBucket
    round_index: int = Field(ge=1)


class PaperCandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_id: str
    candidate_id: str
    prior_id: str
    recipe_id: str
    recipe_version: str
    component_ids: list[str]
    paper_ids: list[str] = Field(default_factory=list)
    changed_variables: list[str]
    component_family: str
    bucket: CandidateBucket
    round_index: int
    decision_context_hash: str
    research_snapshot_hash: str
    eligibility_token: str
    critic: dict[str, Any]


class PaperCandidateRegistrationReport(BaseModel):
    registered: list[str] = Field(default_factory=list)
    deferred: dict[str, str] = Field(default_factory=dict)
    rejected: dict[str, str] = Field(default_factory=dict)
    cohort_size: int = 0
    minimum_cohort: int = 3


class PaperCandidateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assignment_id: str
    post_eval_complete: bool = False
    error_facts_complete: bool = False
    paired_result: PairedExperimentResult | None = None
    target_error_improved_count: int = Field(default=0, ge=0)
    diagnosis_gate_passed: bool | None = None
    missing_evidence: list[str] = Field(default_factory=list)
    gpu_hours: float | None = Field(default=None, ge=0.0)


class PaperCandidateStep(BaseModel):
    action: OrchestratorAction
    assignment: ASHAAssignment | None = None
    round_plan: RoundExecutionPlan | None = None
    queue: ExecutionQueue | None = None
    missing_evidence: list[str] = Field(default_factory=list)
    recommended_candidate_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class PaperCandidateResultUpdate(BaseModel):
    assignment_id: str
    trial_id: str
    trial_status: str
    evidence_complete: bool
    missing_evidence: list[str] = Field(default_factory=list)
    policy_memory_updated: bool = False
    reproduction_updated: bool = False


class PaperCandidateState(BaseModel):
    schema_version: str = "paper_candidate_orchestrator.v1"
    base_run_id: str
    candidates: dict[str, PaperCandidateRecord] = Field(default_factory=dict)
    pending_recovery: dict[str, list[str]] = Field(default_factory=dict)
    family_last_round: dict[str, int] = Field(default_factory=dict)
    completed_prior_ids: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaperCandidateOrchestrator:
    """Register guarded paper recipes and let ASHA alone allocate training."""

    policy_version = "paper_candidate_orchestrator.v1"

    def __init__(
        self,
        run_dir: Path | str,
        *,
        base_run_id: str,
        config: PaperCandidateOrchestratorConfig | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.artifact_dir = self.run_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or PaperCandidateOrchestratorConfig()
        self.state_path = self.artifact_dir / "paper_candidate_state.yaml"
        self.report_path = self.artifact_dir / "paper_recipe_report.yaml"
        self.reproduction_path = self.artifact_dir / "paper_component_reproduction.yaml"
        self.round_plan_path = self.artifact_dir / "round_execution_plan.yaml"
        self.queue_path = self.run_dir / "execution_queue.yaml"
        self.asha_store = ASHAStudyStore(self.artifact_dir / "asha_state.yaml")
        self.scheduler = self.asha_store.load_or_create(base_run_id)
        for rung in self.scheduler.study.rungs:
            if rung.stage_id == "pilot_3":
                rung.minimum_completed = max(
                    rung.minimum_completed,
                    self.config.min_pilot_3_cohort,
                )
        self.ledger = DecisionLedger(self.artifact_dir / "decision_ledger.jsonl")
        self.policy_memory = PolicyMemoryStore(self.run_dir.parent)
        self.state = self._load_state(base_run_id)

    def register_cohort(
        self,
        submissions: list[PaperCandidateSubmission],
    ) -> PaperCandidateRegistrationReport:
        """Register only materialized, gate-approved, critic-approved candidates."""
        report = PaperCandidateRegistrationReport(minimum_cohort=self.config.min_pilot_3_cohort)
        eligible: list[PaperCandidateSubmission] = []
        for submission in submissions:
            candidate_id = submission.source_node.candidate_config.candidate_id
            reason = self._submission_rejection(submission)
            if reason:
                report.rejected[candidate_id] = reason
                self._ledger_registration(submission, "rejected", reason)
                continue
            cooldown = self.state.family_last_round.get(submission.component_family)
            if cooldown is not None and submission.round_index - cooldown <= self.config.family_cooldown_rounds:
                reason = f"component_family_cooldown:{submission.component_family}:last_round={cooldown}"
                report.deferred[candidate_id] = reason
                self._ledger_registration(submission, "deferred", reason)
                continue
            if submission.recipe_prior.prior_id in self.state.completed_prior_ids:
                reason = "duplicate_completed_recipe_prior"
                report.deferred[candidate_id] = reason
                self._ledger_registration(submission, "deferred", reason)
                continue
            eligible.append(submission)

        selected = self._select_exploit_explore(eligible)
        selected_ids = {item.source_node.candidate_config.candidate_id for item in selected}
        for submission in eligible:
            candidate_id = submission.source_node.candidate_config.candidate_id
            if candidate_id not in selected_ids:
                report.deferred[candidate_id] = "deferred_by_exploit_explore_budget"
                self._ledger_registration(submission, "deferred", report.deferred[candidate_id])

        for submission in selected:
            candidate_id = submission.source_node.candidate_config.candidate_id
            trial_id = f"{self.scheduler.study.base_run_id}:paper:{candidate_id}"
            source_node = _prepared_source_node(submission)
            trial = self.scheduler.register_trial(
                trial_id=trial_id,
                candidate_id=candidate_id,
                source_run_id=submission.decision_context.run_id,
                source_node=source_node,
                baseline_control_node=submission.matched_control_node,
                target_error_facts=[dict(item) for item in submission.recipe.target_error_facts],
            )
            self.state.candidates[trial.trial_id] = PaperCandidateRecord(
                trial_id=trial.trial_id,
                candidate_id=candidate_id,
                prior_id=submission.recipe_prior.prior_id,
                recipe_id=submission.recipe.recipe_id,
                recipe_version=submission.recipe.version,
                component_ids=list(submission.recipe.component_ids),
                paper_ids=list(submission.recipe_prior.paper_ids),
                changed_variables=list(submission.eligibility.changed_variables),
                component_family=submission.component_family,
                bucket=submission.bucket,
                round_index=submission.round_index,
                decision_context_hash=submission.decision_context.context_hash,
                research_snapshot_hash=submission.research_snapshot.snapshot_hash,
                eligibility_token=str(submission.eligibility.eligibility_token),
                critic=submission.critic.model_dump(mode="json"),
            )
            report.registered.append(candidate_id)
            self._ledger_registration(submission, "registered", "registered_with_asha_without_direct_budget")
        report.cohort_size = len([
            trial for trial in self.scheduler.study.trials
            if trial.observation("pilot_3") is None and trial.status in {"waiting", "running", "needs_evidence"}
        ])
        self._save()
        return report

    def _submission_rejection(self, submission: PaperCandidateSubmission) -> str:
        if submission.recipe_prior.research_snapshot_hash != submission.research_snapshot.snapshot_hash:
            return "recipe_prior_snapshot_mismatch"
        if submission.decision_context.research_snapshot_hash != submission.research_snapshot.snapshot_hash:
            return "decision_context_snapshot_mismatch"
        if submission.recipe_prior.component_ids != submission.recipe.component_ids:
            return "materialized_recipe_component_mismatch"
        if not submission.eligibility.eligible or not submission.eligibility.eligibility_token:
            return "paper_component_eligibility_gate_rejected"
        if submission.eligibility.execution_class not in {"pilot_candidate", "full_candidate"}:
            return "eligibility_class_not_trainable"
        if not submission.critic.accepted:
            return "recipe_critic_rejected"
        if submission.matched_control_node is None:
            return "matched_control_required"
        if not submission.recipe.target_error_facts:
            return "target_error_fact_required"
        if not _node_has_fixed_imgsz(submission.source_node):
            return "fixed_imgsz_640_required"
        return ""

    def _select_exploit_explore(
        self,
        candidates: list[PaperCandidateSubmission],
    ) -> list[PaperCandidateSubmission]:
        capacity = self.config.max_registered_candidates
        exploit_limit = round(capacity * self.config.exploitation_ratio)
        explore_limit = capacity - exploit_limit
        exploit = sorted(
            (item for item in candidates if item.bucket == "exploitation"),
            key=lambda item: (-item.recipe_prior.confidence, item.recipe_prior.prior_id),
        )
        explore = sorted(
            (item for item in candidates if item.bucket == "exploration"),
            key=lambda item: (-item.recipe_prior.confidence, item.recipe_prior.prior_id),
        )
        selected = [*exploit[:exploit_limit], *explore[:explore_limit]]
        remaining = [item for item in [*exploit[exploit_limit:], *explore[explore_limit:]] if item not in selected]
        selected.extend(remaining[: max(0, capacity - len(selected))])
        return selected

    def next_step(self, *, confirm_full_run: bool = False) -> PaperCandidateStep:
        """Return the next queue projection; policy YAML has no queue authority."""
        if self.state.pending_recovery:
            assignment_id = sorted(self.state.pending_recovery)[0]
            assignment = _assignment_by_id(self.scheduler, assignment_id)
            if assignment is not None:
                return self._evidence_recovery_step(
                    assignment,
                    self.state.pending_recovery[assignment_id],
                )
        assignment = self.scheduler.next_assignment(confirm_full_run=confirm_full_run)
        if assignment is None:
            recommendations = sorted(
                trial.candidate_id
                for trial in self.scheduler.study.trials
                if trial.status == "full_pending_confirmation"
            )
            if recommendations:
                self._write_report()
                return PaperCandidateStep(
                    action="full_candidate_recommendation",
                    recommended_candidate_ids=recommendations,
                    reason="pilot_10 survivors require explicit full confirmation",
                )
            pilot_observations = sum(
                trial.observation("pilot_3") is not None for trial in self.scheduler.study.trials
            )
            if self.scheduler.study.trials and pilot_observations < self.config.min_pilot_3_cohort:
                return PaperCandidateStep(
                    action="awaiting_pilot_3_cohort",
                    reason=(
                        f"pilot_3 cohort incomplete: {pilot_observations}/"
                        f"{self.config.min_pilot_3_cohort} completed"
                    ),
                )
            return PaperCandidateStep(action="idle", reason="ASHA has no eligible assignment")

        trial = self.scheduler.study.trial(assignment.trial_id)
        if assignment.stage_id.startswith("pilot") and trial.baseline_control_node is None:
            return PaperCandidateStep(
                action="idle",
                assignment=assignment,
                reason="matched baseline control is required before pilot queue materialization",
            )
        record = self.state.candidates[trial.trial_id]
        run_id = f"{self.scheduler.study.base_run_id}-{assignment.stage_id}-{assignment.candidate_id}"
        plan = build_asha_assignment_plan(
            run_id=run_id,
            source_node=trial.source_node,
            stage_id=assignment.stage_id,
            epochs=assignment.epochs,
            fraction=assignment.fraction,
            seed=int(assignment.seed),
            seed_index=assignment.seed_index,
            run_name=f"{assignment.candidate_id}_{assignment.stage_id}_seed{assignment.seed_index}",
            baseline_control_node=trial.baseline_control_node,
            assignment_id=assignment.assignment_id,
        )
        payload = plan.model_dump(mode="json")
        payload.update({
            "objective_hash": _objective_hash_from_node(trial.source_node),
            "run_protocol_hash": _protocol_hash_from_node(trial.source_node),
            "decision_context_hash": record.decision_context_hash,
            "selected_recipes": [{
                "prior_id": record.prior_id,
                "recipe_id": record.recipe_id,
                "version": record.recipe_version,
                "component_ids": record.component_ids,
                "eligibility_token": record.eligibility_token,
            }],
            "critic_results": [record.critic],
        })
        plan = RoundExecutionPlan.model_validate(payload)
        for node in plan.execution_nodes:
            if not _node_has_fixed_imgsz(node):
                raise ValueError(f"ASHA emitted non-640 node: {node.node_id}")
        queue = ExecutionQueue.from_round_execution_plan(plan.run_id, plan)
        if any(item.command.command_type == "train" for item in queue.items):
            candidate_items = [item for item in queue.items if not item.command.metadata.get("matched_baseline_control")]
            if len(candidate_items) != 1:
                raise ValueError("ASHA assignment must materialize exactly one candidate training item")
        self.scheduler.mark_running(assignment, run_id=run_id, node_id=plan.execution_nodes[0].node_id)
        plan.to_yaml(self.round_plan_path)
        queue.to_yaml(self.queue_path)
        self._save()
        self._ledger_step(assignment, plan, "queue_assignment", [])
        return PaperCandidateStep(
            action="queue_assignment",
            assignment=assignment,
            round_plan=plan,
            queue=queue,
            reason="ASHA assignment projected through RoundExecutionPlan",
        )

    def _evidence_recovery_step(
        self,
        assignment: ASHAAssignment,
        missing: list[str],
    ) -> PaperCandidateStep:
        trial = self.scheduler.study.trial(assignment.trial_id)
        source = trial.source_node
        recovery_spec = CommandSpec(
            command_type="custom",
            argv=["paper-evidence-recovery", assignment.assignment_id],
            metadata={
                "evidence_recovery_only": True,
                "asha_assignment_id": assignment.assignment_id,
                "post_eval_required": True,
            },
        )
        recovery_node = source.model_copy(update={
            "node_id": f"{source.node_id}__{assignment.stage_id}_evidence_recovery",
            "command": recovery_spec.display(),
            "command_spec": recovery_spec,
            "changed_variables": {},
        })
        round_assignment = RoundAssignment(
            stage_id=assignment.stage_id,
            candidate_id=assignment.candidate_id,
            source_node_id=source.node_id,
            execution_node_id=recovery_node.node_id,
            rank=1,
            status="active",
            reason="evidence_recovery_only",
        )
        plan = RoundExecutionPlan(
            run_id=f"{self.scheduler.study.base_run_id}-evidence-recovery",
            round_id=f"{assignment.assignment_id}-evidence-recovery",
            stages=[RoundStageSpec(
                stage_id=assignment.stage_id,
                training_profile="evidence_recovery",
                epochs=assignment.epochs,
                fraction=assignment.fraction,
                keep_top_k=1,
            )],
            assignments=[round_assignment],
            execution_nodes=[recovery_node],
            deferred_nodes=[source],
            evidence_requirements={recovery_node.node_id: missing},
            active_stage=assignment.stage_id,
            scheduler_mode="external_asha",
            asha_assignment_id=assignment.assignment_id,
        )
        queue = ExecutionQueue.from_round_execution_plan(plan.run_id, plan)
        queue.metadata["evidence_recovery_only"] = True
        plan.to_yaml(self.round_plan_path)
        queue.to_yaml(self.queue_path)
        self._ledger_step(assignment, plan, "evidence_recovery", missing)
        return PaperCandidateStep(
            action="evidence_recovery",
            assignment=assignment,
            round_plan=plan,
            queue=queue,
            missing_evidence=missing,
            reason="post-eval or paired evidence is incomplete; no training item was queued",
        )

    def record_result(self, evidence: PaperCandidateEvidence) -> PaperCandidateResultUpdate:
        """Import post-eval evidence and report exactly one ASHA observation."""
        assignment = _assignment_by_id(self.scheduler, evidence.assignment_id)
        if assignment is None:
            raise KeyError(f"Unknown ASHA assignment: {evidence.assignment_id}")
        trial = self.scheduler.study.trial(assignment.trial_id)
        missing = list(evidence.missing_evidence)
        if not evidence.post_eval_complete:
            missing.append("matched_coco_post_eval")
        if not evidence.error_facts_complete:
            missing.append("complete_candidate_error_facts")
        paired = evidence.paired_result
        if paired is None:
            missing.append("verified_paired_experiment_result")
        else:
            if not paired.verified:
                missing.append("verified_paired_experiment_result")
            if paired.protocol_match_status != "matched" or not paired.matched_control.matched:
                missing.append("matched_baseline_control")
            if paired.candidate_id != assignment.candidate_id:
                missing.append("paired_result_candidate_mismatch")
        missing = sorted(set(missing))
        primary_delta = paired.metric_deltas.get("map50_95") if paired is not None else None
        complete = not missing and primary_delta is not None
        if primary_delta is None:
            missing = sorted(set([*missing, "paired_map50_95_delta"]))
            complete = False
        observation = ASHAObservation(
            stage_id=assignment.stage_id,
            node_id=(paired.candidate_node_id if paired is not None else assignment.assigned_node_id or trial.source_node.node_id),
            seed_index=assignment.seed_index,
            seed=assignment.seed,
            paired_delta=(primary_delta.effect_delta if primary_delta is not None else None),
            paired_result_verified=bool(paired and paired.verified),
            paired_result_hash=(paired.result_hash if paired is not None else None),
            protocol_match_status=(paired.protocol_match_status if paired is not None else "incomplete"),
            paired_experiment_result=paired,
            target_error_improved_count=evidence.target_error_improved_count,
            latency_regression=(paired.latency_delta.effect_delta if paired and paired.latency_delta else None),
            model_size_regression=(paired.model_size_delta.effect_delta if paired and paired.model_size_delta else None),
            diagnosis_gate_passed=evidence.diagnosis_gate_passed,
            evidence_complete=complete,
            failure_reason="",
        )
        updated_trial = self.scheduler.report(assignment.trial_id, observation)
        memory_updated = False
        reproduction_updated = False
        if complete and paired is not None and primary_delta is not None:
            self.state.pending_recovery.pop(assignment.assignment_id, None)
            memory_updated = self._update_policy_memory(
                assignment,
                paired,
                primary_delta.effect_delta,
                evidence.gpu_hours,
            )
            reproduction_updated = self._update_reproduction(assignment, updated_trial.status, paired)
            self._record_history(assignment, primary_delta.effect_delta)
        else:
            self.state.pending_recovery[assignment.assignment_id] = missing
        self._ledger_result(assignment, updated_trial.status, complete, missing, paired)
        self._save()
        return PaperCandidateResultUpdate(
            assignment_id=assignment.assignment_id,
            trial_id=assignment.trial_id,
            trial_status=updated_trial.status,
            evidence_complete=complete,
            missing_evidence=missing,
            policy_memory_updated=memory_updated,
            reproduction_updated=reproduction_updated,
        )

    def _update_policy_memory(
        self,
        assignment: ASHAAssignment,
        paired: PairedExperimentResult,
        effect_delta: float,
        gpu_hours: float | None,
    ) -> bool:
        record = self.state.candidates[assignment.trial_id]
        metric = paired.metric_deltas["map50_95"]
        memory = PolicyMemoryRecord(
            run_id=paired.run_id,
            dataset_version="coco",
            action=record.recipe_id,
            action_fingerprint=ActionFingerprint(
                action=record.recipe_id,
                recipe_id=record.recipe_id,
                recipe_version=record.recipe_version,
                paper_ids=record.paper_ids,
                component_ids=record.component_ids,
                component_versions={item: "snapshot_bound" for item in record.component_ids},
                changed_variable=record.changed_variables[0] if record.changed_variables else "unknown",
                detector_family="yolo26",
                model_family="yolo26",
                dataset_signature="coco",
                protocol_hash=paired.matched_control.match_key.protocol_hash if paired.matched_control.match_key else "unknown",
                snapshot_hash=record.research_snapshot_hash,
                fidelity=assignment.stage_id if assignment.stage_id in {"pilot_3", "pilot_10"} else "full",
                seed=assignment.seed,
                matched_control_hash=metric.match_key_hash,
            ),
            target="map50_95",
            metric_name="map50_95",
            before=metric.baseline_value,
            after=metric.candidate_value,
            delta=metric.paired_delta,
            effect_delta=effect_delta,
            trend="improved" if effect_delta > 0 else ("regressed" if effect_delta < 0 else "unchanged"),
            candidate_id=assignment.candidate_id,
            node_id=paired.candidate_node_id,
            changed_variables={item: record.recipe_id for item in record.changed_variables},
            source="paper_candidate_orchestrator",
            matched_control_hash=metric.match_key_hash,
            cost={"gpu_hours": gpu_hours},
        )
        return bool(self.policy_memory.append([memory]))

    def _update_reproduction(
        self,
        assignment: ASHAAssignment,
        trial_status: str,
        paired: PairedExperimentResult,
    ) -> bool:
        record = self.state.candidates[assignment.trial_id]
        if assignment.stage_id == "pilot_3":
            status = "pilot_running"
        elif assignment.stage_id == "pilot_10":
            status = "pilot_reproduced" if trial_status == "full_pending_confirmation" else "pilot_rejected"
        elif assignment.stage_id == "candidate_full_seed_1":
            status = "full_pending_confirmation"
        else:
            status = "confirmed_multi_seed" if trial_status == "confirmed" else "full_pending_confirmation"
        existing_index = (
            _read_yaml_mapping(self.reproduction_path)
            if self.reproduction_path.exists()
            else {}
        )
        states: dict[str, dict[str, Any]] = dict(existing_index.get("components") or {})
        for component_id in record.component_ids:
            path = self.artifact_dir / f"reproduction_state_{_safe_name(component_id)}.yaml"
            if path.exists():
                state = ReproductionState.from_yaml(path)
            else:
                state = ReproductionState(
                    paper_id=record.prior_id,
                    component_id=component_id,
                    status="smoke_passed",
                )
            state.status = status
            state.local_delta[assignment.stage_id] = {
                "candidate_id": assignment.candidate_id,
                "paired_result_hash": paired.result_hash,
                "map50_95": paired.metric_deltas["map50_95"].model_dump(mode="json"),
            }
            state.evidence[assignment.stage_id] = paired.result_hash
            state.queued_stage = None
            state.queue_id = None
            state.refresh_satisfied_evidence()
            state.to_yaml(path)
            states[component_id] = {
                "status": state.status,
                "path": str(path),
                "paired_result_hash": paired.result_hash,
            }
        write_yaml(self.reproduction_path, {
            "schema_version": "paper_component_reproduction.v1",
            "components": states,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        return bool(states)

    def _record_history(self, assignment: ASHAAssignment, effect_delta: float) -> None:
        record = self.state.candidates[assignment.trial_id]
        self.state.family_last_round[record.component_family] = record.round_index
        if assignment.stage_id in {"pilot_10", "candidate_full_seed_1", "candidate_full_confirmation"}:
            self.state.completed_prior_ids = sorted(set([
                *self.state.completed_prior_ids,
                record.prior_id,
            ]))
        history_path = self.artifact_dir / "paper_candidate_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({
                "assignment_id": assignment.assignment_id,
                "trial_id": assignment.trial_id,
                "candidate_id": assignment.candidate_id,
                "stage_id": assignment.stage_id,
                "recipe_id": record.recipe_id,
                "prior_id": record.prior_id,
                "component_family": record.component_family,
                "effect_delta": effect_delta,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }, sort_keys=True) + "\n")

    def _load_state(self, base_run_id: str) -> PaperCandidateState:
        if not self.state_path.exists():
            return PaperCandidateState(base_run_id=base_run_id)
        payload = _read_yaml_mapping(self.state_path)
        state = PaperCandidateState.model_validate(payload)
        if state.base_run_id != base_run_id:
            raise ValueError(
                f"Paper candidate state belongs to {state.base_run_id}, not {base_run_id}"
            )
        return state

    def _save(self) -> None:
        self.state.updated_at = datetime.now(timezone.utc)
        write_yaml(self.state_path, self.state.model_dump(mode="json"))
        self.asha_store.save(self.scheduler)
        self._write_report()

    def _write_report(self) -> None:
        write_yaml(self.report_path, {
            "schema_version": "paper_recipe_report.v1",
            "base_run_id": self.state.base_run_id,
            "asha_is_budget_authority": True,
            "queue_authority": "RoundExecutionPlan",
            "policy_evaluation_queue_authority": False,
            "registered_candidates": [item.model_dump(mode="json") for item in self.state.candidates.values()],
            "assignments": [item.model_dump(mode="json") for item in self.scheduler.study.assignments],
            "trial_status": {
                item.candidate_id: item.status for item in self.scheduler.study.trials
            },
            "pending_evidence_recovery": self.state.pending_recovery,
            "full_candidate_recommendations": sorted(
                item.candidate_id
                for item in self.scheduler.study.trials
                if item.status == "full_pending_confirmation"
            ),
            "policy_memory_path": str(self.policy_memory.path),
            "reproduction_index_path": str(self.reproduction_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def _ledger_registration(
        self,
        submission: PaperCandidateSubmission,
        decision: str,
        reason: str,
    ) -> None:
        self.ledger.append(DecisionLedgerRecord(
            run_id=submission.decision_context.run_id,
            policy_id=submission.recipe_prior.prior_id,
            decision_type="paper_candidate_registration",
            proposal={
                "recipe_id": submission.recipe.recipe_id,
                "prior_id": submission.recipe_prior.prior_id,
                "component_ids": submission.recipe.component_ids,
                "snapshot_hash": submission.research_snapshot.snapshot_hash,
                "eligibility_token": submission.eligibility.eligibility_token,
            },
            decision=decision,
            blocked_by=[] if decision == "registered" else [reason],
            created_candidate_id=submission.source_node.candidate_config.candidate_id,
            created_node_id=submission.source_node.node_id,
            rationale=reason,
            policy_version=self.policy_version,
        ))

    def _ledger_step(
        self,
        assignment: ASHAAssignment,
        plan: RoundExecutionPlan,
        decision: str,
        missing: list[str],
    ) -> None:
        record = self.state.candidates[assignment.trial_id]
        self.ledger.append(DecisionLedgerRecord(
            run_id=plan.run_id,
            policy_id=record.prior_id,
            decision_type="paper_candidate_assignment",
            proposal={
                "assignment_id": assignment.assignment_id,
                "stage_id": assignment.stage_id,
                "recipe_id": record.recipe_id,
                "snapshot_hash": record.research_snapshot_hash,
                "queue_authority": "RoundExecutionPlan",
                "policy_evaluation_queue_authority": False,
            },
            decision=decision,
            missing_evidence=missing,
            created_candidate_id=assignment.candidate_id,
            created_node_id=plan.execution_nodes[0].node_id if plan.execution_nodes else None,
            rationale="ASHA is the sole budget authority",
            policy_version=self.policy_version,
        ))

    def _ledger_result(
        self,
        assignment: ASHAAssignment,
        trial_status: str,
        complete: bool,
        missing: list[str],
        paired: PairedExperimentResult | None,
    ) -> None:
        record = self.state.candidates[assignment.trial_id]
        self.ledger.append(DecisionLedgerRecord(
            run_id=(paired.run_id if paired is not None else self.state.base_run_id),
            policy_id=record.prior_id,
            decision_type="paper_candidate_result",
            proposal={
                "assignment_id": assignment.assignment_id,
                "stage_id": assignment.stage_id,
                "recipe_id": record.recipe_id,
                "paired_result_hash": paired.result_hash if paired is not None else None,
                "post_eval_complete": complete,
            },
            decision=trial_status,
            missing_evidence=missing,
            created_candidate_id=assignment.candidate_id,
            created_node_id=paired.candidate_node_id if paired is not None else assignment.assigned_node_id,
            rationale=(
                "verified paired evidence reported to ASHA"
                if complete else "evidence recovery required before ASHA can promote"
            ),
            policy_version=self.policy_version,
        ))


def _prepared_source_node(submission: PaperCandidateSubmission) -> ExperimentNode:
    metadata = dict(submission.source_node.command_spec.metadata)
    metadata.update({
        "paper_candidate": True,
        "paper_prior_id": submission.recipe_prior.prior_id,
        "paper_recipe_id": submission.recipe.recipe_id,
        "research_snapshot_hash": submission.research_snapshot.snapshot_hash,
        "eligibility_token": submission.eligibility.eligibility_token,
        "matched_pilot_required": True,
        "post_eval_required": True,
        "imgsz": 640,
    })
    command_spec = submission.source_node.command_spec.model_copy(update={"metadata": metadata})
    return submission.source_node.model_copy(update={
        "command_spec": command_spec,
        "command": command_spec.display(),
    })


def _node_has_fixed_imgsz(node: ExperimentNode) -> bool:
    values = [
        node.command_spec.metadata.get("imgsz"),
        _argv_option(node.command_spec.argv, "imgsz"),
        node.changed_variables.get("imgsz"),
    ]
    declared = [value for value in values if value is not None]
    return bool(declared) and all(str(value) == "640" for value in declared)


def _objective_hash_from_node(node: ExperimentNode) -> str:
    return str(node.command_spec.metadata.get("objective_hash") or "unknown")


def _protocol_hash_from_node(node: ExperimentNode) -> str:
    return str(node.command_spec.metadata.get("protocol_hash") or "unknown")


def _assignment_by_id(
    scheduler: ASHAScheduler,
    assignment_id: str,
) -> ASHAAssignment | None:
    return next(
        (item for item in scheduler.study.assignments if item.assignment_id == assignment_id),
        None,
    )


def _argv_option(argv: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for item in argv:
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


__all__ = [
    "PaperCandidateEvidence",
    "PaperCandidateOrchestrator",
    "PaperCandidateOrchestratorConfig",
    "PaperCandidateRegistrationReport",
    "PaperCandidateResultUpdate",
    "PaperCandidateStep",
    "PaperCandidateSubmission",
]
