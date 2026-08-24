"""Persistent ASHA budget allocation across automatic optimization rounds."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import stdev
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from yolo_agent.agents.candidate_generator import CandidateEvaluationContract
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.execution_fingerprint import (
    execution_fingerprint,
    execution_identity_payload,
    paired_evidence_is_valid,
)
from yolo_agent.core.paired_experiment import PairedExperimentResult
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.core.readiness_state import ReadinessState


ASHA_SCHEMA_VERSION = "1.3"
ASHAStageId = Literal["pilot_3", "pilot_10", "candidate_full_seed_1", "candidate_full_confirmation"]
ASHAAssignmentStatus = Literal["issued", "running", "completed", "failed", "deferred"]
ASHATrialStatus = Literal[
    "waiting",
    "running",
    "promotion_pending",
    "full_pending_confirmation",
    "confirmation_pending",
    "eliminated",
    "confirmed",
    "failed",
    "needs_evidence",
]


class ASHARungSpec(BaseModel):
    """One resource rung and its deterministic promotion guard."""

    stage_id: ASHAStageId
    epochs: int = Field(ge=1)
    fraction: float = Field(gt=0.0, le=1.0)
    reduction_factor: int = Field(default=3, ge=2)
    minimum_completed: int = Field(default=3, ge=1)
    minimum_promotions: int = Field(default=0, ge=0)
    require_positive_paired_delta: bool = True
    paired_delta_noise_floor: float | None = None
    require_target_error_improvement: bool = False


class ASHAObservation(BaseModel):
    """Imported evidence for one trial at one ASHA rung."""

    stage_id: ASHAStageId
    node_id: str
    seed_index: int = Field(default=1, ge=1)
    seed: int | str
    paired_delta: float | None = None
    paired_result_verified: bool = False
    paired_result_hash: str | None = None
    protocol_match_status: str | None = None
    paired_experiment_result: PairedExperimentResult | None = None
    target_error_improved_count: int = Field(default=0, ge=0)
    latency_regression: float | None = None
    model_size_regression: float | None = None
    diagnosis_gate_passed: bool | None = None
    diagnosis_checks: list[dict[str, object]] = Field(default_factory=list)
    promotion_rejection_reasons: list[str] = Field(default_factory=list)
    evidence_complete: bool = True
    failure_reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ASHATrial(BaseModel):
    """One guarded candidate tracked across child runs and fidelity levels."""

    trial_id: str
    candidate_id: str
    source_run_id: str
    source_node: ExperimentNode
    recipe_fingerprint: str = ""
    execution_fingerprint: str = ""
    paper_ids: list[str] = Field(default_factory=list)
    method_profile_ids: list[str] = Field(default_factory=list)
    mechanism_ids: list[str] = Field(default_factory=list)
    combination_id: str | None = None
    combination_fingerprint: str | None = None
    required_evidence: list[str] = Field(default_factory=list)
    readiness_state: ReadinessState | None = None
    readiness_blockers: list[str] = Field(default_factory=list)
    paper_specific_configuration: dict[str, object] = Field(default_factory=dict)
    baseline_control_node: ExperimentNode | None = None
    target_error_facts: list[dict[str, object]] = Field(default_factory=list)
    evaluation_contract: CandidateEvaluationContract = Field(
        default_factory=CandidateEvaluationContract
    )
    status: ASHATrialStatus = "waiting"
    pending_stage: ASHAStageId | None = "pilot_3"
    observations: list[ASHAObservation] = Field(default_factory=list)
    eliminated_reason: str = ""
    deferred_reason: str = ""
    confirmation_ci_low: float | None = None
    confirmation_ci_high: float | None = None
    promoted_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def fill_recipe_fingerprint(self) -> "ASHATrial":
        computed = _recipe_fingerprint(self.source_node)
        if self.execution_fingerprint and self.recipe_fingerprint:
            if self.execution_fingerprint != self.recipe_fingerprint:
                raise ValueError("ASHA execution and legacy recipe fingerprints disagree")
        fingerprint = self.execution_fingerprint or self.recipe_fingerprint or computed
        self.execution_fingerprint = fingerprint
        self.recipe_fingerprint = fingerprint
        self.paper_ids = sorted(set(self.paper_ids))
        self.method_profile_ids = sorted(set(self.method_profile_ids))
        self.mechanism_ids = sorted(set(self.mechanism_ids))
        self.required_evidence = sorted(set(self.required_evidence))
        return self

    def observation(self, stage_id: ASHAStageId, seed_index: int = 1) -> ASHAObservation | None:
        """Return the newest observation for a rung and confirmation seed index."""
        matches = [
            item
            for item in self.observations
            if item.stage_id == stage_id and item.seed_index == seed_index
        ]
        return max(matches, key=lambda item: item.created_at) if matches else None


class ASHAAssignment(BaseModel):
    """The next bounded training allocation selected by ASHA."""

    assignment_id: str = ""
    trial_id: str
    candidate_id: str
    stage_id: ASHAStageId
    seed_index: int = Field(default=1, ge=1)
    seed: int | str
    epochs: int
    fraction: float
    reason: str
    status: ASHAAssignmentStatus = "issued"
    assigned_run_id: str | None = None
    assigned_node_id: str | None = None
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def fill_assignment_id(self) -> "ASHAAssignment":
        if not self.assignment_id:
            self.assignment_id = (
                f"{self.trial_id}:{self.stage_id}:seed{self.seed_index}"
            )
        return self


class ASHAStudy(BaseModel, YAMLModelMixin):
    """Replayable ASHA state shared by all child runs of a base optimization run."""

    schema_version: str = ASHA_SCHEMA_VERSION
    study_id: str
    base_run_id: str
    run_protocol_hash: str | None = None
    rungs: list[ASHARungSpec] = Field(default_factory=list)
    trials: list[ASHATrial] = Field(default_factory=list)
    assignments: list[ASHAAssignment] = Field(default_factory=list)
    confirmation_seeds: list[int] = Field(default_factory=lambda: [42, 43, 44], min_length=3)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def trial(self, trial_id: str) -> ASHATrial:
        """Return one trial or raise a stable lookup error."""
        for item in self.trials:
            if item.trial_id == trial_id:
                return item
        raise KeyError(f"Unknown ASHA trial: {trial_id}")


class ASHAScheduler:
    """Allocate later budgets only after enough lower-rung evidence exists."""

    def __init__(self, study: ASHAStudy) -> None:
        self.study = study
        self._quarantine_unauthorized_paper_trials()

    def _quarantine_unauthorized_paper_trials(self) -> None:
        """Prevent legacy persisted paper trials from bypassing readiness."""
        changed = False
        for trial in self.study.trials:
            if not _is_paper_candidate_node(trial.source_node):
                continue
            metadata = trial.source_node.command_spec.metadata if trial.source_node.command_spec else {}
            if str(metadata.get("paper_readiness_state", "")) == "asha_eligible" and not _readiness_blockers(metadata):
                continue
            if trial.status in {"waiting", "running", "promotion_pending", "full_pending_confirmation", "confirmation_pending"}:
                trial.status = "needs_evidence"
                trial.pending_stage = None
                trial.eliminated_reason = "paper_readiness_state_missing_or_not_eligible"
                trial.readiness_state = "pre_registered"
                trial.readiness_blockers = ["paper_readiness_state_missing_or_not_eligible"]
                changed = True
        if changed:
            self._touch()

    @classmethod
    def create(cls, base_run_id: str) -> "ASHAScheduler":
        """Create the default COCO ASHA ladder."""
        return cls(
            ASHAStudy(
                study_id=f"{base_run_id}_asha",
                base_run_id=base_run_id,
                rungs=default_asha_rungs(),
            )
        )

    def register_trial(
        self,
        *,
        trial_id: str,
        candidate_id: str,
        source_run_id: str,
        source_node: ExperimentNode,
        baseline_control_node: ExperimentNode | None = None,
        target_error_facts: list[dict[str, object]] | None = None,
        paper_ids: list[str] | None = None,
        method_profile_ids: list[str] | None = None,
        mechanism_ids: list[str] | None = None,
        combination_id: str | None = None,
        combination_fingerprint: str | None = None,
        required_evidence: list[str] | None = None,
        paper_specific_configuration: dict[str, object] | None = None,
        readiness_state: ReadinessState | None = None,
        readiness_blockers: list[str] | None = None,
    ) -> ASHATrial:
        """Register a guarded candidate once without resetting prior evidence."""
        if any(
            str(item).startswith("inference.")
            for item in source_node.candidate_config.components
        ):
            raise ValueError("inference-only candidate cannot enter training ASHA")
        paper_candidate = _is_paper_candidate_node(source_node)
        shadow_evidence_only = _is_assignment_shadow_node(source_node)
        if paper_candidate:
            metadata = source_node.command_spec.metadata if source_node.command_spec else {}
            node_state = str(
                metadata.get("paper_readiness_state", metadata.get("readiness_state", ""))
            )
            node_blockers = _readiness_blockers(metadata)
            if shadow_evidence_only:
                # Assignment shadows are evidence allocations. They may be
                # scheduled by ASHA, but they never receive an mAP claim.
                if node_state not in {"", "pre_registered", "shadow_evidence_complete"}:
                    raise ValueError(
                        "assignment shadow requires an evidence-only readiness state; "
                        f"received {node_state}"
                    )
                if node_blockers:
                    raise ValueError(
                        "assignment shadow cannot retain readiness blockers: "
                        + ",".join(node_blockers)
                    )
                readiness_state = readiness_state or "pre_registered"
            elif node_state != "asha_eligible":
                raise ValueError(
                    "paper ASHA registration requires paper_readiness_state=asha_eligible; "
                    f"received {node_state or 'missing'}"
                )
            if not shadow_evidence_only and node_blockers:
                raise ValueError(
                    "paper ASHA registration cannot retain readiness blockers: "
                    + ",".join(node_blockers)
                )
            if not shadow_evidence_only:
                readiness_state = readiness_state or "asha_eligible"
        if (
            readiness_state is not None
            and readiness_state != "asha_eligible"
            and not shadow_evidence_only
        ):
            raise ValueError(
                "ASHA trial registration requires readiness_state=asha_eligible; "
                f"received {readiness_state}"
            )
        if readiness_state == "asha_eligible" and readiness_blockers:
            raise ValueError("ASHA-eligible trial cannot retain readiness blockers")
        recipe_fingerprint = _recipe_fingerprint(source_node)
        for trial in self.study.trials:
            if trial.trial_id == trial_id:
                if trial.readiness_state == "pre_registered":
                    _activate_pre_registered_trial(
                        trial,
                        baseline_control_node=baseline_control_node,
                        required_evidence=required_evidence,
                    )
                _merge_trial_provenance(
                    trial,
                    paper_ids,
                    method_profile_ids,
                    mechanism_ids=mechanism_ids,
                )
                return trial
            if (
                trial.execution_fingerprint == recipe_fingerprint
                and (
                    (
                        trial.observation("pilot_3") is None
                        and trial.status in {"waiting", "running", "needs_evidence"}
                    )
                    or _trial_has_valid_paired_evidence(trial)
                )
            ):
                if trial.readiness_state == "pre_registered":
                    _activate_pre_registered_trial(
                        trial,
                        baseline_control_node=baseline_control_node,
                        required_evidence=required_evidence,
                    )
                _merge_trial_provenance(
                    trial,
                    paper_ids,
                    method_profile_ids,
                    mechanism_ids=mechanism_ids,
                )
                self._touch()
                return trial
        trial = ASHATrial(
            trial_id=trial_id,
            candidate_id=candidate_id,
            source_run_id=source_run_id,
            source_node=source_node,
            recipe_fingerprint=recipe_fingerprint,
            execution_fingerprint=recipe_fingerprint,
            paper_ids=sorted(set(paper_ids or [])),
            method_profile_ids=sorted(set(method_profile_ids or [])),
            mechanism_ids=sorted(set(mechanism_ids or [])),
            combination_id=combination_id,
            combination_fingerprint=combination_fingerprint,
            required_evidence=sorted(set(required_evidence or [])),
            readiness_state=readiness_state,
            readiness_blockers=sorted(set(readiness_blockers or [])),
            paper_specific_configuration=dict(paper_specific_configuration or {}),
            baseline_control_node=baseline_control_node,
            target_error_facts=list(target_error_facts or []),
            evaluation_contract=source_node.candidate_config.evaluation_contract,
        )
        self.study.trials.append(trial)
        self._touch()
        return trial

    def pre_register_trial(
        self,
        *,
        trial_id: str,
        candidate_id: str,
        source_run_id: str,
        source_node: ExperimentNode,
        baseline_control_node: ExperimentNode | None = None,
        target_error_facts: list[dict[str, object]] | None = None,
        paper_ids: list[str] | None = None,
        method_profile_ids: list[str] | None = None,
        mechanism_ids: list[str] | None = None,
        combination_id: str | None = None,
        combination_fingerprint: str | None = None,
        required_evidence: list[str] | None = None,
        paper_specific_configuration: dict[str, object] | None = None,
        blockers: list[str] | None = None,
    ) -> ASHATrial:
        """Reserve a non-runnable ASHA identity for a blocked candidate.

        This is deliberately separate from :meth:`register_trial`: a
        pre-registered trial is provenance and recovery state only.  It has
        no pending rung and can never be returned by ``next_assignment`` until
        a later eligible registration upgrades the same execution identity.
        """
        if any(
            str(item).startswith("inference.")
            for item in source_node.candidate_config.components
        ):
            raise ValueError("inference-only candidate cannot enter training ASHA")
        fingerprint = _recipe_fingerprint(source_node)
        trial = next(
            (
                item
                for item in self.study.trials
                if item.trial_id == trial_id
                or item.execution_fingerprint == fingerprint
            ),
            None,
        )
        if trial is not None:
            if trial.readiness_state != "asha_eligible":
                trial.readiness_state = "pre_registered"
                trial.status = "needs_evidence"
                trial.pending_stage = None
                trial.readiness_blockers = sorted(set(blockers or []))
                trial.eliminated_reason = ""
                trial.deferred_reason = ""
                trial.baseline_control_node = baseline_control_node
                trial.required_evidence = sorted(
                    set(trial.required_evidence) | set(required_evidence or [])
                )
                trial.target_error_facts = list(target_error_facts or trial.target_error_facts)
                trial.updated_at = datetime.now(timezone.utc)
            _merge_trial_provenance(
                trial,
                paper_ids,
                method_profile_ids,
                mechanism_ids=mechanism_ids,
            )
            self._touch()
            return trial
        trial = ASHATrial(
            trial_id=trial_id,
            candidate_id=candidate_id,
            source_run_id=source_run_id,
            source_node=source_node,
            recipe_fingerprint=fingerprint,
            execution_fingerprint=fingerprint,
            paper_ids=sorted(set(paper_ids or [])),
            method_profile_ids=sorted(set(method_profile_ids or [])),
            mechanism_ids=sorted(set(mechanism_ids or [])),
            combination_id=combination_id,
            combination_fingerprint=combination_fingerprint,
            required_evidence=sorted(set(required_evidence or [])),
            readiness_state="pre_registered",
            readiness_blockers=sorted(set(blockers or [])),
            paper_specific_configuration=dict(paper_specific_configuration or {}),
            baseline_control_node=baseline_control_node,
            target_error_facts=list(target_error_facts or []),
            evaluation_contract=source_node.candidate_config.evaluation_contract,
            status="needs_evidence",
            pending_stage=None,
        )
        self.study.trials.append(trial)
        self._touch()
        return trial

    def report(self, trial_id: str, observation: ASHAObservation) -> ASHATrial:
        """Import one rung result and update promotion eligibility."""
        trial = self.study.trial(trial_id)
        trial.observations = [
            item
            for item in trial.observations
            if not (
                item.stage_id == observation.stage_id
                and item.seed_index == observation.seed_index
            )
        ]
        trial.observations.append(observation)
        self._finish_assignment(
            trial_id,
            observation,
            failed=bool(observation.failure_reason),
        )
        trial.updated_at = datetime.now(timezone.utc)
        if observation.failure_reason:
            trial.status = "failed"
            trial.pending_stage = None
            trial.eliminated_reason = observation.failure_reason
        elif (
            not observation.evidence_complete
            or observation.paired_delta is None
            or not observation.paired_result_verified
            or observation.paired_experiment_result is None
            or not observation.paired_experiment_result.verified
        ):
            trial.status = "needs_evidence"
            trial.pending_stage = observation.stage_id
        elif observation.stage_id == "pilot_3":
            trial.status = "waiting"
            trial.pending_stage = None
            self._refresh_pilot_3_promotions()
        elif observation.stage_id == "pilot_10":
            self._finish_pilot_10(trial, observation)
        elif observation.stage_id == "candidate_full_seed_1":
            self._finish_full_seed_1(trial, observation)
        else:
            self._finish_confirmation(trial)
        self._touch()
        return trial

    def next_assignment(self, *, confirm_full_run: bool = False) -> ASHAAssignment | None:
        """Issue or recover one assignment; completed allocations are never reissued."""
        paper_pending = self._has_executable_paper_trial(
            confirm_full_run=confirm_full_run
        )
        self._defer_native_outstanding_assignments(paper_pending=paper_pending)
        outstanding = self._recoverable_assignment(
            confirm_full_run=confirm_full_run,
            paper_only=paper_pending,
        )
        if outstanding is not None:
            return outstanding
        if confirm_full_run:
            confirmation = self._next_confirmation_assignment(paper_only=paper_pending)
            if confirmation is not None:
                return confirmation
            for trial in self._ordered_trials(paper_only=paper_pending):
                if trial.status == "full_pending_confirmation":
                    return self._assignment(trial, "candidate_full_seed_1", seed_index=1)
        for trial in self._ordered_trials(paper_only=paper_pending):
            if (
                trial.status == "waiting"
                and trial.pending_stage == "pilot_3"
                and trial.observation("pilot_3") is None
            ):
                return self._assignment(trial, "pilot_3", seed_index=1)
        for trial in self._ordered_trials(paper_only=paper_pending):
            if trial.status == "promotion_pending" and trial.pending_stage == "pilot_10":
                return self._assignment(trial, "pilot_10", seed_index=1)
        return None

    def mark_running(
        self,
        assignment: ASHAAssignment,
        *,
        run_id: str | None = None,
        node_id: str | None = None,
    ) -> ASHATrial:
        """Claim an issued assignment or idempotently resume its bound execution."""
        persisted = self._persisted_assignment(assignment.assignment_id)
        if persisted is None:
            raise KeyError(f"Unknown ASHA assignment: {assignment.assignment_id}")
        if persisted.status in {"completed", "failed"}:
            raise RuntimeError(
                f"ASHA assignment {persisted.assignment_id} was already consumed as {persisted.status}."
            )
        if persisted.status == "running":
            if run_id and persisted.assigned_run_id not in {None, run_id}:
                raise RuntimeError(
                    f"ASHA assignment {persisted.assignment_id} is already bound to {persisted.assigned_run_id}."
                )
            if node_id and persisted.assigned_node_id not in {None, node_id}:
                raise RuntimeError(
                    f"ASHA assignment {persisted.assignment_id} is already bound to {persisted.assigned_node_id}."
                )
        persisted.status = "running"
        persisted.assigned_run_id = persisted.assigned_run_id or run_id
        persisted.assigned_node_id = persisted.assigned_node_id or node_id
        persisted.started_at = persisted.started_at or datetime.now(timezone.utc)
        trial = self.study.trial(assignment.trial_id)
        trial.status = "running"
        trial.pending_stage = assignment.stage_id
        trial.updated_at = datetime.now(timezone.utc)
        self._touch()
        return trial

    def complete_evidence_only_trial(
        self,
        trial_id: str,
        *,
        node_id: str,
        reason: str,
        succeeded: bool,
    ) -> ASHATrial:
        """Consume a non-ranking evidence allocation without creating an mAP observation."""
        trial = self.study.trial(trial_id)
        now = datetime.now(timezone.utc)
        for assignment in self.study.assignments:
            if assignment.trial_id != trial_id or assignment.status not in {"issued", "running"}:
                continue
            assignment.status = "completed" if succeeded else "failed"
            assignment.assigned_node_id = assignment.assigned_node_id or node_id
            assignment.completed_at = assignment.completed_at or now
        trial.status = "eliminated" if succeeded else "failed"
        trial.pending_stage = None
        trial.eliminated_reason = reason
        trial.updated_at = datetime.now(timezone.utc)
        self._touch()
        return trial

    def _recoverable_assignment(
        self,
        *,
        confirm_full_run: bool,
        paper_only: bool = False,
    ) -> ASHAAssignment | None:
        candidates = [
            assignment
            for assignment in self.study.assignments
            if assignment.status in {"issued", "running"}
            and not (
                assignment.stage_id.startswith("candidate_full")
                and not confirm_full_run
            )
            and (
                not paper_only
                or self._is_paper_trial(self.study.trial(assignment.trial_id))
            )
        ]
        if not candidates and paper_only:
            return None
        for assignment in candidates:
            return assignment
        return None

    def _has_executable_paper_trial(self, *, confirm_full_run: bool) -> bool:
        """Return whether an executable paper adapter still needs ASHA work."""
        runnable_statuses = {
            "waiting",
            "running",
            "promotion_pending",
        }
        if confirm_full_run:
            runnable_statuses.update(
                {"full_pending_confirmation", "confirmation_pending"}
            )
        return any(
            self._is_paper_trial(trial)
            and trial.status in runnable_statuses
            for trial in self.study.trials
        )

    def _ordered_trials(self, *, paper_only: bool) -> list[ASHATrial]:
        """Return paper trials first, preserving registration order within a class."""
        trials = [
            trial
            for trial in self.study.trials
            if not paper_only or self._is_paper_trial(trial)
        ]
        return sorted(trials, key=lambda trial: not self._is_paper_trial(trial))

    def _defer_native_outstanding_assignments(self, *, paper_pending: bool) -> None:
        """Pause stale native work while executable paper methods are available."""
        if not paper_pending:
            return
        changed = False
        for assignment in self.study.assignments:
            if assignment.status not in {"issued", "running"}:
                continue
            trial = self.study.trial(assignment.trial_id)
            if self._is_paper_trial(trial):
                continue
            assignment.status = "deferred"
            trial.status, trial.pending_stage = _deferred_trial_state(assignment.stage_id)
            trial.deferred_reason = "native_trial_deferred_for_executable_paper_adapter"
            changed = True
        if changed:
            self._touch()

    @staticmethod
    def _is_paper_trial(trial: ASHATrial) -> bool:
        """Identify a real adapter-backed paper candidate, not a paper label."""
        config = trial.source_node.candidate_config
        command = trial.source_node.command_spec
        metadata = command.metadata if command is not None else {}
        return bool(
            config.components
            and metadata.get("adapter_runtime_entrypoint")
            and not any(str(item).startswith("inference.") for item in config.components)
            and metadata.get("training_attribution_allowed", True) is not False
        )

    def _persisted_assignment(self, assignment_id: str) -> ASHAAssignment | None:
        return next(
            (item for item in self.study.assignments if item.assignment_id == assignment_id),
            None,
        )

    def _finish_assignment(
        self,
        trial_id: str,
        observation: ASHAObservation,
        *,
        failed: bool,
    ) -> None:
        assignment_id = f"{trial_id}:{observation.stage_id}:seed{observation.seed_index}"
        assignment = self._persisted_assignment(assignment_id)
        if assignment is None:
            return
        assignment.status = "failed" if failed else "completed"
        assignment.assigned_node_id = assignment.assigned_node_id or observation.node_id
        assignment.completed_at = assignment.completed_at or datetime.now(timezone.utc)

    def _refresh_pilot_3_promotions(self) -> None:
        rung = self._rung("pilot_3")
        for is_paper in (True, False):
            cohort = [
                trial
                for trial in self.study.trials
                if self._is_paper_trial(trial) == is_paper
            ]
            completed = [
                trial
                for trial in cohort
                if (observation := trial.observation("pilot_3")) is not None
                and observation.evidence_complete
                and observation.paired_delta is not None
                and trial.status not in {"failed", "needs_evidence"}
            ]
            for trial in completed:
                observation = trial.observation("pilot_3")
                if (
                    observation is not None
                    and rung.paired_delta_noise_floor is not None
                    and observation.paired_delta < rung.paired_delta_noise_floor
                ):
                    trial.status = "eliminated"
                    trial.pending_stage = None
                    trial.eliminated_reason = "pilot_3_delta_below_noise_floor"
                elif (
                    observation is not None
                    and rung.require_positive_paired_delta
                    and observation.paired_delta <= 0
                ):
                    trial.status = "eliminated"
                    trial.pending_stage = None
                    trial.eliminated_reason = "pilot_3_non_positive_paired_delta"
                elif observation is not None and observation.diagnosis_gate_passed is False:
                    trial.status = "eliminated"
                    trial.pending_stage = None
                    trial.eliminated_reason = (
                        ";".join(observation.promotion_rejection_reasons)
                        or "pilot_3_diagnosis_promotion_gate_failed"
                    )
            eligible = [trial for trial in completed if trial.status != "eliminated"]
            pending_cohort = [
                trial
                for trial in cohort
                if trial.pending_stage == "pilot_3"
                and trial.observation("pilot_3") is None
                and trial.status not in {"eliminated", "failed"}
            ]
            if pending_cohort:
                continue
            if not completed:
                continue
            # Paper adapters are often registered one at a time. Once that
            # paper cohort is complete, do not wait for unrelated native knobs
            # to manufacture the minimum cohort size.
            minimum_completed = 1 if is_paper else rung.minimum_completed
            if len(completed) < minimum_completed:
                continue
            slots = (
                max(1, rung.minimum_promotions, len(completed) // rung.reduction_factor)
                if is_paper
                else max(rung.minimum_promotions, len(completed) // rung.reduction_factor)
            )
            slots = min(slots, len(eligible))
            if slots <= 0:
                continue
            ranked = sorted(
                eligible,
                key=lambda trial: trial.observation("pilot_3").paired_delta,  # type: ignore[union-attr]
                reverse=True,
            )
            for trial in ranked[:slots]:
                if trial.observation("pilot_10") is None and trial.status == "waiting":
                    trial.status = "promotion_pending"
                    trial.pending_stage = "pilot_10"
                    trial.promoted_at = datetime.now(timezone.utc)

    def _finish_pilot_10(self, trial: ASHATrial, observation: ASHAObservation) -> None:
        rung = self._rung("pilot_10")
        if rung.require_positive_paired_delta and observation.paired_delta is not None and observation.paired_delta <= 0:
            trial.status = "eliminated"
            trial.pending_stage = None
            trial.eliminated_reason = "pilot_10_non_positive_paired_delta"
            return
        if observation.diagnosis_gate_passed is not True:
            trial.status = "eliminated"
            trial.pending_stage = None
            trial.eliminated_reason = (
                ";".join(observation.promotion_rejection_reasons)
                or "pilot_10_diagnosis_promotion_gate_failed"
            )
            return
        if rung.require_target_error_improvement and observation.target_error_improved_count < 1:
            trial.status = "eliminated"
            trial.pending_stage = None
            trial.eliminated_reason = "pilot_10_target_error_fact_not_improved"
            return
        trial.status = "full_pending_confirmation"
        trial.pending_stage = "candidate_full_seed_1"

    def _finish_full_seed_1(self, trial: ASHATrial, observation: ASHAObservation) -> None:
        if observation.paired_delta is None or observation.paired_delta <= 0:
            trial.status = "eliminated"
            trial.pending_stage = None
            trial.eliminated_reason = "candidate_full_seed_1_non_positive_paired_delta"
            return
        if observation.diagnosis_gate_passed is not True:
            trial.status = "eliminated"
            trial.pending_stage = None
            trial.eliminated_reason = (
                ";".join(observation.promotion_rejection_reasons)
                or "candidate_full_seed_1_diagnosis_gate_failed"
            )
            return
        trial.status = "confirmation_pending"
        trial.pending_stage = "candidate_full_confirmation"

    def _finish_confirmation(self, trial: ASHATrial) -> None:
        observations = [
            trial.observation("candidate_full_seed_1", 1),
            *[
                trial.observation("candidate_full_confirmation", seed_index)
                for seed_index in range(2, len(self.study.confirmation_seeds) + 1)
            ],
        ]
        if any(item is None for item in observations):
            trial.status = "confirmation_pending"
            trial.pending_stage = "candidate_full_confirmation"
            return
        complete = all(
            item is not None
            and item.paired_delta is not None
            and item.paired_delta > 0
            and item.diagnosis_gate_passed is True
            for item in observations
        )
        deltas = [float(item.paired_delta) for item in observations if item and item.paired_delta is not None]
        interval = _paired_seed_confidence_interval(deltas)
        trial.confirmation_ci_low = interval[0] if interval is not None else None
        trial.confirmation_ci_high = interval[1] if interval is not None else None
        if complete and interval is not None and interval[0] > 0.0:
            trial.status = "confirmed"
            trial.pending_stage = None
            return
        trial.status = "eliminated"
        trial.pending_stage = None
        trial.eliminated_reason = (
            "candidate_full_confirmation_confidence_interval_not_positive"
            if complete
            else "candidate_full_confirmation_not_consistently_positive"
        )

    def _next_confirmation_assignment(self, *, paper_only: bool = False) -> ASHAAssignment | None:
        for trial in self._ordered_trials(paper_only=paper_only):
            if trial.status != "confirmation_pending":
                continue
            for seed_index in range(2, len(self.study.confirmation_seeds) + 1):
                if trial.observation("candidate_full_confirmation", seed_index) is None:
                    return self._assignment(trial, "candidate_full_confirmation", seed_index=seed_index)
        return None

    def _assignment(self, trial: ASHATrial, stage_id: ASHAStageId, *, seed_index: int) -> ASHAAssignment:
        rung = self._rung(stage_id)
        assignment = ASHAAssignment(
            trial_id=trial.trial_id,
            candidate_id=trial.candidate_id,
            stage_id=stage_id,
            seed_index=seed_index,
            seed=self.study.confirmation_seeds[seed_index - 1],
            epochs=rung.epochs,
            fraction=rung.fraction,
            reason=f"asha_budget_promoted_to_{stage_id}",
        )
        existing = self._persisted_assignment(assignment.assignment_id)
        if existing is not None:
            if existing.status == "deferred":
                existing.status = "issued"
                existing.assigned_run_id = None
                existing.assigned_node_id = None
                existing.issued_at = datetime.now(timezone.utc)
                existing.started_at = None
                existing.completed_at = None
                self._touch()
            return existing
        self.study.assignments.append(assignment)
        self._touch()
        return assignment

    def _rung(self, stage_id: ASHAStageId) -> ASHARungSpec:
        for rung in self.study.rungs:
            if rung.stage_id == stage_id:
                return rung
        raise KeyError(f"Missing ASHA rung: {stage_id}")

    def _touch(self) -> None:
        self.study.updated_at = datetime.now(timezone.utc)


def _is_paper_candidate_node(node: ExperimentNode) -> bool:
    config = node.candidate_config
    metadata = node.command_spec.metadata if node.command_spec is not None else {}
    return bool(
        config.components
        and metadata.get("adapter_runtime_entrypoint")
        and not any(str(item).startswith("inference.") for item in config.components)
    )


def _is_assignment_shadow_node(node: ExperimentNode) -> bool:
    """Return whether a node is an evidence-only assignment shadow."""
    command = node.command_spec
    metadata = command.metadata if command is not None else {}
    if str(metadata.get("assignment_execution_mode", "")) == "shadow":
        return True
    return any(
        str(key).startswith("training_config.assignment.")
        and str(value) == "shadow"
        for key, value in node.changed_variables.items()
    )


def _readiness_blockers(metadata: dict[str, object]) -> list[str]:
    raw = metadata.get("paper_readiness_blockers", metadata.get("readiness_blockers", []))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [raw] if raw else []
    return [str(item) for item in raw] if isinstance(raw, list) else []


def _activate_pre_registered_trial(
    trial: ASHATrial,
    *,
    baseline_control_node: ExperimentNode | None,
    required_evidence: list[str] | None,
) -> None:
    """Upgrade a reserved identity only after formal ASHA admission."""
    trial.readiness_state = "asha_eligible"
    trial.readiness_blockers = []
    trial.status = "waiting"
    trial.pending_stage = "pilot_3"
    trial.baseline_control_node = baseline_control_node or trial.baseline_control_node
    trial.required_evidence = sorted(
        set(trial.required_evidence) | set(required_evidence or [])
    )
    trial.eliminated_reason = ""
    trial.deferred_reason = ""
    trial.updated_at = datetime.now(timezone.utc)


class ASHAStudyStore:
    """Filesystem persistence for a base run's cross-round ASHA state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load_or_create(self, base_run_id: str) -> ASHAScheduler:
        """Load an existing study or initialize the default ladder."""
        if self.path.is_file():
            return ASHAScheduler(ASHAStudy.from_yaml(self.path))
        return ASHAScheduler.create(base_run_id)

    def save(self, scheduler: ASHAScheduler) -> Path:
        """Persist the complete scheduler state."""
        return scheduler.study.to_yaml(self.path)


def default_asha_rungs() -> list[ASHARungSpec]:
    """Return the fixed-imgsz COCO budget ladder."""
    return [
        ASHARungSpec(
            stage_id="pilot_3",
            epochs=3,
            fraction=0.1,
            reduction_factor=3,
            minimum_completed=3,
            require_positive_paired_delta=False,
            paired_delta_noise_floor=-0.0015,
        ),
        ASHARungSpec(
            stage_id="pilot_10",
            epochs=10,
            fraction=0.1,
            reduction_factor=2,
            minimum_completed=1,
            require_target_error_improvement=True,
        ),
        ASHARungSpec(
            stage_id="candidate_full_seed_1",
            epochs=100,
            fraction=1.0,
            minimum_completed=1,
            reduction_factor=2,
        ),
        ASHARungSpec(
            stage_id="candidate_full_confirmation",
            epochs=100,
            fraction=1.0,
            minimum_completed=1,
            reduction_factor=2,
        ),
    ]


def _deferred_trial_state(stage_id: ASHAStageId) -> tuple[ASHATrialStatus, ASHAStageId]:
    """Restore the scheduler state represented by a deferred allocation."""
    if stage_id == "pilot_3":
        return "waiting", "pilot_3"
    if stage_id == "pilot_10":
        return "promotion_pending", "pilot_10"
    if stage_id == "candidate_full_seed_1":
        return "full_pending_confirmation", "candidate_full_seed_1"
    return "confirmation_pending", "candidate_full_confirmation"


def _recipe_fingerprint(node: ExperimentNode) -> str:
    return execution_fingerprint(node)


def _trial_has_valid_paired_evidence(trial: ASHATrial) -> bool:
    identity = execution_identity_payload(trial.source_node)
    expected_protocol = str(identity["baseline_protocol_hash"])
    expected_dataset = str(identity["dataset_manifest_hash"])
    return any(
        observation.evidence_complete
        and observation.paired_result_verified
        and paired_evidence_is_valid(
            observation.paired_experiment_result,
            expected_candidate_id=trial.candidate_id,
            expected_protocol_hash=(
                expected_protocol if expected_protocol != "unknown" else None
            ),
            expected_dataset_manifest_hash=(
                expected_dataset if expected_dataset != "unknown" else None
            ),
        )
        for observation in trial.observations
    )


def _merge_trial_provenance(
    trial: ASHATrial,
    paper_ids: list[str] | None,
    method_profile_ids: list[str] | None,
    *,
    mechanism_ids: list[str] | None = None,
) -> None:
    trial.paper_ids = sorted(set(trial.paper_ids) | set(paper_ids or []))
    trial.method_profile_ids = sorted(
        set(trial.method_profile_ids) | set(method_profile_ids or [])
    )
    trial.mechanism_ids = sorted(set(trial.mechanism_ids) | set(mechanism_ids or []))
    trial.updated_at = datetime.now(timezone.utc)


def _paired_seed_confidence_interval(values: list[float]) -> tuple[float, float] | None:
    """Conservative two-sided 95% Student-t interval across paired seeds."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(
        len(values) - 1, 1.96
    )
    margin = critical * stdev(values) / (len(values) ** 0.5)
    return mean - margin, mean + margin
