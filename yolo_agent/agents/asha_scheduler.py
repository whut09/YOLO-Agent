"""Persistent ASHA budget allocation across automatic optimization rounds."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import stdev
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.paired_experiment import PairedExperimentResult
from yolo_agent.core.yaml_io import YAMLModelMixin


ASHA_SCHEMA_VERSION = "1.1"
ASHAStageId = Literal["pilot_3", "pilot_10", "candidate_full_seed_1", "candidate_full_confirmation"]
ASHAAssignmentStatus = Literal["issued", "running", "completed", "failed"]
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
    require_positive_paired_delta: bool = True
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
    baseline_control_node: ExperimentNode | None = None
    target_error_facts: list[dict[str, object]] = Field(default_factory=list)
    status: ASHATrialStatus = "waiting"
    pending_stage: ASHAStageId | None = "pilot_3"
    observations: list[ASHAObservation] = Field(default_factory=list)
    eliminated_reason: str = ""
    confirmation_ci_low: float | None = None
    confirmation_ci_high: float | None = None
    promoted_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def fill_recipe_fingerprint(self) -> "ASHATrial":
        if not self.recipe_fingerprint:
            self.recipe_fingerprint = _recipe_fingerprint(self.source_node)
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
    ) -> ASHATrial:
        """Register a guarded candidate once without resetting prior evidence."""
        recipe_fingerprint = _recipe_fingerprint(source_node)
        for trial in self.study.trials:
            if trial.trial_id == trial_id:
                return trial
            if (
                trial.recipe_fingerprint == recipe_fingerprint
                and trial.observation("pilot_3") is None
                and trial.status in {"waiting", "running", "needs_evidence"}
            ):
                return trial
        trial = ASHATrial(
            trial_id=trial_id,
            candidate_id=candidate_id,
            source_run_id=source_run_id,
            source_node=source_node,
            recipe_fingerprint=recipe_fingerprint,
            baseline_control_node=baseline_control_node,
            target_error_facts=list(target_error_facts or []),
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
        outstanding = self._recoverable_assignment(confirm_full_run=confirm_full_run)
        if outstanding is not None:
            return outstanding
        if confirm_full_run:
            confirmation = self._next_confirmation_assignment()
            if confirmation is not None:
                return confirmation
            for trial in self.study.trials:
                if trial.status == "full_pending_confirmation":
                    return self._assignment(trial, "candidate_full_seed_1", seed_index=1)
        for trial in self.study.trials:
            if (
                trial.status == "waiting"
                and trial.pending_stage == "pilot_3"
                and trial.observation("pilot_3") is None
            ):
                return self._assignment(trial, "pilot_3", seed_index=1)
        for trial in self.study.trials:
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

    def _recoverable_assignment(self, *, confirm_full_run: bool) -> ASHAAssignment | None:
        for assignment in self.study.assignments:
            if assignment.status not in {"issued", "running"}:
                continue
            if assignment.stage_id.startswith("candidate_full") and not confirm_full_run:
                continue
            return assignment
        return None

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
        completed = [
            trial
            for trial in self.study.trials
            if (observation := trial.observation("pilot_3")) is not None
            and observation.evidence_complete
            and observation.paired_delta is not None
            and trial.status not in {"failed", "needs_evidence"}
        ]
        for trial in completed:
            observation = trial.observation("pilot_3")
            if observation is not None and rung.require_positive_paired_delta and observation.paired_delta <= 0:
                trial.status = "eliminated"
                trial.pending_stage = None
                trial.eliminated_reason = "pilot_3_non_positive_paired_delta"
        eligible = [trial for trial in completed if trial.status != "eliminated"]
        pending_cohort = [
            trial
            for trial in self.study.trials
            if trial.pending_stage == "pilot_3" and trial.observation("pilot_3") is None
        ]
        if pending_cohort:
            return
        if len(completed) < rung.minimum_completed:
            return
        slots = len(completed) // rung.reduction_factor
        if slots <= 0:
            return
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

    def _next_confirmation_assignment(self) -> ASHAAssignment | None:
        for trial in self.study.trials:
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
        ASHARungSpec(stage_id="pilot_3", epochs=3, fraction=0.1, reduction_factor=3, minimum_completed=3),
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


def _recipe_fingerprint(node: ExperimentNode) -> str:
    payload = {
        "base_model": node.candidate_config.base_model,
        "components": sorted(node.candidate_config.components),
        "action_domain": node.candidate_config.action_domain,
        "action_id": node.candidate_config.action_id,
        "train_overrides": node.candidate_config.train_overrides,
        "changed_variables": node.changed_variables,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


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
