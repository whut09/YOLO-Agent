"""Persistent ASHA budget allocation across automatic optimization rounds."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.yaml_io import YAMLModelMixin


ASHA_SCHEMA_VERSION = "1.0"
ASHAStageId = Literal["pilot_3", "pilot_10", "candidate_full_seed_1", "candidate_full_confirmation"]
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
    target_error_improved_count: int = Field(default=0, ge=0)
    latency_regression: float | None = None
    model_size_regression: float | None = None
    evidence_complete: bool = True
    failure_reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ASHATrial(BaseModel):
    """One guarded candidate tracked across child runs and fidelity levels."""

    trial_id: str
    candidate_id: str
    source_run_id: str
    source_node: ExperimentNode
    target_error_facts: list[dict[str, object]] = Field(default_factory=list)
    status: ASHATrialStatus = "waiting"
    pending_stage: ASHAStageId | None = "pilot_3"
    observations: list[ASHAObservation] = Field(default_factory=list)
    eliminated_reason: str = ""
    promoted_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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

    trial_id: str
    candidate_id: str
    stage_id: ASHAStageId
    seed_index: int = Field(default=1, ge=1)
    seed: int | str
    epochs: int
    fraction: float
    reason: str


class ASHAStudy(BaseModel, YAMLModelMixin):
    """Replayable ASHA state shared by all child runs of a base optimization run."""

    schema_version: str = ASHA_SCHEMA_VERSION
    study_id: str
    base_run_id: str
    rungs: list[ASHARungSpec] = Field(default_factory=list)
    trials: list[ASHATrial] = Field(default_factory=list)
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
        target_error_facts: list[dict[str, object]] | None = None,
    ) -> ASHATrial:
        """Register a guarded candidate once without resetting prior evidence."""
        for trial in self.study.trials:
            if trial.trial_id == trial_id:
                return trial
        trial = ASHATrial(
            trial_id=trial_id,
            candidate_id=candidate_id,
            source_run_id=source_run_id,
            source_node=source_node,
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
        trial.updated_at = datetime.now(timezone.utc)
        if observation.failure_reason:
            trial.status = "failed"
            trial.pending_stage = None
            trial.eliminated_reason = observation.failure_reason
        elif not observation.evidence_complete or observation.paired_delta is None:
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
        """Return one pending promotion, preferring confirmation over new pilot budget."""
        if confirm_full_run:
            confirmation = self._next_confirmation_assignment()
            if confirmation is not None:
                return confirmation
            for trial in self.study.trials:
                if trial.status == "full_pending_confirmation":
                    return self._assignment(trial, "candidate_full_seed_1", seed_index=1)
        for trial in self.study.trials:
            if trial.status == "promotion_pending" and trial.pending_stage == "pilot_10":
                return self._assignment(trial, "pilot_10", seed_index=1)
        return None

    def mark_running(self, assignment: ASHAAssignment) -> ASHATrial:
        """Mark an assignment as consumed so it cannot be scheduled twice."""
        trial = self.study.trial(assignment.trial_id)
        trial.status = "running"
        trial.pending_stage = assignment.stage_id
        trial.updated_at = datetime.now(timezone.utc)
        self._touch()
        return trial

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
        if all(item is not None and item.paired_delta is not None and item.paired_delta > 0 for item in observations):
            trial.status = "confirmed"
            trial.pending_stage = None
            return
        trial.status = "eliminated"
        trial.pending_stage = None
        trial.eliminated_reason = "candidate_full_confirmation_not_consistently_positive"

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
        return ASHAAssignment(
            trial_id=trial.trial_id,
            candidate_id=trial.candidate_id,
            stage_id=stage_id,
            seed_index=seed_index,
            seed=self.study.confirmation_seeds[seed_index - 1],
            epochs=rung.epochs,
            fraction=rung.fraction,
            reason=f"asha_budget_promoted_to_{stage_id}",
        )

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
