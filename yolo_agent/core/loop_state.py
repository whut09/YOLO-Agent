"""State machine primitives for orchestrated optimization loops."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


LoopStage = Literal[
    "init",
    "profile_data",
    "advise_labels",
    "diagnose_errors",
    "generate_loop_plan",
    "evaluate_policies",
    "generate_candidates",
    "ablate",
    "smoke",
    "import_metrics",
    "report",
    "next_round",
]
StageStatus = Literal["pending", "running", "completed", "blocked", "failed", "skipped"]

DEFAULT_STAGE_ORDER: list[LoopStage] = [
    "init",
    "profile_data",
    "advise_labels",
    "diagnose_errors",
    "generate_loop_plan",
    "evaluate_policies",
    "generate_candidates",
    "ablate",
    "smoke",
    "import_metrics",
    "report",
    "next_round",
]


class LoopStageState(BaseModel):
    """State for one loop stage."""

    stage: LoopStage
    status: StageStatus = "pending"
    attempts: int = 0
    artifacts: dict[str, Path] = Field(default_factory=dict)
    message: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LoopState(BaseModel):
    """Serializable state for one run loop."""

    run_id: str
    dataset_version: str = "unversioned"
    task_spec: Path | None = None
    stage: LoopStage = "init"
    current_stage: LoopStage = "init"
    completed: list[LoopStage] = Field(default_factory=list)
    pending: list[LoopStage] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    failed: list[LoopStage] = Field(default_factory=list)
    skipped: list[LoopStage] = Field(default_factory=list)
    artifacts: dict[str, Path] = Field(default_factory=dict)
    stages: dict[LoopStage, LoopStageState] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        run_id: str,
        stage_order: list[LoopStage] | None = None,
        dataset_version: str = "unversioned",
        task_spec: Path | str | None = None,
    ) -> "LoopState":
        """Create a pending state machine."""
        order = stage_order or DEFAULT_STAGE_ORDER
        state = cls(
            run_id=run_id,
            dataset_version=dataset_version,
            task_spec=Path(task_spec) if task_spec is not None else None,
            stage=order[0],
            current_stage=order[0],
            stages={stage: LoopStageState(stage=stage) for stage in order},
        )
        state.refresh_summary()
        return state

    def mark(
        self,
        stage: LoopStage,
        status: StageStatus,
        message: str = "",
        artifacts: dict[str, Path] | None = None,
    ) -> None:
        """Update one stage."""
        record = self.stages.setdefault(stage, LoopStageState(stage=stage))
        record.status = status
        if status == "running":
            record.attempts += 1
        record.message = message
        if artifacts:
            record.artifacts.update(artifacts)
            self.artifacts.update(artifacts)
        now = datetime.now(timezone.utc)
        record.updated_at = now
        self.updated_at = now
        self.stage = stage
        self.current_stage = stage
        self.refresh_summary()

    def next_pending(self) -> LoopStage | None:
        """Return next pending stage in configured order."""
        for stage in DEFAULT_STAGE_ORDER:
            record = self.stages.get(stage)
            if record is not None and record.status == "pending":
                return stage
        return None

    def has_blocker(self) -> bool:
        """Return whether any stage is blocked or failed."""
        return any(stage.status in {"blocked", "failed"} for stage in self.stages.values())

    def first_blocked(self) -> LoopStage | None:
        """Return the first blocked stage in configured order."""
        for stage in DEFAULT_STAGE_ORDER:
            record = self.stages.get(stage)
            if record is not None and record.status == "blocked":
                return stage
        return None

    def reset_for_resume(self) -> LoopStage | None:
        """Reset the first blocked stage to pending so the loop can retry it."""
        stage = self.first_blocked()
        if stage is None:
            return None
        record = self.stages[stage]
        record.status = "pending"
        record.message = "Resuming from blocked stage."
        record.updated_at = datetime.now(timezone.utc)
        self.stage = stage
        self.current_stage = stage
        self.refresh_summary()
        return stage

    def refresh_summary(self) -> None:
        """Refresh top-level checkpoint lists from detailed stage records."""
        self.completed = _stages_with_status(self.stages, "completed")
        self.pending = _stages_with_status(self.stages, "pending")
        self.failed = _stages_with_status(self.stages, "failed")
        self.skipped = _stages_with_status(self.stages, "skipped")
        self.blocked = [
            _blocked_reason(stage, record.message)
            for stage in DEFAULT_STAGE_ORDER
            if (record := self.stages.get(stage)) is not None and record.status == "blocked"
        ]
        artifacts: dict[str, Path] = {}
        for record in self.stages.values():
            artifacts.update(record.artifacts)
        self.artifacts = artifacts

    def to_yaml(self, path: Path | str) -> Path:
        """Write loop state YAML."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.model_dump(mode="json"), file, sort_keys=False)
        return output_path

    @classmethod
    def from_yaml(cls, path: Path | str) -> "LoopState":
        """Load loop state YAML."""
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8-sig") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Loop state YAML must contain a mapping: {input_path}")
        state = cls.model_validate(data)
        state.refresh_summary()
        return state


def _stages_with_status(stages: dict[LoopStage, LoopStageState], status: StageStatus) -> list[LoopStage]:
    return [
        stage
        for stage in DEFAULT_STAGE_ORDER
        if (record := stages.get(stage)) is not None and record.status == status
    ]


def _blocked_reason(stage: LoopStage, message: str) -> str:
    lowered = message.lower()
    if "detection_errors" in lowered or "detection errors" in lowered:
        return "missing_detection_errors"
    if "dataset_report" in lowered:
        return "missing_dataset_report"
    if "loop_diagnosis" in lowered:
        return "missing_loop_diagnosis"
    if "loop_plan" in lowered:
        return "missing_loop_plan"
    if "policy_evaluation" in lowered:
        return "missing_policy_evaluation"
    if "candidate plan" in lowered:
        return "missing_candidate_plan"
    if "metrics_input_path" in lowered or "metrics" in lowered:
        return "missing_metrics"
    return f"blocked_{stage}"
