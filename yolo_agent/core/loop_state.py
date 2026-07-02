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
    current_stage: LoopStage = "init"
    stages: dict[LoopStage, LoopStageState] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(cls, run_id: str, stage_order: list[LoopStage] | None = None) -> "LoopState":
        """Create a pending state machine."""
        order = stage_order or DEFAULT_STAGE_ORDER
        return cls(
            run_id=run_id,
            current_stage=order[0],
            stages={stage: LoopStageState(stage=stage) for stage in order},
        )

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
        now = datetime.now(timezone.utc)
        record.updated_at = now
        self.updated_at = now
        self.current_stage = stage

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
        return cls.model_validate(data)
