"""Append-only event log for loop harness runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.loop_state import LoopStage, StageStatus


EventType = Literal[
    "run_initialized",
    "resume_requested",
    "queue_enqueued",
    "queue_refreshed",
    "queue_item_started",
    "queue_item_completed",
    "queue_item_failed",
    "queue_item_resource_blocked",
    "queue_item_skipped",
    "round_plan_reconciled",
    "executor_started",
    "executor_log",
    "executor_metric",
    "executor_completed",
    "executor_failed",
    "executor_timeout",
    "auto_round_started",
    "auto_round_decision",
    "auto_round_completed",
    "auto_round_blocked",
    "active_learning_mined",
    "stage_started",
    "stage_completed",
    "stage_blocked",
    "stage_failed",
    "stage_skipped",
    "contract_blocked",
    "component_maturity_changed",
    "reproduction_state_transition",
    "reproduction_state_failed",
]


class EventLogEntry(BaseModel):
    """One append-only loop event."""

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    event_type: EventType
    stage: LoopStage | None = None
    status: StageStatus | None = None
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Path] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_serializer("artifacts")
    def serialize_artifacts(self, value: dict[str, Path]) -> dict[str, str]:
        """Serialize artifact paths portably."""
        return {key: path.as_posix() for key, path in value.items()}


class EventLog:
    """JSONL event log stored under runs/{run_id}/events.jsonl."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(
        self,
        run_id: str,
        event_type: EventType,
        stage: LoopStage | None = None,
        status: StageStatus | None = None,
        message: str = "",
        details: dict[str, Any] | None = None,
        artifacts: dict[str, Path] | None = None,
    ) -> EventLogEntry:
        """Append one event and return the persisted entry."""
        entry = EventLogEntry(
            run_id=run_id,
            event_type=event_type,
            stage=stage,
            status=status,
            message=message,
            details=details or {},
            artifacts=artifacts or {},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry.model_dump(mode="json"), sort_keys=True) + "\n")
        return entry

    def read(self) -> list[EventLogEntry]:
        """Read all persisted events."""
        if not self.path.exists():
            return []
        entries: list[EventLogEntry] = []
        with self.path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                text = line.strip()
                if text:
                    entries.append(EventLogEntry.model_validate(json.loads(text)))
        return entries
