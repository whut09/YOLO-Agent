"""Shared loop harness types."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from yolo_agent.core.loop_state import LoopStage, StageStatus


class StageResult(BaseModel):
    """Result of one orchestrated stage."""

    stage: LoopStage
    status: StageStatus
    message: str = ""
    artifacts: dict[str, Path] = Field(default_factory=dict)
