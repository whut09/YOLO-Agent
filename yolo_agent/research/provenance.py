"""Provenance contracts for external research catalog imports."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from yolo_agent.research.schemas import PaperProvenance


ImportProvenance = PaperProvenance


def assert_research_production_allowed() -> None:
    """Prevent import/sync code from running inside the training runtime."""
    phase = os.getenv("YOLO_AGENT_RUNTIME_PHASE", "").strip().lower()
    active = os.getenv("YOLO_AGENT_TRAINING_ACTIVE", "").strip().lower()
    if phase == "training" or active in {"1", "true", "yes"}:
        raise RuntimeError("research import and network sync are disabled during training")


class ProvenanceHistoryEntry(BaseModel):
    """A prior source record retained when an imported paper is updated."""

    source_commit: str = "unknown"
    source_record_hash: str
    imported_at: str
    importer_version: str
    snapshot: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ImportProvenance",
    "PaperProvenance",
    "ProvenanceHistoryEntry",
    "assert_research_production_allowed",
]
