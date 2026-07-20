"""Provenance contracts for external research catalog imports."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from yolo_agent.research.schemas import PaperProvenance


ImportProvenance = PaperProvenance


class ProvenanceHistoryEntry(BaseModel):
    """A prior source record retained when an imported paper is updated."""

    source_commit: str = "unknown"
    source_record_hash: str
    imported_at: str
    importer_version: str
    snapshot: dict[str, Any] = Field(default_factory=dict)


__all__ = ["ImportProvenance", "PaperProvenance", "ProvenanceHistoryEntry"]
