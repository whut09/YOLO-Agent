"""Decision ledger for policy-to-experiment audit trails."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DecisionLedgerRecord(BaseModel):
    """One auditable decision from proposal evaluation."""

    decision_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    policy_id: str
    proposal: dict[str, Any] = Field(default_factory=dict)
    decision: str
    priority: float = 0.0
    blocked_by: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    deployment_constraints: list[dict[str, Any]] = Field(default_factory=list)
    compatibility_warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    created_candidate_id: str | None = None
    created_node_id: str | None = None
    candidate_config: dict[str, Any] | None = None
    experiment_node: dict[str, Any] | None = None
    rationale: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionLedger:
    """Append-only JSONL decision ledger."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, record: DecisionLedgerRecord) -> DecisionLedgerRecord:
        """Append one decision record."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        return record

    def write(self, records: list[DecisionLedgerRecord]) -> Path:
        """Replace the ledger with a batch of decision records."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        return self.path

    def read(self) -> list[DecisionLedgerRecord]:
        """Read all decision records."""
        if not self.path.exists():
            return []
        records: list[DecisionLedgerRecord] = []
        with self.path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                text = line.strip()
                if text:
                    records.append(DecisionLedgerRecord.model_validate(json.loads(text)))
        return records
