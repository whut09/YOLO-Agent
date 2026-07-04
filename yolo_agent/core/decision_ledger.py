"""Decision ledger for policy-to-experiment audit trails."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from yolo_agent.core.artifact_manifest import sha256_directory, sha256_file


DECISION_LEDGER_SCHEMA_VERSION = "1.1"


class DecisionReplaySnapshot(BaseModel):
    """Stable hashes of inputs needed to replay a policy decision."""

    task_spec_sha256: str | None = None
    component_registry_sha256: str | None = None
    loop_plan_sha256: str | None = None
    evidence_gate_sha256: str | None = None
    policy_version: str = "unknown"


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
    budget_bucket: str | None = None
    budget_reason: str = ""
    requires_human_confirmation: bool = False
    created_candidate_id: str | None = None
    created_node_id: str | None = None
    candidate_config: dict[str, Any] | None = None
    experiment_node: dict[str, Any] | None = None
    rationale: str = ""
    task_spec_sha256: str | None = None
    component_registry_sha256: str | None = None
    loop_plan_sha256: str | None = None
    evidence_gate_sha256: str | None = None
    policy_version: str = "unknown"
    replay_snapshot: DecisionReplaySnapshot | None = None
    schema_version: str = DECISION_LEDGER_SCHEMA_VERSION
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


def build_replay_snapshot(
    task_spec_path: Path | str | None = None,
    component_registry_path: Path | str | None = None,
    loop_plan_path: Path | str | None = None,
    evidence_gate: BaseModel | dict[str, Any] | None = None,
    policy_version: str = "unknown",
) -> DecisionReplaySnapshot:
    """Build stable replay hashes for policy-decision inputs."""
    return DecisionReplaySnapshot(
        task_spec_sha256=sha256_path(task_spec_path),
        component_registry_sha256=sha256_path(component_registry_path),
        loop_plan_sha256=sha256_path(loop_plan_path),
        evidence_gate_sha256=sha256_model(evidence_gate) if evidence_gate is not None else None,
        policy_version=policy_version,
    )


def sha256_path(path: Path | str | None) -> str | None:
    """Return a stable hash for a file or directory path."""
    if path is None:
        return None
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if target.is_dir():
        return sha256_directory(target)
    return None


def sha256_model(value: BaseModel | dict[str, Any]) -> str:
    """Return a stable SHA-256 for a Pydantic model or mapping."""
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
