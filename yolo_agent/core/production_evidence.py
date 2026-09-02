"""Recovery plans for production evidence that is not yet available.

Recovery plans contain instructions and provenance only.  They never create
metrics, manifests, checkpoints, or baseline results on behalf of a run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


RecoveryAction = Literal[
    "recover_train_hard_negative_evidence",
    "generate_matched_baseline_control",
    "reconcile_protocol_hash",
    "reconcile_dataset_manifest_hash",
]


class EvidenceRecoveryPlan(BaseModel):
    """An auditable action list for one blocked production evidence path."""

    schema_version: str = "evidence_recovery_plan.v1"
    run_id: str
    candidate_id: str
    node_id: str
    actions: list[RecoveryAction] = Field(min_length=1)
    blockers: list[str] = Field(min_length=1)
    dataset_manifest_hash: str | None = None
    protocol_hash: str | None = None
    source_artifact: str | None = None
    plan_hash: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_plan(self) -> "EvidenceRecoveryPlan":
        if not self.run_id.strip() or not self.candidate_id.strip() or not self.node_id.strip():
            raise ValueError("evidence recovery plan requires run, candidate, and node IDs")
        expected = self.compute_hash()
        if self.plan_hash and self.plan_hash != expected:
            raise ValueError("evidence recovery plan hash mismatch")
        self.plan_hash = expected
        return self

    def compute_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"plan_hash", "created_at"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def write(self, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(output)
        return output


def write_evidence_recovery_plan(
    *,
    output_path: Path | str,
    run_id: str,
    candidate_id: str,
    node_id: str,
    actions: list[RecoveryAction],
    blockers: list[str],
    dataset_manifest_hash: str | None = None,
    protocol_hash: str | None = None,
    source_artifact: Path | str | None = None,
) -> EvidenceRecoveryPlan:
    """Write a recovery plan without materializing any claimed evidence."""
    plan = EvidenceRecoveryPlan(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        actions=list(dict.fromkeys(actions)),
        blockers=list(dict.fromkeys(blockers)),
        dataset_manifest_hash=dataset_manifest_hash,
        protocol_hash=protocol_hash,
        source_artifact=str(Path(source_artifact).resolve())
        if source_artifact is not None
        else None,
    )
    plan.write(output_path)
    return plan


__all__ = [
    "EvidenceRecoveryPlan",
    "RecoveryAction",
    "write_evidence_recovery_plan",
]
