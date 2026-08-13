"""Audit and quarantine runs created before the current protocol contract."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.asha_scheduler import ASHAStudy
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentPlan, MetricEvidence
from yolo_agent.core.round_execution_plan import RoundExecutionPlan
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.run_protocol import RunProtocolVersion
from yolo_agent.core.yaml_io import YAMLModelMixin


MigrationAction = Literal["continue_current_run", "start_new_run"]


class LegacyRunAssessment(BaseModel):
    run_id: str
    legacy_run: bool
    reasons: list[str] = Field(default_factory=list)
    run_protocol_hash: str | None = None
    latest_trusted_node_id: str | None = None
    latest_trusted_candidate_id: str | None = None
    trusted_metric_count: int = 0


class RunMigrationReport(BaseModel, YAMLModelMixin):
    schema_version: str = "run_migration.v1"
    run_id: str
    legacy_run: bool
    reasons: list[str] = Field(default_factory=list)
    action: MigrationAction
    suggested_run_id: str | None = None
    latest_trusted_node_id: str | None = None
    latest_trusted_candidate_id: str | None = None
    trusted_metric_count: int = 0
    evidence_policy: str = "legacy candidate metrics are inherited_context only"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunProtocolRecoveryReport(BaseModel, YAMLModelMixin):
    """Audit record for restoring a base protocol overwritten during resume."""

    schema_version: str = "run_protocol_recovery.v1"
    run_id: str
    recovered: bool
    reason: str
    overwritten_protocol_hash: str
    recovered_protocol_hash: str
    evidence_path: Path
    evidence_code_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def recover_overwritten_run_protocol(context: RunContext) -> RunProtocolRecoveryReport | None:
    """Restore an immutable base protocol from a completed execution artifact.

    A previous resume path rebuilt ``run_protocol.yaml`` with the current code
    commit while intentionally preserving a non-empty ASHA study.  That left
    the two protocol hashes inconsistent.  Recovery is allowed only when a
    completed execution artifact proves that changing *only* ``code_version``
    reconstructs the exact ASHA protocol hash.
    """
    protocol_path = context.run_protocol_path or context.artifact_path("run_protocol.yaml")
    configured_asha_path = str(context.metadata.get("asha_state_path") or "")
    asha_path = Path(configured_asha_path) if configured_asha_path else context.artifact_path("asha_state.yaml")
    if not protocol_path.is_file() or not asha_path.is_file():
        return None
    try:
        overwritten = RunProtocolVersion.from_yaml(protocol_path)
        asha = ASHAStudy.from_yaml(asha_path)
    except (OSError, ValueError):
        return None
    if not asha.run_protocol_hash or asha.run_protocol_hash == overwritten.protocol_hash:
        return None
    execution_dir = context.artifact_path("execution_results")
    if not execution_dir.is_dir():
        return None
    for evidence_path in sorted(execution_dir.glob("*.json")):
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or str(payload.get("status") or "") != "completed":
            continue
        command = payload.get("command")
        metadata = command.get("metadata") if isinstance(command, dict) else None
        if not isinstance(metadata, dict):
            continue
        evidence_hash = str(metadata.get("run_protocol_hash") or "")
        code_version = str(metadata.get("code_version") or "")
        if evidence_hash != asha.run_protocol_hash or not code_version:
            continue
        candidate_payload = overwritten.model_dump(mode="python")
        candidate_payload.update(code_version=code_version, protocol_hash="")
        try:
            recovered = RunProtocolVersion.model_validate(candidate_payload)
        except ValueError:
            continue
        if recovered.protocol_hash != asha.run_protocol_hash:
            continue
        archive_path = context.artifact_path(
            f"run_protocol.overwritten.{overwritten.protocol_hash[:12]}.yaml"
        )
        if not archive_path.is_file():
            overwritten.to_yaml(archive_path)
        recovered.to_yaml(protocol_path)
        context.run_protocol_path = protocol_path
        context.run_protocol_hash = recovered.protocol_hash
        context.legacy_run = False
        context.metadata.update(
            {
                "run_protocol_hash": recovered.protocol_hash,
                "code_version": recovered.code_version,
                "legacy_run_reasons": [],
                "migration_suggested_run_id": None,
            }
        )
        report = RunProtocolRecoveryReport(
            run_id=context.run_id,
            recovered=True,
            reason="resume_overwrote_code_version_only",
            overwritten_protocol_hash=overwritten.protocol_hash,
            recovered_protocol_hash=recovered.protocol_hash,
            evidence_path=evidence_path.resolve(),
            evidence_code_version=code_version,
        )
        report_path = context.artifact_path("run_protocol_recovery.yaml")
        report.to_yaml(report_path)
        context.metadata["run_protocol_recovery_path"] = report_path.resolve().as_posix()
        context.to_yaml()
        context.to_json()
        return report
    return None


def assess_run_protocol(context: RunContext, evidence_store: EvidenceStore) -> LegacyRunAssessment:
    """Return whether an existing run satisfies the current durable protocol contract."""
    reasons: list[str] = []
    protocol_path = context.run_protocol_path or context.artifact_path("run_protocol.yaml")
    protocol: RunProtocolVersion | None = None
    if not protocol_path.is_file():
        reasons.append("missing_run_protocol")
    else:
        try:
            protocol = RunProtocolVersion.from_yaml(protocol_path)
        except (OSError, ValueError):
            reasons.append("invalid_run_protocol")
    if protocol is not None:
        if not context.run_protocol_hash:
            reasons.append("missing_context_protocol_hash")
        elif context.run_protocol_hash != protocol.protocol_hash:
            reasons.append("context_protocol_hash_mismatch")
        if not protocol.eval_protocol_hash:
            reasons.append("missing_post_eval_protocol")
        context_eval_hash = str(context.metadata.get("post_eval_protocol_hash") or "")
        if not context_eval_hash:
            reasons.append("missing_context_post_eval_protocol")
        elif context_eval_hash != protocol.eval_protocol_hash:
            reasons.append("context_post_eval_protocol_mismatch")

    objective_hash = str(context.metadata.get("optimization_objective_hash") or "")
    objective_path = Path(str(context.metadata.get("optimization_objective_path") or ""))
    if not objective_hash:
        reasons.append("missing_objective_hash")
    if not objective_path.is_file():
        reasons.append("missing_optimization_objective")

    configured_asha_path = str(context.metadata.get("asha_state_path") or "")
    asha_path = Path(configured_asha_path) if configured_asha_path else context.artifact_path("asha_state.yaml")
    if not asha_path.is_file():
        reasons.append("missing_asha_state")
    else:
        try:
            asha = ASHAStudy.from_yaml(asha_path)
            if not asha.run_protocol_hash:
                reasons.append("missing_asha_protocol_hash")
            elif protocol is not None and asha.run_protocol_hash != protocol.protocol_hash:
                reasons.append("asha_protocol_hash_mismatch")
        except (OSError, ValueError):
            reasons.append("invalid_asha_state")

    plan_path = context.artifact_path("experiment_plan.yaml")
    if plan_path.is_file():
        try:
            plan = ExperimentPlan.from_yaml(plan_path)
            if not plan.run_protocol_hash:
                reasons.append("missing_experiment_plan_protocol_hash")
        except (OSError, ValueError):
            reasons.append("invalid_experiment_plan")
    round_path = context.artifact_path("round_execution_plan.yaml")
    if round_path.is_file():
        try:
            round_plan = RoundExecutionPlan.from_yaml(round_path)
            if not round_plan.run_protocol_hash:
                reasons.append("missing_round_plan_protocol_hash")
        except (OSError, ValueError):
            reasons.append("invalid_round_execution_plan")

    evidence = evidence_store.load_run(context.run_id)
    trusted = [record for record in evidence.metric_records if _trusted_current_record(record, context.run_id)]
    latest = max(trusted, key=lambda item: item.created_at) if trusted else None
    return LegacyRunAssessment(
        run_id=context.run_id,
        legacy_run=bool(reasons),
        reasons=list(dict.fromkeys(reasons)),
        run_protocol_hash=protocol.protocol_hash if protocol is not None else None,
        latest_trusted_node_id=latest.node_id if latest is not None else None,
        latest_trusted_candidate_id=latest.candidate_id if latest is not None else None,
        trusted_metric_count=len(trusted),
    )


def write_migration_report(context: RunContext, assessment: LegacyRunAssessment) -> RunMigrationReport:
    """Persist legacy status without deleting or rewriting existing evidence."""
    suggested = _available_migrated_run_id(context) if assessment.legacy_run else None
    report = RunMigrationReport(
        run_id=context.run_id,
        legacy_run=assessment.legacy_run,
        reasons=assessment.reasons,
        action="start_new_run" if assessment.legacy_run else "continue_current_run",
        suggested_run_id=suggested,
        latest_trusted_node_id=assessment.latest_trusted_node_id,
        latest_trusted_candidate_id=assessment.latest_trusted_candidate_id,
        trusted_metric_count=assessment.trusted_metric_count,
    )
    path = context.artifact_path("run_migration_report.yaml")
    report.to_yaml(path)
    context.legacy_run = assessment.legacy_run
    context.metadata["run_migration_report_path"] = path.as_posix()
    context.metadata["legacy_run_reasons"] = assessment.reasons
    context.metadata["migration_suggested_run_id"] = suggested
    context.to_yaml()
    context.to_json()
    return report


def _trusted_current_record(record: MetricEvidence, run_id: str) -> bool:
    return bool(
        record.verified
        and record.evidence_role == "current_observation"
        and record.inheritance_depth == 0
        and (record.origin_run_id or record.run_id) == run_id
        and not record.source.startswith(("inherited:", "legacy:"))
        and record.protocol_hash
        and record.dataset_manifest_sha256
        and record.subset_manifest_sha256
        and record.eval_protocol_hash
        and record.seed is not None
        and record.epochs is not None
        and record.batch_policy_hash
        and record.ultralytics_version
        and record.imgsz is not None
    )


def _available_migrated_run_id(context: RunContext) -> str:
    base = f"{context.run_id}-v2"
    candidate = base
    index = 2
    while (context.run_root / candidate).exists():
        candidate = f"{base}-{index}"
        index += 1
    return candidate


__all__ = [
    "LegacyRunAssessment",
    "RunMigrationReport",
    "RunProtocolRecoveryReport",
    "assess_run_protocol",
    "recover_overwritten_run_protocol",
    "write_migration_report",
]
