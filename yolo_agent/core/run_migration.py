"""Audit and quarantine runs created before the current protocol contract."""

from __future__ import annotations

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
    "assess_run_protocol",
    "write_migration_report",
]
