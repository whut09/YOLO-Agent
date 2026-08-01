"""Candidate/control COCO evidence contract for paper acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.certification.runner import BackendEvaluation, BackendRun
from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.pilot_evidence import (
    PilotEvidenceCompletenessGate,
    validate_coco_evidence_artifacts,
)
from yolo_agent.core.yaml_io import YAMLModelMixin


class PaperPilotEvidenceBundle(BaseModel, YAMLModelMixin):
    """Artifact-backed completeness result for one matched pilot pair."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_pilot_evidence_bundle.v1"
    stage_id: str
    run_id: str
    protocol_hash: str
    control_node_id: str
    candidate_node_id: str
    complete: bool = False
    control_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    candidate_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    candidate_fact_count: int = Field(default=0, ge=0)
    baseline_fact_count: int = Field(default=0, ge=0)
    missing_evidence: list[str] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    bundle_hash: str = ""

    @model_validator(mode="after")
    def validate_bundle_hash(self) -> "PaperPilotEvidenceBundle":
        expected = self.calculate_hash()
        if self.bundle_hash and self.bundle_hash != expected:
            raise ValueError("paper pilot evidence bundle hash mismatch")
        self.bundle_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"bundle_hash"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def validate_paper_pilot_evidence(
    *,
    store: EvidenceStore,
    run_id: str,
    stage_id: str,
    protocol_hash: str,
    control_run: BackendRun,
    control_evaluation: BackendEvaluation,
    candidate_run: BackendRun,
    candidate_evaluation: BackendEvaluation,
    output_path: Path | str,
) -> PaperPilotEvidenceBundle:
    """Require complete current-node evidence before ASHA can observe a pilot."""
    control_contract = validate_coco_evidence_artifacts(
        predictions_path=control_evaluation.predictions_path,
        eval_path=control_evaluation.eval_path,
        error_report_path=control_evaluation.error_report_path,
    )
    candidate_contract = validate_coco_evidence_artifacts(
        predictions_path=candidate_evaluation.predictions_path,
        eval_path=candidate_evaluation.eval_path,
        error_report_path=candidate_evaluation.error_report_path,
    )
    candidate_gate = PilotEvidenceCompletenessGate(store).evaluate(
        run_id=run_id,
        candidate_id=candidate_run.candidate_id,
        node_id=candidate_run.node_id,
        protocol_hash=protocol_hash,
    )
    facts = ErrorFactStore(store.root).read(run_id)
    candidate_facts = [
        item
        for item in facts
        if item.run_id == run_id
        and (item.origin_run_id or item.run_id) == run_id
        and item.inheritance_depth == 0
        and item.candidate_id == candidate_run.candidate_id
        and item.node_id == candidate_run.node_id
        and item.protocol_hash == protocol_hash
        and item.evidence_role == "current_observation"
    ]
    baseline_facts = [
        item
        for item in facts
        if item.run_id == run_id
        and (item.origin_run_id or item.run_id) == run_id
        and item.inheritance_depth == 0
        and item.candidate_id == control_run.candidate_id
        and item.node_id == control_run.node_id
        and item.protocol_hash == protocol_hash
        and item.evidence_role == "baseline_reference"
    ]
    missing: list[str] = []
    if not control_contract.valid:
        missing.append("control_coco_artifact_contract")
    if not candidate_contract.valid:
        missing.append("candidate_coco_artifact_contract")
    missing.extend(f"candidate:{item}" for item in candidate_gate.missing_metrics)
    missing.extend(f"candidate:{item}" for item in candidate_gate.missing_artifacts)
    missing.extend(
        f"candidate_fact_group:{item}" for item in candidate_gate.missing_fact_groups
    )
    if not baseline_facts:
        missing.append("matched_baseline_error_facts")
    if not candidate_facts:
        missing.append("current_candidate_error_facts")
    recovery = list(candidate_gate.evidence_actions)
    if not control_contract.valid:
        recovery.append("recover_control_coco_post_eval")
    if not candidate_contract.valid:
        recovery.append("recover_candidate_coco_post_eval")
    if not baseline_facts:
        recovery.append("import_matched_baseline_error_facts")
    if not candidate_facts:
        recovery.append("import_current_candidate_error_facts")
    bundle = PaperPilotEvidenceBundle(
        stage_id=stage_id,
        run_id=run_id,
        protocol_hash=protocol_hash,
        control_node_id=control_run.node_id,
        candidate_node_id=candidate_run.node_id,
        complete=not missing and candidate_gate.complete,
        control_artifact_hashes=control_contract.artifact_hashes,
        candidate_artifact_hashes=candidate_contract.artifact_hashes,
        candidate_fact_count=len(candidate_facts),
        baseline_fact_count=len(baseline_facts),
        missing_evidence=list(dict.fromkeys(missing)),
        recovery_actions=list(dict.fromkeys(recovery)),
    )
    bundle.to_yaml(output_path, exclude_none=True, sort_keys=False)
    return bundle


__all__ = ["PaperPilotEvidenceBundle", "validate_paper_pilot_evidence"]
