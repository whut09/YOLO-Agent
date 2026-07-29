"""Matched-control availability gate for materialized paper candidates."""

from __future__ import annotations

from pydantic import BaseModel, Field

from yolo_agent.core.experiment_graph import ExperimentNode


class MatchedControlAssessment(BaseModel):
    available: bool
    protocol_hash: str | None = None
    reasons: list[str] = Field(default_factory=list)


def assess_matched_control(
    candidate: ExperimentNode,
    control: ExperimentNode | None,
    *,
    required_protocol_hash: str,
) -> MatchedControlAssessment:
    if control is None:
        return MatchedControlAssessment(
            available=False,
            reasons=["matched_control_missing"],
        )
    candidate_command = candidate.command_spec
    control_command = control.command_spec
    if candidate_command is None or control_command is None:
        return MatchedControlAssessment(
            available=False,
            reasons=["matched_control_command_spec_missing"],
        )
    reasons: list[str] = []
    candidate_protocol = _protocol_hash(candidate)
    control_protocol = _protocol_hash(control)
    if candidate_protocol != required_protocol_hash:
        reasons.append("candidate_protocol_mismatch")
    if control_protocol != required_protocol_hash:
        reasons.append("matched_control_protocol_mismatch")
    if not bool(control_command.metadata.get("matched_baseline_control")):
        reasons.append("matched_control_role_missing")
    if not _fixed_640(candidate) or not _fixed_640(control):
        reasons.append("matched_control_imgsz_640_required")
    for key in (
        "dataset_manifest_sha256",
        "subset_manifest_sha256",
        "batch_policy_hash",
        "eval_protocol_hash",
        "ultralytics_version",
        "epochs",
        "seed",
    ):
        candidate_value = candidate_command.metadata.get(key)
        control_value = control_command.metadata.get(key)
        if candidate_value is not None and control_value is not None and candidate_value != control_value:
            reasons.append(f"matched_control_identity_mismatch:{key}")
    return MatchedControlAssessment(
        available=not reasons,
        protocol_hash=control_protocol,
        reasons=reasons,
    )


def _protocol_hash(node: ExperimentNode) -> str:
    metadata = node.command_spec.metadata if node.command_spec is not None else {}
    return str(
        metadata.get("run_protocol_hash")
        or metadata.get("protocol_hash")
        or metadata.get("baseline_protocol_hash")
        or ""
    )


def _fixed_640(node: ExperimentNode) -> bool:
    if node.command_spec is None:
        return False
    values = [
        item.split("=", 1)[1]
        for item in node.command_spec.argv
        if item.startswith("imgsz=")
    ]
    return bool(values) and all(value == "640" for value in values)


__all__ = ["MatchedControlAssessment", "assess_matched_control"]
