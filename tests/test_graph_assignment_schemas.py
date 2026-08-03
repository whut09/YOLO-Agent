from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.graph_assignment_schemas import (
    AssignmentShadowCpuReport,
    GraphCpuReport,
)


def test_graph_report_requires_all_runtime_contract_checks(tmp_path: Path) -> None:
    checks = {
        key: True
        for key in (
            "atomic_recipe_verified",
            "real_forward",
            "native_loss_preserved",
            "backward",
            "amp",
            "partial_checkpoint_audit",
            "export",
            "resource_guard",
            "matched_control_required",
        )
    }
    report = GraphCpuReport(
        component_id="head.p2_small_object",
        recipe_id="yolo26_small_object_p2",
        status="passed",
        protocol_hash="p2-protocol",
        runtime_payload_hash="a" * 64,
        runtime_payload_path=tmp_path / "runtime.yaml",
        manifest_path=tmp_path / "manifest.json",
        checkpoint_audit_path=tmp_path / "checkpoint.json",
        checks=checks,
    )
    assert len(report.report_hash) == 64


def test_assignment_report_rejects_missing_shadow_metrics(tmp_path: Path) -> None:
    checks = {
        key: True
        for key in (
            "atomic_recipe_verified",
            "shadow_mode_only",
            "native_audit_verified",
            "positive_ratio_recorded",
            "conflict_rate_recorded",
            "matching_stability_recorded",
            "per_path_metrics_recorded",
            "native_loss_equivalent",
            "native_one_to_one_preserved",
            "matched_control_required",
            "active_pilot_blocked_until_explicit_gate",
        )
    }
    with pytest.raises(ValueError, match="requires metric"):
        AssignmentShadowCpuReport(
            component_id="assigner.task_aligned",
            method="tood_tal",
            recipe_id="yolo26_tood_tal_assignment_shadow",
            status="passed",
            protocol_hash="assigner-protocol",
            runtime_payload_hash="a" * 64,
            runtime_payload_path=tmp_path / "runtime.yaml",
            shadow_evidence_path=tmp_path / "shadow.json",
            checks=checks,
        )
