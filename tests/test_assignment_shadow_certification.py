from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from yolo_agent.certification.assignment_shadow import (
    run_assignment_shadow_cpu_fixture,
)
from yolo_agent.components.adapters import AdapterContext
from yolo_agent.components.adapters.assigners.yolo26_assignment import (
    ASSIGNMENT_SPECS,
    YOLO26AssignmentAdapter,
)
from yolo_agent.components.contracts import load_contracts


@pytest.mark.parametrize(
    "component_id",
    sorted(ASSIGNMENT_SPECS),
)
def test_assignment_shadow_cpu_fixture_certifies_native_equivalence(
    component_id: str,
    tmp_path: Path,
) -> None:
    contract = next(
        item
        for item in load_contracts("configs/components/assigner/yolo26_assignment.yaml")
        if item.component_id == component_id
    )
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=tmp_path / component_id,
        options={
            "assignment.minimum_shadow_batches": 1,
            "assignment.maximum_conflict_rate": 1.0,
            "assignment.evidence_interval": 1,
        },
    )
    payload = YOLO26AssignmentAdapter().build_runtime_payload(
        context,
        protocol_hash=f"{component_id}-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    payload_path = payload.write(
        tmp_path / component_id / "adapter_runtime_payload.yaml"
    )

    report = run_assignment_shadow_cpu_fixture(
        runtime_payload_path=payload_path,
        workspace=tmp_path / component_id / "golden",
    )

    assert report.status == "passed", report.errors
    assert report.method == ASSIGNMENT_SPECS[component_id].method
    assert report.checks["native_loss_equivalent"] is True
    assert report.checks["native_one_to_one_preserved"] is True
    assert report.checks["positive_ratio_recorded"] is True
    assert report.checks["conflict_rate_recorded"] is True
    assert report.checks["matching_stability_recorded"] is True
    assert report.checks["per_path_metrics_recorded"] is True
    assert report.checks["matched_control_required"] is True
    assert report.checks["active_pilot_blocked_until_explicit_gate"] is True
    if component_id == "assigner.dual_path":
        assert "one_to_many.matching_stability" in report.metrics
        assert "one_to_one.matching_stability" in report.metrics


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("YOLO_AGENT_RUN_GPU_TESTS") != "1",
    reason="set YOLO_AGENT_RUN_GPU_TESTS=1 for optional assignment GPU smoke",
)
@pytest.mark.parametrize("component_id", sorted(ASSIGNMENT_SPECS))
def test_optional_assignment_shadow_gpu_smoke(
    component_id: str,
    tmp_path: Path,
) -> None:
    contract = next(
        item
        for item in load_contracts("configs/components/assigner/yolo26_assignment.yaml")
        if item.component_id == component_id
    )
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=tmp_path / component_id,
    )

    result = YOLO26AssignmentAdapter().gpu_smoke_test(context)

    assert result.passed, result.errors
    assert result.checks["shadow_mode_only"] is True
    assert result.checks["native_loss_equivalent"] is True
    assert result.checks["native_one_to_one_preserved"] is True
    assert result.checks["native_audit_verified"] is True
    assert result.checks["positive_ratio_recorded"] is True
    assert result.checks["conflict_rate_recorded"] is True
    assert result.checks["shadow_artifact_passed"] is True
