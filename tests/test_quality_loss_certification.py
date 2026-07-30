from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.quality_loss import run_quality_loss_cpu_fixture
from yolo_agent.components.adapters import AdapterContext
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.contracts import load_contracts


@pytest.mark.parametrize(
    "component_id",
    [
        "loss.quality.correlation",
        "loss.calibration.bpc",
        "loss.quality.pseudo_iou",
    ],
)
def test_quality_loss_cpu_fixture_runs_real_trainer_bridge(
    component_id: str,
    tmp_path: Path,
) -> None:
    contract = next(
        item
        for item in load_contracts("configs/components/loss/quality_alignment.yaml")
        if item.component_id == component_id
    )
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=tmp_path,
    )
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="quality-loss-protocol",
        base_command=[
            "yolo",
            "detect",
            "train",
            "model=yolo26n.pt",
            "data=coco.yaml",
            "imgsz=640",
        ],
        generated_config={},
    )
    payload_path = payload.write(tmp_path / component_id / "adapter_runtime_payload.yaml")

    report = run_quality_loss_cpu_fixture(
        runtime_payload_path=payload_path,
        workspace=tmp_path / component_id / "golden",
    )

    assert report.status == "passed", report.errors
    assert report.component_id == component_id
    assert report.checks["trainer_bridge_called"] is True
    assert report.checks["total_loss_changed"] is True
    assert report.checks["student_backward"] is True
    assert report.checks["zero_weight_native_equivalent"] is True
    assert report.checks["exact_reproduction_false"] is True
