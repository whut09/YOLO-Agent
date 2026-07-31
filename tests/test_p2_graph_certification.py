from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.p2_graph import run_p2_graph_cpu_fixture
from yolo_agent.components.adapters import AdapterContext
from yolo_agent.components.adapters.head.p2_head import P2HeadAdapter, P2HeadManifest
from yolo_agent.components.contracts import load_contracts


def test_p2_graph_cpu_fixture_certifies_runtime_graph(tmp_path: Path) -> None:
    contract = load_contracts(
        "configs/components/head/yolo26_p2_small_object.yaml"
    )[0]
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=tmp_path / "runtime",
        options={
            "audit_imgsz": 64,
            "latency_warmup": 0,
            "latency_iterations": 1,
        },
    )
    payload = P2HeadAdapter().build_runtime_payload(
        context,
        protocol_hash="p2-certification-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    payload_path = payload.write(tmp_path / "runtime" / "adapter_runtime_payload.yaml")

    report = run_p2_graph_cpu_fixture(
        runtime_payload_path=payload_path,
        workspace=tmp_path / "golden",
    )

    assert report.status == "passed", report.errors
    assert report.checks["real_forward"] is True
    assert report.checks["native_loss_preserved"] is True
    assert report.checks["backward"] is True
    assert report.checks["amp"] is True
    assert report.checks["partial_checkpoint_audit"] is True
    assert report.checks["export"] is True
    assert report.checks["resource_guard"] is True
    assert report.checks["matched_control_required"] is True
    manifest = P2HeadManifest.model_validate_json(
        (tmp_path / "runtime" / "p2_head_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.graph_integrated is True
    assert manifest.detection_head_integrated is True
    assert manifest.native_loss_integrated is True
    assert manifest.checkpoint_integrated is True
    assert manifest.runtime_payload_hash == payload.payload_hash
    assert manifest.changed_variables["model.p2_head"]
