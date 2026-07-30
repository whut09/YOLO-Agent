from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.distillation import run_distillation_cpu_fixture
from yolo_agent.components.adapters import AdapterContext
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.contracts import load_contracts


def test_distillation_cpu_fixture_runs_teacher_and_student_paths(
    tmp_path: Path,
) -> None:
    contract = load_contracts(
        "configs/components/distillation/yolo26_teacher_student.yaml"
    )[0]
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    teacher = tmp_path / "yolo26s.pt"
    student = tmp_path / "yolo26n.pt"
    teacher.write_bytes(b"teacher-checkpoint")
    student.write_bytes(b"student-checkpoint")
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=tmp_path,
        options={
            "teacher": str(teacher),
            "student": str(student),
            "teacher_data": "fixture-coco.yaml",
            "student_data": "fixture-coco.yaml",
        },
    )
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="distillation-protocol",
        base_command=[
            "yolo",
            "detect",
            "train",
            f"model={student}",
            "data=fixture-coco.yaml",
            "imgsz=640",
        ],
        generated_config={},
    )
    payload_path = payload.write(tmp_path / "runtime" / "adapter_runtime_payload.yaml")

    report = run_distillation_cpu_fixture(
        runtime_payload_path=payload_path,
        workspace=tmp_path / "golden",
    )

    assert report.status == "passed", report.errors
    assert report.checks["trainer_bridge_called"] is True
    assert report.checks["total_loss_changed"] is True
    assert report.checks["student_backward"] is True
    assert report.checks["teacher_no_grad"] is True
    assert report.checks["zero_weight_native_equivalent"] is True
    assert report.checks["method_profiles_only"] is True
    assert report.checks["exact_reproduction_false"] is True
