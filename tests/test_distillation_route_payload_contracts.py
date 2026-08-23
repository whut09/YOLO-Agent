from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    YOLO26DistillationAdapter,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.distillation import DISTILLATION_COMPONENTS


@pytest.mark.parametrize("component_id", sorted(DISTILLATION_COMPONENTS))
def test_each_distillation_component_emits_its_own_route_payload(
    component_id: str,
    tmp_path: Path,
) -> None:
    spec = DISTILLATION_COMPONENTS[component_id]
    teacher = tmp_path / "yolo26s.pt"
    teacher_m = tmp_path / "yolo26m.pt"
    student = tmp_path / "yolo26n.pt"
    teacher.write_bytes(b"teacher")
    teacher_m.write_bytes(b"teacher-m")
    student.write_bytes(b"student")
    contract = ComponentContract(
        component_id=component_id,
        display_name=component_id,
        category="distillation",
        implementation_path="yolo_agent.components.adapters.distillation.yolo26_distillation",
        adapter_class="YOLO26DistillationAdapter",
        maturity="adapter_implemented",
        fixed_imgsz_compatible=True,
    )
    payload = YOLO26DistillationAdapter().build_runtime_payload(
        AdapterContext(
            contract=contract,
            detector_family="yolo26",
            imgsz=640,
            workspace=tmp_path,
            options={
                "teacher": str(teacher),
                "student": str(student),
                "teacher_data": "coco.yaml",
                "student_data": "coco.yaml",
                "teachers": [str(teacher_m)] if component_id == "distillation.teacher_ensemble" else [],
            },
        ),
        protocol_hash="route-protocol",
        base_command=[
            "yolo",
            "detect",
            "train",
            f"model={student}",
            "data=coco.yaml",
            "imgsz=640",
        ],
        generated_config={},
    )
    options = payload.loss_plugin[0].options
    assert options["mechanism"] == spec.mechanism
    assert options["component_id"] == component_id
    assert options["changed_variable"] == spec.changed_variable
    assert options["branch_id"]
    assert options["teacher"] == str(teacher.resolve())
    assert options["student"] == str(student.resolve())
    assert payload.changed_variables == {spec.changed_variable: 1.0}
