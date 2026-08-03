from pathlib import Path

from yolo_agent.components.adapters import AdapterContext
from yolo_agent.components.adapters.assigners.yolo26_assignment import (
    ASSIGNMENT_SPECS,
    YOLO26AssignmentAdapter,
)
from yolo_agent.components.adapters.audit_contract import (
    EXPECTED_RUNTIME_ADAPTERS,
    validate_audited_runtime_payload,
)
from yolo_agent.components.contracts import load_contracts


def test_all_assignment_payloads_match_the_runtime_audit_contract(
    tmp_path: Path,
) -> None:
    contracts = {
        item.component_id: item
        for item in load_contracts(
            "configs/components/assigner/yolo26_assignment.yaml"
        )
    }

    assert set(ASSIGNMENT_SPECS).issubset(EXPECTED_RUNTIME_ADAPTERS)
    for component_id, spec in ASSIGNMENT_SPECS.items():
        context = AdapterContext(
            contract=contracts[component_id],
            detector_family="yolo26",
            head="one_to_one",
            imgsz=640,
            workspace=tmp_path / component_id,
        )
        payload = YOLO26AssignmentAdapter().build_runtime_payload(
            context,
            protocol_hash=f"protocol-{component_id}",
            base_command=["yolo", "detect", "train", "imgsz=640"],
            generated_config={"imgsz": 640},
        )

        checks = validate_audited_runtime_payload(payload, component_id)

        assert checks["audited_runtime_component"] is True
        assert checks["audited_plugin_kind"] == "assigner_plugin"
        assert checks["audited_changed_variable"] == spec.changed_variable
