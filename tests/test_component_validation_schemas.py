from __future__ import annotations

from pathlib import Path

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.validation_schemas import (
    ComponentValidationResult,
    ComponentValidationStageReport,
)


def _contract() -> ComponentContract:
    return ComponentContract(
        component_id="sampling.test",
        display_name="Test sampler",
        category="sampling",
        implementation_path="tests.fixtures",
        adapter_class="TestAdapter",
        maturity="adapter_implemented",
    )


def test_validation_stage_report_round_trips_yaml(tmp_path: Path) -> None:
    report = ComponentValidationStageReport(
        component_id="sampling.test",
        stage="runtime_integrated",
        status="passed",
        protocol_hash="protocol-1",
        validation_key="a" * 64,
        checks={"payload_importable": True},
        artifacts={"runtime_payload": tmp_path / "runtime.yaml"},
    )

    path = report.to_yaml(tmp_path / "runtime_report.yaml")

    assert ComponentValidationStageReport.from_yaml(path) == report


def test_validation_result_preserves_updated_contract(tmp_path: Path) -> None:
    result = ComponentValidationResult(
        status="completed",
        component_id="sampling.test",
        initial_maturity="adapter_implemented",
        final_maturity="runtime_integrated",
        target_maturity="runtime_integrated",
        protocol_hash="protocol-1",
        validation_key="b" * 64,
        contract=_contract().model_copy(update={"maturity": "runtime_integrated"}),
        stage_reports={"runtime_integrated": tmp_path / "runtime_report.yaml"},
    )

    path = result.to_yaml(tmp_path / "component_validation.yaml")
    loaded = ComponentValidationResult.from_yaml(path)

    assert loaded.final_maturity == "runtime_integrated"
    assert loaded.contract.maturity == "runtime_integrated"
