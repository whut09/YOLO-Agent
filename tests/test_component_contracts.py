from pathlib import Path

import pytest

from yolo_agent.components.contracts import (
    ComponentContract,
    ComponentExecutionError,
    contract_from_card,
    load_contracts,
)
from yolo_agent.components.maturity import (
    MaturityTransitionError,
    can_transition,
    transition_maturity,
)
from yolo_agent.components.schema import ComponentCard
from yolo_agent.core.event_log import EventLog


def _contract(**updates: object) -> ComponentContract:
    data = {
        "component_id": "test.component",
        "display_name": "Test component",
        "category": "assigner",
        "implementation_path": "tests.fixtures",
        "adapter_class": "DummyAdapter",
        "maturity": "metadata_only",
        "fixed_imgsz_compatible": True,
    }
    data.update(updates)
    return ComponentContract.model_validate(data)


def test_contract_round_trips_yaml_and_forbids_unknown_fields(tmp_path: Path) -> None:
    contract = _contract()
    path = contract.to_yaml(tmp_path / "contract.yaml")
    loaded = ComponentContract.from_yaml(path)
    assert loaded.component_id == contract.component_id
    with pytest.raises(Exception):
        ComponentContract.model_validate({**contract.model_dump(), "unknown": True})


def test_metadata_and_pre_smoke_contracts_cannot_execute() -> None:
    for maturity in ("metadata_only", "reference_code_available", "adapter_implemented", "unit_tested"):
        with pytest.raises(ComponentExecutionError):
            _contract(maturity=maturity).assert_executable()


def test_smoke_requires_adapter_and_can_execute() -> None:
    _contract(maturity="smoke_passed").assert_executable(detector_family="generic", imgsz=640)
    with pytest.raises(ComponentExecutionError):
        _contract(maturity="smoke_passed", adapter_class=None).assert_executable()


def test_execution_checks_head_and_fixed_imgsz() -> None:
    with pytest.raises(ComponentExecutionError):
        _contract(maturity="smoke_passed", incompatible_heads=["yolo26_one_to_one"]).assert_executable(
            head="yolo26_one_to_one"
        )
    with pytest.raises(ComponentExecutionError):
        _contract(maturity="smoke_passed", fixed_imgsz_compatible=False).assert_executable(imgsz=640)


def test_maturity_is_sequential_and_event_logged(tmp_path: Path) -> None:
    assert can_transition("metadata_only", "reference_code_available")
    assert not can_transition("metadata_only", "smoke_passed")
    contract = _contract()
    log = EventLog(tmp_path / "events.jsonl")
    for target in ("reference_code_available", "adapter_implemented", "unit_tested", "smoke_passed"):
        contract = transition_maturity(contract, target, reason="test", event_log=log, run_id="run-1")
    assert contract.maturity == "smoke_passed"
    assert len(log.read()) == 4
    assert log.read()[-1].event_type == "component_maturity_changed"
    with pytest.raises(MaturityTransitionError):
        transition_maturity(contract, "production_eligible", reason="skip levels")
    with pytest.raises(MaturityTransitionError):
        transition_maturity(contract, "adapter_implemented", reason="")


def test_forced_demotion_requires_reason_and_is_logged(tmp_path: Path) -> None:
    contract = _contract(maturity="smoke_passed")
    log = EventLog(tmp_path / "events.jsonl")
    demoted = transition_maturity(contract, "adapter_implemented", reason="smoke regression", force=True, event_log=log)
    assert demoted.maturity == "adapter_implemented"
    assert log.read()[-1].details["forced"] is True


def test_legacy_card_conversion_is_metadata_only() -> None:
    card = ComponentCard(id="legacy", name="Legacy", type="assigner")
    contract = contract_from_card(card)
    assert contract.component_id == "legacy"
    assert contract.maturity == "metadata_only"
    assert not contract.can_execute


def test_compatibility_config_loads() -> None:
    contracts = load_contracts(Path("configs/component_compatibility.yaml"))
    assert contracts[0].component_id == "assigner.stal"
    assert contracts[0].maturity == "metadata_only"
