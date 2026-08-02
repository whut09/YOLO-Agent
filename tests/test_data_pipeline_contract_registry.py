from __future__ import annotations

from yolo_agent.components.adapters.audit_contract import EXPECTED_RUNTIME_ADAPTERS
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.contracts import load_contracts


def test_data_pipeline_contracts_resolve_distinct_adapters() -> None:
    contracts = load_contracts(
        "configs/components/data_pipeline/paper_data_adapters.yaml"
    )
    registry = ComponentAdapterRegistry()

    assert len(contracts) == 9
    assert all(item.maturity == "adapter_implemented" for item in contracts)
    assert all(not item.can_execute for item in contracts)
    for contract in contracts:
        adapter = registry.create_for_contract(contract)
        expectation = EXPECTED_RUNTIME_ADAPTERS[contract.component_id]
        assert adapter.component_id == contract.component_id
        assert adapter.changed_variable == expectation.changed_variable
