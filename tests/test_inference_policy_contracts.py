from pathlib import Path

from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.contracts import load_contracts


def test_isolated_inference_contracts_are_conservative_and_loadable() -> None:
    contracts = load_contracts(
        Path("configs/components/inference/isolated_policies.yaml")
    )
    registry = ComponentAdapterRegistry()

    assert len(contracts) == 5
    for contract in contracts:
        adapter = registry.create_for_contract(contract)
        assert contract.component_id.startswith("inference.")
        assert contract.inference_only is True
        assert contract.training_only is False
        assert contract.changes_model_graph is False
        assert contract.fixed_imgsz_compatible is True
        assert contract.maturity == "adapter_implemented"
        assert contract.can_execute is False
        assert type(adapter).__name__ == contract.adapter_class
