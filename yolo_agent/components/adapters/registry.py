"""Registry for explicitly registered component adapters."""

from __future__ import annotations

import importlib
from typing import TypeVar

from yolo_agent.components.adapters.base import ComponentAdapter
from yolo_agent.components.contracts import ComponentContract

AdapterType = TypeVar("AdapterType", bound=type[ComponentAdapter])


class ComponentAdapterRegistry:
    """In-memory adapter registry; registration never mutates framework code."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[ComponentAdapter]] = {}

    def register(self, component_id: str, adapter: type[ComponentAdapter]) -> None:
        if not component_id.strip():
            raise ValueError("component_id must not be empty")
        if not issubclass(adapter, ComponentAdapter):
            raise TypeError("adapter must subclass ComponentAdapter")
        self._adapters[component_id] = adapter

    def get(self, component_id: str) -> type[ComponentAdapter] | None:
        return self._adapters.get(component_id)

    def create(self, component_id: str, **kwargs: object) -> ComponentAdapter:
        adapter = self.get(component_id)
        if adapter is None:
            raise KeyError(f"No adapter registered for component: {component_id}")
        return adapter(**kwargs)

    def create_for_contract(self, contract: ComponentContract, **kwargs: object) -> ComponentAdapter:
        """Create a registered adapter or resolve the contract's local implementation."""
        registered = self.get(contract.component_id)
        if registered is not None:
            return registered(**kwargs)
        if not contract.implementation_path or not contract.adapter_class:
            raise KeyError(f"Component contract has no adapter implementation: {contract.component_id}")
        module = importlib.import_module(contract.implementation_path)
        adapter_type = getattr(module, contract.adapter_class, None)
        if not isinstance(adapter_type, type) or not issubclass(adapter_type, ComponentAdapter):
            raise TypeError(
                f"Contract adapter is not a ComponentAdapter: "
                f"{contract.implementation_path}.{contract.adapter_class}"
            )
        self.register(contract.component_id, adapter_type)
        return adapter_type(**kwargs)

    def ids(self) -> list[str]:
        return sorted(self._adapters)


__all__ = ["ComponentAdapterRegistry"]
