"""Registry for explicitly registered component adapters."""

from __future__ import annotations

from typing import TypeVar

from yolo_agent.components.adapters.base import ComponentAdapter

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

    def ids(self) -> list[str]:
        return sorted(self._adapters)


__all__ = ["ComponentAdapterRegistry"]
