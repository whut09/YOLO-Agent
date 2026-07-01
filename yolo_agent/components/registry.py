"""Component card registry and filtering helpers."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.components.schema import ComponentCard, ComponentType, FrameworkName
from yolo_agent.core.task_spec import TaskSpec


def load_cards(path: Path | str) -> list[ComponentCard]:
    """Load component cards from a YAML file or directory of YAML files."""
    source = Path(path)
    if source.is_file():
        return [ComponentCard.from_yaml(source)]
    if not source.exists():
        raise FileNotFoundError(f"Component card path does not exist: {source}")

    cards: list[ComponentCard] = []
    for card_path in sorted(source.glob("*.yaml")):
        cards.append(ComponentCard.from_yaml(card_path))
    return cards


class ComponentRegistry:
    """In-memory registry for component card lookup."""

    def __init__(self, cards: list[ComponentCard]) -> None:
        self.cards = cards

    @classmethod
    def from_path(cls, path: Path | str) -> "ComponentRegistry":
        """Create a registry from a YAML file or directory."""
        return cls(load_cards(path))

    def get_by_type(self, component_type: ComponentType) -> list[ComponentCard]:
        """Return cards matching a component type."""
        return [card for card in self.cards if card.type == component_type]

    def get_by_problem(self, problem: str) -> list[ComponentCard]:
        """Return cards that target a problem tag."""
        normalized = problem.lower().replace("-", "_")
        return [
            card
            for card in self.cards
            if normalized in {target.lower().replace("-", "_") for target in card.target_problems}
        ]

    def get_compatible(self, task_spec: TaskSpec, framework: FrameworkName | str) -> list[ComponentCard]:
        """Return cards compatible with a task spec and framework."""
        return [
            card
            for card in self.cards
            if _matches_framework(card, framework) and task_spec.task_type in card.compatible_tasks
        ]


_default_registry: ComponentRegistry | None = None


def configure(cards: list[ComponentCard]) -> None:
    """Set the module-level registry used by convenience query functions."""
    global _default_registry
    _default_registry = ComponentRegistry(cards)


def get_by_type(component_type: ComponentType) -> list[ComponentCard]:
    """Return cards matching a component type from the default registry."""
    return _require_default_registry().get_by_type(component_type)


def get_by_problem(problem: str) -> list[ComponentCard]:
    """Return cards targeting a problem from the default registry."""
    return _require_default_registry().get_by_problem(problem)


def get_compatible(task_spec: TaskSpec, framework: FrameworkName | str) -> list[ComponentCard]:
    """Return compatible cards from the default registry."""
    return _require_default_registry().get_compatible(task_spec, framework)


def _require_default_registry() -> ComponentRegistry:
    if _default_registry is None:
        raise RuntimeError("Component registry is not configured. Call configure(load_cards(path)) first.")
    return _default_registry


def _matches_framework(card: ComponentCard, framework: FrameworkName | str) -> bool:
    return framework in card.compatible_frameworks or "generic" in card.compatible_frameworks

