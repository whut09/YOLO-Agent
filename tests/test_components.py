"""Component card registry tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.components.registry import ComponentRegistry, configure, get_by_problem, get_by_type, load_cards
from yolo_agent.components.schema import ComponentCard
from yolo_agent.core.task_spec import TaskSpec


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_DIR = ROOT / "configs" / "components"
SCENARIO_DIR = ROOT / "configs" / "scenarios"


def test_load_cards_from_directory() -> None:
    """Bundled component cards should load from YAML."""
    cards = load_cards(COMPONENT_DIR)

    assert len(cards) == 8
    assert all(isinstance(card, ComponentCard) for card in cards)


def test_registry_filters_by_type_and_problem() -> None:
    """The registry should filter cards by component type and problem tag."""
    registry = ComponentRegistry.from_path(COMPONENT_DIR)

    bbox_losses = registry.get_by_type("bbox_loss")
    tiny_object_cards = registry.get_by_problem("tiny-objects")

    assert {card.id for card in bbox_losses} >= {
        "loss.bbox.ciou",
        "loss.bbox.wiou",
        "loss.bbox.mpdiou",
        "loss.bbox.nwd",
    }
    assert "loss.bbox.nwd" in {card.id for card in tiny_object_cards}
    assert "head.p2_small_object" in {card.id for card in tiny_object_cards}


def test_registry_filters_compatible_cards() -> None:
    """Compatibility filtering should honor task type and framework."""
    registry = ComponentRegistry.from_path(COMPONENT_DIR)
    task_spec = TaskSpec.from_yaml(SCENARIO_DIR / "infrared_small_target.yaml")

    compatible = registry.get_compatible(task_spec, "ultralytics")
    compatible_ids = {card.id for card in compatible}

    assert "loss.bbox.nwd" in compatible_ids
    assert "head.p2_small_object" in compatible_ids
    assert "assigner.stal" not in compatible_ids


def test_module_level_registry_helpers() -> None:
    """Convenience helpers should use the configured default registry."""
    configure(load_cards(COMPONENT_DIR))

    assert get_by_type("neck")
    assert get_by_problem("infrared_small_target")


def test_component_card_serialization_roundtrip(tmp_path: Path) -> None:
    """A card should serialize to YAML and load back without losing identity."""
    card = ComponentCard.from_yaml(COMPONENT_DIR / "neck.bifpn.yaml")
    output_path = tmp_path / "neck.bifpn.yaml"

    card.to_yaml(output_path)
    reloaded = ComponentCard.from_yaml(output_path)

    assert reloaded == card
    assert reloaded.search_space.default["fusion"] == "weighted"

