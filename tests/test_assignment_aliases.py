"""Awesome assignment alias coverage tests."""

from __future__ import annotations

from yolo_agent.research.component_aliases import ComponentAliasResolver


def test_awesome_assignment_aliases_resolve_to_verified_shadow_adapters() -> None:
    resolver = ComponentAliasResolver.from_yaml()

    tood = resolver.resolve("task_aligned_assigner").mappings[0]
    ota = resolver.resolve("optimal_transport_assigner").mappings[0]
    dsla = resolver.resolve("dynamic_smooth_labels").mappings[0]

    assert tood.canonical_component_id == "assigner.task_aligned"
    assert tood.executable is True
    assert ota.canonical_component_id == "assigner.optimal_transport"
    assert ota.executable is True
    assert dsla.canonical_component_id == "assigner.dynamic_smooth_label"
    assert dsla.executable is True
