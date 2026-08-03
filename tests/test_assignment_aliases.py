"""Awesome assignment alias coverage tests."""

from __future__ import annotations

from yolo_agent.research.component_aliases import ComponentAliasResolver


def test_awesome_assignment_aliases_resolve_without_granting_execution() -> None:
    resolver = ComponentAliasResolver.from_yaml()

    tood = resolver.resolve("task_aligned_assigner").mappings[0]
    ota = resolver.resolve("optimal_transport_assigner").mappings[0]
    dsla = resolver.resolve("dynamic_smooth_labels").mappings[0]

    assert tood.canonical_component_id == "assigner.task_aligned"
    assert tood.adapter_verified is True and tood.executable is False
    assert ota.canonical_component_id == "assigner.optimal_transport"
    assert ota.adapter_verified is True and ota.executable is False
    assert dsla.canonical_component_id == "assigner.dynamic_smooth_label"
    assert dsla.adapter_verified is True and dsla.executable is False


def test_reusable_assignment_aliases_resolve_without_granting_maturity() -> None:
    resolver = ComponentAliasResolver.from_yaml()
    aliases = {
        "task_aligned_weighting": "assigner.task_aligned_weighting",
        "dynamic_topk_matching": "assigner.dynamic_topk",
        "quality_aware_matching": "assigner.quality_aware",
        "soft_label_assignment": "assigner.soft_label",
        "dual_path_assignment": "assigner.dual_path",
        "conflict_aware_positive_selection": "assigner.conflict_aware",
    }

    for alias, component_id in aliases.items():
        mapping = resolver.resolve(alias).mappings[0]
        assert mapping.canonical_component_id == component_id
        assert mapping.adapter_verified is True
        assert mapping.maturity == "adapter_implemented"
        assert mapping.executable is False


def test_generic_assignment_remains_unresolved() -> None:
    assert ComponentAliasResolver.from_yaml().resolve("assignment").match_type == (
        "unresolved"
    )
