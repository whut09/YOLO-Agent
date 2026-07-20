"""Tests for conservative paper component alias resolution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yolo_agent.research.component_aliases import (
    CanonicalComponentDefinition,
    ComponentAliasConfig,
    ComponentAliasResolver,
)


@pytest.mark.parametrize(
    ("paper_id", "canonical_id"),
    [
        ("deformable_attention", "attention.deformable"),
        ("multi_scale_features", "feature_pyramid.multi_scale"),
        ("dynamic_head", "detection_head.dynamic"),
        ("p2_head", "head.p2_small_object"),
        ("learnable_proposals", "detection_head.learnable_proposals"),
        ("hybrid_matching", "matching.hybrid"),
        ("task_aligned_assignment", "assigner.task_aligned"),
        ("IoU_aware_classification", "quality_estimation.iou_aware_classification"),
        ("denoising", "augmentation.denoising"),
        ("feature_pyramid", "feature_pyramid.standard"),
        ("small_object_sampling", "sampling.small_object"),
        ("slicing", "inference.slicing"),
        ("distillation", "distillation.yolo26_teacher_student"),
        ("domain_adaptation", "domain_adaptation.general"),
        ("open_vocabulary_detection", "detection_head.open_vocabulary"),
    ],
)
def test_required_catalog_component_ids_resolve(paper_id: str, canonical_id: str) -> None:
    result = ComponentAliasResolver.from_yaml().resolve(paper_id)

    assert result.match_type == "exact_match"
    assert result.mappings[0].canonical_component_id == canonical_id


def test_default_aliases_cover_exact_normalized_semantic_and_unresolved() -> None:
    resolver = ComponentAliasResolver.from_yaml()

    exact = resolver.resolve("deformable_attention")
    normalized = resolver.resolve("Dynamic Head")
    semantic = resolver.resolve("knowledge distillation")
    unresolved = resolver.resolve("experimental_magic_adapter_v9")

    assert exact.match_type == "exact_match"
    assert exact.mappings[0].canonical_component_id == "attention.deformable"
    assert normalized.match_type == "normalized_match"
    assert normalized.mappings[0].canonical_component_id == "detection_head.dynamic"
    assert semantic.match_type == "semantic_match"
    assert semantic.mappings[0].canonical_component_id == "distillation.yolo26_teacher_student"
    assert unresolved.match_type == "unresolved"
    assert unresolved.mappings == []


def test_real_adapter_status_comes_from_contract_not_alias_name() -> None:
    resolver = ComponentAliasResolver.from_yaml()

    resolved = resolver.resolve("small_object_sampling")
    guessed = resolver.resolve("small_object_sampling_next")

    mapping = resolved.mappings[0]
    assert mapping.canonical_component_id == "sampling.small_object"
    assert mapping.adapter_verified is True
    assert mapping.maturity == "smoke_passed"
    assert mapping.implementation_status == "smoke_passed"
    assert mapping.executable is True
    assert guessed.match_type == "unresolved"
    assert resolver.resolve("p2_head").mappings[0].executable is True


def test_alias_without_contract_cannot_claim_adapter_implementation() -> None:
    definition = CanonicalComponentDefinition(
        canonical_component_id="attention.claimed",
        category="attention",
        aliases=["claimed_attention"],
        mapping_reason="Paper taxonomy only.",
    )
    resolver = ComponentAliasResolver(ComponentAliasConfig(canonical_components=[definition]))

    mapping = resolver.resolve("claimed_attention").mappings[0]

    assert mapping.adapter_verified is False
    assert mapping.maturity == "metadata_only"
    assert mapping.implementation_status == "metadata_only"
    assert mapping.executable is False


def test_compound_alias_records_split_reason_and_multiple_mappings() -> None:
    result = ComponentAliasResolver.from_yaml().resolve("small_object_multiscale_recipe")

    assert result.resolved is True
    assert result.split_reason
    assert {item.canonical_component_id for item in result.mappings} == {
        "feature_pyramid.multi_scale",
        "sampling.small_object",
    }
    assert all("Split from a broad paper concept" in item.mapping_reason for item in result.mappings)


def test_synonymous_aliases_resolve_to_same_canonical_component() -> None:
    resolver = ComponentAliasResolver.from_yaml()

    first = resolver.resolve("IoU_aware_classification")
    second = resolver.resolve("iou-aware-classification")

    assert first.mappings[0].canonical_component_id == "quality_estimation.iou_aware_classification"
    assert second.mappings[0].canonical_component_id == first.mappings[0].canonical_component_id
    assert second.match_type == "normalized_match"


def test_conflicting_aliases_are_rejected() -> None:
    first = CanonicalComponentDefinition(
        canonical_component_id="attention.first",
        category="attention",
        aliases=["shared-alias"],
        mapping_reason="First.",
    )
    second = CanonicalComponentDefinition(
        canonical_component_id="attention.second",
        category="attention",
        aliases=["shared_alias"],
        mapping_reason="Second.",
    )

    with pytest.raises(ValidationError, match="conflicting component alias"):
        ComponentAliasConfig(canonical_components=[first, second])
