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
        ("IoU_aware_classification", "loss.quality.iou_aware_classification"),
        ("denoising", "augmentation.denoising"),
        ("feature_pyramid", "feature_pyramid.standard"),
        ("small_object_sampling", "sampling.small_object"),
        ("slicing", "inference.sahi_slicing"),
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
    assert mapping.maturity == "adapter_implemented"
    assert mapping.implementation_status == "adapter_implemented"
    assert mapping.executable is False
    assert guessed.match_type == "unresolved"
    p2 = resolver.resolve("p2_head").mappings[0]
    assert p2.executable is False
    assert p2.adapter_verified is True
    assert p2.maturity == "adapter_implemented"


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

    assert first.mappings[0].canonical_component_id == "loss.quality.iou_aware_classification"
    assert second.mappings[0].canonical_component_id == first.mappings[0].canonical_component_id
    assert second.match_type == "normalized_match"


@pytest.mark.parametrize(
    ("paper_component_id", "canonical_id"),
    [
        ("localization_aware_classification", "loss.quality.localization_aware"),
        ("boundary_aware_loss", "loss.boundary_aware"),
        (
            "uncertainty_weighted_regression",
            "loss.localization.uncertainty_weighted",
        ),
        ("hard_negative_classification", "loss.hard_negative_classification"),
        ("class_balanced_focal", "loss.class_balanced_focal"),
    ],
)
def test_extended_loss_aliases_resolve_to_real_conservative_adapters(
    paper_component_id: str,
    canonical_id: str,
) -> None:
    mapping = ComponentAliasResolver.from_yaml().resolve(paper_component_id).mappings[0]

    assert mapping.canonical_component_id == canonical_id
    assert mapping.adapter_verified is True
    assert mapping.maturity == "adapter_implemented"
    assert mapping.artifact_execution_ready is False


@pytest.mark.parametrize(
    "paper_component_id",
    ["sliced_inference", "high_resolution_tiling", "overlap_merge"],
)
def test_explicit_sahi_mechanisms_resolve_without_mapping_small_object_task(
    paper_component_id: str,
) -> None:
    resolver = ComponentAliasResolver.from_yaml()

    result = resolver.resolve(paper_component_id)

    assert result.mappings[0].canonical_component_id == "inference.sahi_slicing"
    assert resolver.resolve("small_object").match_type == "unresolved"
    assert resolver.resolve("tiny_object").match_type == "unresolved"


@pytest.mark.parametrize(
    ("paper_component_id", "canonical_id"),
    [
        ("classification_localization", "quality_alignment.general"),
        ("quality_estimation", "quality_alignment.general"),
        ("mutual_supervision", "quality_alignment.general"),
        ("task_aligned_head", "detection_head.task_aligned"),
    ],
)
def test_broad_quality_alignment_does_not_claim_a_specific_loss_adapter(
    paper_component_id: str,
    canonical_id: str,
) -> None:
    result = ComponentAliasResolver.from_yaml().resolve(paper_component_id)

    mapping = result.mappings[0]
    assert mapping.canonical_component_id == canonical_id
    assert mapping.adapter_verified is False
    assert mapping.artifact_execution_ready is False


@pytest.mark.parametrize(
    ("paper_component_id", "canonical_id"),
    [
        ("localization_distillation", "distillation.localization"),
        ("feature_distillation", "distillation.feature"),
        ("logits_distillation", "distillation.logits"),
        ("relation_distillation", "distillation.relation"),
        ("attention_distillation", "distillation.attention"),
        ("masked_feature_distillation", "distillation.masked_feature"),
        ("quality_aware_distillation", "distillation.quality_aware"),
        ("teacher_ensemble_distillation", "distillation.teacher_ensemble"),
        ("visual_linguistic_distillation", "distillation.vision_language"),
        ("cross_modality_distillation", "distillation.cross_modal"),
    ],
)
def test_distillation_variants_preserve_detector_family_boundaries(
    paper_component_id: str,
    canonical_id: str,
) -> None:
    mapping = ComponentAliasResolver.from_yaml().resolve(paper_component_id).mappings[0]

    assert mapping.canonical_component_id == canonical_id
    if canonical_id in {"distillation.vision_language", "distillation.cross_modal"}:
        assert mapping.yolo26_compatibility == "incompatible"
        assert mapping.adapter_verified is False


def test_cross_scale_fusion_reuses_neck_but_cross_modal_fusion_does_not() -> None:
    resolver = ComponentAliasResolver.from_yaml()

    cross_scale = resolver.resolve("cross_scale_fusion")

    assert cross_scale.mappings[0].canonical_component_id == "neck.multi_scale_fusion"
    assert resolver.resolve("cross_modal_fusion").match_type == "unresolved"
    assert resolver.resolve("vision_language_fusion").match_type == "unresolved"
    assert resolver.resolve("feature_reuse").match_type == "unresolved"


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
