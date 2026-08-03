from __future__ import annotations

import pytest
from pydantic import ValidationError

from yolo_agent.research.mechanism_clusters import (
    ClusterEvidence,
    MechanismClusterConfig,
    MechanismClusterDefinition,
    PaperMechanismClusterMatch,
)


def _cluster(cluster_id: str, semantic: str) -> MechanismClusterDefinition:
    return MechanismClusterDefinition(
        cluster_id=cluster_id,
        display_name=cluster_id,
        training_semantic=semantic,
        adapter_family="loss.shared",
        method_families=[cluster_id],
        insertion_points=["trainer_loss"],
        required_runtime_hooks=["compute_loss"],
    )


def test_cluster_config_rejects_duplicate_training_semantics() -> None:
    with pytest.raises(ValidationError, match="duplicate adapter-family"):
        MechanismClusterConfig(clusters=[
            _cluster("quality-a", "auxiliary_quality_target"),
            _cluster("quality-b", "auxiliary_quality_target"),
        ])


def test_semantic_match_requires_evidence_and_confidence() -> None:
    with pytest.raises(ValidationError, match="requires source evidence"):
        PaperMechanismClusterMatch(
            paper_id="paper",
            profile_id="profile",
            cluster_id="quality_alignment",
            adapter_family="loss.quality_alignment",
            training_semantic="auxiliary_quality_target",
            match_type="semantic_match",
            confidence="medium",
            confidence_score=0.7,
            match_reason="semantic alias",
        )

    match = PaperMechanismClusterMatch(
        paper_id="paper",
        profile_id="profile",
        cluster_id="quality_alignment",
        adapter_family="loss.quality_alignment",
        training_semantic="auxiliary_quality_target",
        match_type="semantic_match",
        confidence="high",
        confidence_score=0.9,
        evidence=[ClusterEvidence(
            field_name="method_family",
            value="quality_alignment",
            source="summary",
            source_location="summary:paragraph:1",
            confidence="high",
        )],
        match_reason="explicit method family and trainer loss insertion",
    )

    assert match.evidence[0].source_location == "summary:paragraph:1"


def test_bundled_cluster_taxonomy_covers_required_runtime_mechanisms() -> None:
    config = MechanismClusterConfig.from_yaml()

    assert len(config.clusters) == 47
    assert {item.cluster_id for item in config.clusters} == {
        "sampling_class_balancing",
        "small_object_weighted_sampling",
        "class_balanced_sampling",
        "repeat_factor_sampling",
        "hard_negative_replay",
        "false_negative_class_boost",
        "hard_example_mining",
        "augmentation",
        "rare_class_copy_paste",
        "scale_aware_crop",
        "object_centric_crop",
        "multi_image_sampling_schedule",
        "label_assignment",
        "task_aligned_weighting",
        "dynamic_topk_matching",
        "quality_aware_matching",
        "soft_label_assignment",
        "dual_path_assignment",
        "conflict_aware_positive_selection",
        "quality_alignment",
        "confidence_calibration",
        "localization_loss",
        "class_balanced_focal",
        "feature_distillation",
        "logits_distillation",
        "localization_distillation",
        "relation_distillation",
        "attention_distillation",
        "masked_feature_distillation",
        "quality_aware_distillation",
        "teacher_ensemble_distillation",
        "multi_scale_feature_fusion",
        "weighted_feature_pyramid",
        "bidirectional_feature_fusion",
        "gather_distribute_fusion",
        "small_object_head",
        "channel_attention",
        "spatial_attention",
        "deformable_feature_aggregation",
        "reparameterized_convolution",
        "lightweight_neck",
        "large_kernel_neck",
        "feature_alignment",
        "domain_adaptation",
        "open_vocabulary",
        "slicing_inference",
        "post_processing_calibration",
    }
    feature = next(item for item in config.clusters if item.cluster_id == "feature_distillation")
    logits = next(item for item in config.clusters if item.cluster_id == "logits_distillation")
    assert feature.training_semantic != logits.training_semantic
    train_calibration = next(item for item in config.clusters if item.cluster_id == "confidence_calibration")
    posthoc = next(item for item in config.clusters if item.cluster_id == "post_processing_calibration")
    assert train_calibration.adapter_family != posthoc.adapter_family
