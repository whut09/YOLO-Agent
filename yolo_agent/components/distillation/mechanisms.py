"""Stable identities for reusable YOLO26 distillation mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DistillationMechanism = Literal[
    "logits",
    "feature",
    "localization",
    "relation",
    "attention",
    "masked_feature",
    "quality_aware",
    "teacher_ensemble",
    "source_free_teacher",
    "cross_domain_teacher",
    "contrastive",
    "pearson_feature",
    "richness_masked",
    "classifier_response",
    "instance_conditional",
    "structural_similarity",
    "prediction_guided",
    "hetero_assist",
    "global_prototype",
    "base_novel_commonality",
    "bovw_consistency",
    "glam_attention",
    "query_distillation",
    "cross_scale_self",
    "early_learning",
]


@dataclass(frozen=True)
class DistillationMechanismSpec:
    mechanism: DistillationMechanism
    component_id: str
    changed_variable: str
    requires_features: bool = False
    requires_boxes: bool = False
    requires_multiple_teachers: bool = False


DISTILLATION_MECHANISMS = {
    item.mechanism: item
    for item in (
        DistillationMechanismSpec(
            mechanism="logits",
            component_id="distillation.logits",
            changed_variable="loss.distillation.logits.weight",
        ),
        DistillationMechanismSpec(
            mechanism="feature",
            component_id="distillation.feature",
            changed_variable="loss.distillation.feature.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="localization",
            component_id="distillation.localization",
            changed_variable="loss.distillation.localization.weight",
            requires_boxes=True,
        ),
        DistillationMechanismSpec(
            mechanism="relation",
            component_id="distillation.relation",
            changed_variable="loss.distillation.relation.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="attention",
            component_id="distillation.attention",
            changed_variable="loss.distillation.attention.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="masked_feature",
            component_id="distillation.masked_feature",
            changed_variable="loss.distillation.masked_feature.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="quality_aware",
            component_id="distillation.quality_aware",
            changed_variable="loss.distillation.quality_aware.weight",
        ),
        DistillationMechanismSpec(
            mechanism="teacher_ensemble",
            component_id="distillation.teacher_ensemble",
            changed_variable="loss.distillation.teacher_ensemble.weight",
            requires_multiple_teachers=True,
        ),
        DistillationMechanismSpec(
            mechanism="source_free_teacher",
            component_id="distillation.source_free_teacher",
            changed_variable="loss.distillation.source_free_teacher.weight",
        ),
        DistillationMechanismSpec(
            mechanism="cross_domain_teacher",
            component_id="distillation.cross_domain_teacher",
            changed_variable="loss.distillation.cross_domain_teacher.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="contrastive",
            component_id="distillation.contrastive",
            changed_variable="loss.distillation.contrastive.weight",
            requires_features=True,
        ),
        # Paper-specific mechanisms recovered from full-text evidence (frozen-83
        # gap closure).  Each keeps its own loss family and changed variable so
        # no two papers ever collapse into one generic adapter.
        DistillationMechanismSpec(
            mechanism="pearson_feature",
            component_id="distillation.pearson_feature",
            changed_variable="loss.distillation.pearson_feature.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="richness_masked",
            component_id="distillation.richness_masked",
            changed_variable="loss.distillation.richness_masked.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="classifier_response",
            component_id="distillation.classifier_response",
            changed_variable="loss.distillation.classifier_response.weight",
        ),
        DistillationMechanismSpec(
            mechanism="instance_conditional",
            component_id="distillation.instance_conditional",
            changed_variable="loss.distillation.instance_conditional.weight",
            requires_features=True,
            requires_boxes=True,
        ),
        DistillationMechanismSpec(
            mechanism="structural_similarity",
            component_id="distillation.structural_similarity",
            changed_variable="loss.distillation.structural_similarity.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="prediction_guided",
            component_id="distillation.prediction_guided",
            changed_variable="loss.distillation.prediction_guided.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="hetero_assist",
            component_id="distillation.hetero_assist",
            changed_variable="loss.distillation.hetero_assist.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="global_prototype",
            component_id="distillation.global_prototype",
            changed_variable="loss.distillation.global_prototype.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="base_novel_commonality",
            component_id="distillation.base_novel_commonality",
            changed_variable="loss.distillation.base_novel_commonality.weight",
        ),
        DistillationMechanismSpec(
            mechanism="bovw_consistency",
            component_id="distillation.bovw_consistency",
            changed_variable="loss.distillation.bovw_consistency.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="glam_attention",
            component_id="distillation.glam_attention",
            changed_variable="loss.distillation.glam_attention.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="query_distillation",
            component_id="distillation.query_distillation",
            changed_variable="loss.distillation.query_distillation.weight",
            requires_features=True,
            requires_boxes=True,
        ),
        DistillationMechanismSpec(
            mechanism="cross_scale_self",
            component_id="distillation.cross_scale_self",
            changed_variable="loss.distillation.cross_scale_self.weight",
            requires_features=True,
        ),
        DistillationMechanismSpec(
            mechanism="early_learning",
            component_id="distillation.early_learning",
            changed_variable="loss.distillation.early_learning.weight",
        ),
    )
}

DISTILLATION_COMPONENTS = {
    item.component_id: item for item in DISTILLATION_MECHANISMS.values()
}

# Runtime payloads use the paper-facing route identity.  The shorter
# mechanism names remain the loss implementation keys for backwards
# compatibility with the native criterion.
DISTILLATION_ROUTE_IDS = {
    "logits": "logits_distillation",
    "feature": "feature_distillation",
    "relation": "relation_distillation",
    "localization": "localization_distillation",
    "attention": "attention_distillation",
    "masked_feature": "masked_feature_distillation",
    "quality_aware": "quality_aware_distillation",
    "teacher_ensemble": "teacher_ensemble",
    "source_free_teacher": "source_free_teacher",
    "cross_domain_teacher": "cross_domain_teacher",
    "contrastive": "contrastive_distillation",
    "pearson_feature": "pearson_feature_distillation",
    "richness_masked": "richness_masked_distillation",
    "classifier_response": "classifier_response_distillation",
    "instance_conditional": "instance_conditional_distillation",
    "structural_similarity": "structural_similarity_distillation",
    "prediction_guided": "prediction_guided_distillation",
    "hetero_assist": "hetero_assist_distillation",
    "global_prototype": "global_prototype_distillation",
    "base_novel_commonality": "base_novel_commonality_distillation",
    "bovw_consistency": "bovw_consistency_distillation",
    "glam_attention": "glam_attention_distillation",
    "query_distillation": "query_distillation",
    "cross_scale_self": "cross_scale_self_distillation",
    "early_learning": "early_learning_distillation",
}


__all__ = [
    "DISTILLATION_COMPONENTS",
    "DISTILLATION_MECHANISMS",
    "DISTILLATION_ROUTE_IDS",
    "DistillationMechanism",
    "DistillationMechanismSpec",
]
