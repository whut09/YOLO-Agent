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
    )
}

DISTILLATION_COMPONENTS = {
    item.component_id: item for item in DISTILLATION_MECHANISMS.values()
}


__all__ = [
    "DISTILLATION_COMPONENTS",
    "DISTILLATION_MECHANISMS",
    "DistillationMechanism",
    "DistillationMechanismSpec",
]
