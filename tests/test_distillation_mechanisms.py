from __future__ import annotations

from yolo_agent.components.distillation.mechanisms import (
    DISTILLATION_COMPONENTS,
    DISTILLATION_MECHANISMS,
)


def test_distillation_mechanisms_have_independent_runtime_identities() -> None:
    assert set(DISTILLATION_MECHANISMS) == {
        "logits",
        "feature",
        "localization",
        "relation",
        "attention",
        "masked_feature",
        "quality_aware",
        "teacher_ensemble",
    }
    assert len(DISTILLATION_COMPONENTS) == 8
    assert len(
        {item.changed_variable for item in DISTILLATION_MECHANISMS.values()}
    ) == 8
    assert all(
        item.changed_variable == f"loss.distillation.{item.mechanism}.weight"
        for item in DISTILLATION_MECHANISMS.values()
    )


def test_distillation_mechanism_requirements_are_explicit() -> None:
    assert DISTILLATION_MECHANISMS["feature"].requires_features
    assert DISTILLATION_MECHANISMS["relation"].requires_features
    assert DISTILLATION_MECHANISMS["attention"].requires_features
    assert DISTILLATION_MECHANISMS["masked_feature"].requires_features
    assert DISTILLATION_MECHANISMS["localization"].requires_boxes
    assert DISTILLATION_MECHANISMS["teacher_ensemble"].requires_multiple_teachers
    assert not DISTILLATION_MECHANISMS["logits"].requires_features
