import pytest

from yolo_agent.recipes.coupled_library import (
    CoupledRecipeTemplateConfig,
    CouplingEvidence,
)


def test_bundled_library_contains_only_five_allowlisted_pairs() -> None:
    config = CoupledRecipeTemplateConfig.from_yaml()

    assert [item.template_id for item in config.templates] == [
        "p2_small_object_sampling",
        "feature_fusion_quality_loss",
        "distillation_class_balanced_sampling",
        "assignment_quality_alignment",
        "slicing_confidence_calibration",
    ]
    assert config.templates[-1].execution_track == "inference"
    assert all(item.target_error_facts for item in config.templates)


def test_template_matches_only_one_component_from_each_side() -> None:
    template = CoupledRecipeTemplateConfig.from_yaml().templates[0]

    assert template.match(["sampling.small_object", "head.p2_small_object"]) == (
        "head.p2_small_object",
        "sampling.small_object",
    )
    assert template.match(["head.p2_small_object"]) is None
    assert template.match(
        ["head.p2_small_object", "sampling.small_object", "loss.quality.correlation"]
    ) is None


def test_coupling_evidence_requires_explicit_typed_source() -> None:
    evidence = CouplingEvidence(
        evidence_kind="local_diagnosis",
        source_id="diagnosis-1",
        component_ids=["head.p2_small_object", "sampling.small_object"],
        reason="Small-object exposure and stride-4 features address distinct observed causes.",
        source_locations=["diagnosis.yaml#finding-1"],
        error_fact_ids=["fact:ap_small", "fact:small_fn"],
        verified=True,
    )

    assert len(evidence.evidence_hash) == 64
    with pytest.raises(ValueError, match="paper_ids"):
        CouplingEvidence(
            evidence_kind="method_profile",
            source_id="profile-1",
            component_ids=["a", "b"],
            reason="Explicit paper coupling.",
            source_locations=["note.md#method"],
        )
