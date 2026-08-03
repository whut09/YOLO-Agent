import pytest

from yolo_agent.recipes.coupled_library import (
    LocalCouplingDiagnosis,
    coupling_evidence_from_diagnosis,
    coupling_evidence_from_method_profile,
)
from yolo_agent.research.method_profiles import PaperMethodProfile


def _profile(*, reason: str | None) -> PaperMethodProfile:
    return PaperMethodProfile(
        profile_id="profile:paper-1",
        paper_id="paper-1",
        method_name="Small object detector",
        canonical_component_ids=[
            "head.p2_small_object",
            "sampling.small_object",
            "loss.quality.correlation",
        ],
        paper_parameters={"coupling_reason": reason} if reason else {},
        source_locations=["notes/paper-1.md#method"],
    )


def test_method_profile_requires_explicit_reason_for_exact_pair() -> None:
    evidence = coupling_evidence_from_method_profile(
        _profile(reason="The method couples increased small-instance exposure with stride-4 features."),
        ["head.p2_small_object", "sampling.small_object"],
    )

    assert evidence.evidence_kind == "method_profile"
    assert evidence.paper_ids == ["paper-1"]
    assert evidence.verified is True

    with pytest.raises(ValueError, match="no explicit coupling_reason"):
        coupling_evidence_from_method_profile(
            _profile(reason=None),
            ["head.p2_small_object", "sampling.small_object"],
        )


def test_same_paper_component_list_does_not_authorize_arbitrary_bundle() -> None:
    with pytest.raises(ValueError, match="requested component pair"):
        coupling_evidence_from_method_profile(
            _profile(reason="Explicit reason for the documented pair."),
            [
                "head.p2_small_object",
                "sampling.small_object",
                "loss.quality.correlation",
            ],
        )


def test_local_diagnosis_must_be_verified_and_fact_bound() -> None:
    diagnosis = LocalCouplingDiagnosis(
        diagnosis_id="diagnosis-1",
        component_ids=["head.p2_small_object", "sampling.small_object"],
        reason="Observed AP_small and small-object FN indicate complementary representation and exposure causes.",
        error_fact_ids=["fact:ap_small", "fact:small_fn"],
        source_location="runs/one/diagnosis.yaml#small-object",
        confidence=0.8,
        verified=True,
    )

    evidence = coupling_evidence_from_diagnosis(diagnosis)

    assert evidence.evidence_kind == "local_diagnosis"
    assert evidence.error_fact_ids == ["fact:ap_small", "fact:small_fn"]
    assert evidence.paper_ids == []

    with pytest.raises(ValueError, match="must be verified"):
        coupling_evidence_from_diagnosis(diagnosis.model_copy(update={"verified": False}))
