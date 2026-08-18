from __future__ import annotations

from yolo_agent.research.component_aliases import ComponentAliasConfig
from yolo_agent.research.mechanism_evidence import PaperMechanismEvidence
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodProfile,
)
from yolo_agent.research.paper_mechanism_resolver import (
    PaperMechanismResolver,
    merge_paper_mechanism_resolutions,
)


def _profile(
    paper_id: str,
    *,
    components: list[str],
    changed_variables: list[str] | None = None,
    evidence: list[PaperMechanismEvidence] | None = None,
) -> PaperMethodProfile:
    return PaperMethodProfile(
        profile_id=f"profile:{paper_id}",
        paper_id=paper_id,
        paper_component_ids=components,
        paper_parameters={"changed_variables": changed_variables or []},
        source_locations=["paper_record.component_ids"],
        mechanism_evidence=evidence or [],
    )


def _decision(
    paper_id: str,
    canonical: list[str],
) -> PaperImplementationDecision:
    return PaperImplementationDecision(
        paper_id=paper_id,
        profile_id=f"profile:{paper_id}",
        decision="new_method_profile",
        canonical_component_ids=canonical,
        reasons=["fixture"],
    )


def _resolver() -> PaperMechanismResolver:
    return PaperMechanismResolver.from_alias_config(
        ComponentAliasConfig.from_yaml()
    )


def test_generic_distillation_stays_unresolved() -> None:
    result = _resolver().resolve_profile(
        _profile("paper-a", components=["knowledge_distillation"]),
        _decision("paper-a", ["distillation.yolo26_teacher_student"]),
    ).resolutions[0]

    assert result.resolved is False
    assert result.executable_candidate is False
    assert result.paper_specific_mechanism_id is None
    assert "generic mechanism" in (result.unresolved_reason or "")


def test_summary_mechanism_resolves_relation_distillation() -> None:
    result = _resolver().resolve_profile(
        _profile(
            "paper-a",
            components=["knowledge_distillation"],
            changed_variables=["loss.distillation.relation.weight"],
        ),
        _decision("paper-a", ["distillation.yolo26_teacher_student"]),
    ).resolutions[0]

    assert result.paper_specific_mechanism_id == "relation_distillation"
    assert result.canonical_component_id == "distillation.relation"
    assert result.required_adapter == "distillation.relation"
    assert result.compatibility == "adapter_required"


def test_title_only_mechanism_does_not_resolve() -> None:
    evidence = PaperMechanismEvidence(
        paper_id="paper-a",
        source_term="response_distillation",
        canonical_component_id="distillation.logits",
        source="title",
        source_location="paper_record.title",
        alias_match_type="exact",
    )
    result = _resolver().resolve_profile(
        _profile(
            "paper-a",
            components=["knowledge_distillation"],
            evidence=[evidence],
        ),
        _decision("paper-a", ["distillation.yolo26_teacher_student"]),
    ).resolutions[0]

    assert result.resolved is False


def test_quality_and_assignment_implementations_remain_distinct() -> None:
    result = _resolver().resolve_profile(
        _profile("paper-a", components=["quality_assignment"]),
        _decision(
            "paper-a",
            [
                "assigner.optimal_transport",
                "assigner.task_aligned",
                "loss.quality.correlation",
                "loss.quality.pseudo_iou",
            ],
        ),
    )

    canonical = {
        item.canonical_component_id for item in result.resolutions
    }
    assert canonical == {
        "assigner.optimal_transport",
        "assigner.task_aligned",
        "loss.quality.correlation",
        "loss.quality.pseudo_iou",
    }
    assert len({item.execution_fingerprint for item in result.resolutions}) == 4


def test_same_execution_merges_paper_ids_without_title_or_year() -> None:
    resolver = _resolver()
    first = resolver.resolve_profile(
        _profile(
            "paper-a",
            components=["relation_distillation"],
            changed_variables=["loss.distillation.relation.weight"],
        ),
        _decision("paper-a", ["distillation.yolo26_teacher_student"]),
    ).resolutions[0]
    second = resolver.resolve_profile(
        _profile(
            "paper-b",
            components=["relation_distillation"],
            changed_variables=["loss.distillation.relation.weight"],
        ),
        _decision("paper-b", ["distillation.yolo26_teacher_student"]),
    ).resolutions[0]

    assert first.execution_fingerprint == second.execution_fingerprint
    groups = merge_paper_mechanism_resolutions([first, second])
    assert len(groups) == 1
    assert groups[0].paper_ids == ["paper-a", "paper-b"]


def test_different_implementations_in_one_paper_are_not_deduplicated() -> None:
    result = _resolver().resolve_profile(
        _profile(
            "paper-a",
            components=["knowledge_distillation"],
            changed_variables=[
                "loss.distillation.logits.weight",
                "loss.distillation.feature.weight",
            ],
        ),
        _decision("paper-a", ["distillation.yolo26_teacher_student"]),
    )

    assert {
        item.paper_specific_mechanism_id for item in result.resolutions
    } == {"feature_distillation", "logits_distillation"}
    assert len({item.execution_fingerprint for item in result.resolutions}) == 2


def test_incompatible_specific_component_stays_incompatible() -> None:
    result = _resolver().resolve_profile(
        _profile("paper-a", components=["vision_language_distillation"]),
        _decision("paper-a", ["distillation.vision_language"]),
    ).resolutions[0]

    assert result.paper_specific_mechanism_id == "distillation.vision_language"
    assert result.compatibility == "incompatible"
    assert result.executable_candidate is False
