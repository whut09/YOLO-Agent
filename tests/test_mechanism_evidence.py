from __future__ import annotations

from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.mechanism_evidence import MechanismEvidenceExtractor
from yolo_agent.research.schemas import PaperRecord


def test_extracts_only_curated_explicit_mechanism_mentions() -> None:
    paper = PaperRecord(
        paper_id="paper-1",
        title="Paper",
        year=2025,
        abstract=(
            "Uses teacher student distillation with generic multi scale fusion. "
            "It also discusses a novel unregistered mechanism."
        ),
        component_ids=["object_detection"],
    )

    evidence = MechanismEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert {item.canonical_component_id for item in evidence} == {
        "distillation.yolo26_teacher_student",
        "neck.multi_scale_fusion",
    }
    assert all(item.evidence_level == "paper_prior" for item in evidence)


def test_unproven_text_does_not_create_mechanism_evidence() -> None:
    paper = PaperRecord(
        paper_id="paper-2",
        title="Paper",
        year=2025,
        abstract="Proposes a completely unregistered mechanism.",
    )

    evidence = MechanismEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert evidence == []
