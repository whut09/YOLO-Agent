from __future__ import annotations

from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.mechanism_evidence import MechanismEvidenceExtractor
from yolo_agent.research.note_parser import (
    PaperDiagnosticHint,
    PaperEvidenceSummary,
    PaperMethodClaim,
)
from yolo_agent.research.schemas import PaperProvenance, PaperRecord


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


def test_note_hint_and_official_code_mentions_remain_source_scoped() -> None:
    paper = PaperRecord(
        paper_id="paper-3",
        title="Paper",
        year=2025,
        official_code_url="https://github.com/owner/teacher_student_distillation",
        provenance=PaperProvenance(
            source_repository="local",
            source_path="papers.json",
            source_record_hash="hash",
            importer_version="test",
            original_harness_hints=["Try small object oversampling."],
        ),
    )
    summary = PaperEvidenceSummary(
        paper_id=paper.paper_id,
        method_claims=[PaperMethodClaim(
            method_name="correlation loss",
            source_location="note:paragraph:2",
        )],
        diagnostic_hints=[PaperDiagnosticHint(
            paper_id=paper.paper_id,
            candidate_component_ids=["pseudo_iou"],
            source_location="harness_hints[1]",
        )],
    )

    evidence = MechanismEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper, evidence_summary=summary)

    by_mechanism = {item.canonical_component_id: item.source for item in evidence}
    assert by_mechanism["loss.quality.correlation"] == "note"
    assert by_mechanism["loss.quality.pseudo_iou"] == "harness_hint"
    assert by_mechanism["sampling.small_object"] == "harness_hint"
    assert by_mechanism["distillation.yolo26_teacher_student"] == (
        "official_code_metadata"
    )
    assert all(item.evidence_level == "paper_prior" for item in evidence)
