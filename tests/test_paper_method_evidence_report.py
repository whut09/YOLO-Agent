from __future__ import annotations

from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.method_profiles import PaperMethodProfileBuilder
from yolo_agent.research.paper_method_evidence_report import (
    build_method_evidence_coverage_report,
    write_method_evidence_coverage_markdown,
)
from yolo_agent.research.schemas import PaperRecord


def _paper(abstract: str) -> PaperRecord:
    return PaperRecord(
        paper_id="paper",
        title="Detector",
        year=2025,
        abstract=abstract,
        component_ids=["object_detection"],
    )


def test_reports_field_gaps_and_explicit_conversion_delta() -> None:
    builder = PaperMethodProfileBuilder(ComponentAliasResolver.from_yaml())
    previous = builder.build([_paper("Object detection study.")])
    current = builder.build([_paper(
        "Small object sampling modifies image weights in the train dataloader."
    )])

    report = build_method_evidence_coverage_report(
        current,
        previous=previous,
    )

    assert report.paper_count == 1
    assert report.audited_paper_count == 1
    assert report.authorizing_profile_count == 1
    assert report.baseline_insufficient_information_count == 1
    assert report.insufficient_information_count == 0
    assert report.insufficient_information_delta == -1
    assert report.converted_from_insufficient_paper_ids == ["paper"]
    assert report.audits[0].source_locations == ["summary"]
    assert report.audits[0].missing_fields == []
    assert report.report_hash


def test_prior_only_profile_retains_precise_missing_fields() -> None:
    report = PaperMethodProfileBuilder(
        ComponentAliasResolver.from_yaml()
    ).build([
        PaperRecord(
            paper_id="prior",
            title="Teacher Student Distillation for Detection",
            year=2025,
            abstract="Object detection study.",
            component_ids=["object_detection"],
        )
    ])

    audit = build_method_evidence_coverage_report(report).audits[0]

    assert audit.authorizes_method_profile is False
    assert "insertion_points" in audit.missing_fields
    assert "changed_variables" in audit.missing_fields
    assert audit.insufficiency_reasons == ["unresolved_paper_component_alias"]


def test_markdown_separates_profile_evidence_from_runtime_maturity(tmp_path) -> None:
    current = PaperMethodProfileBuilder(
        ComponentAliasResolver.from_yaml()
    ).build([_paper(
        "Small object sampling modifies image weights in the train dataloader."
    )])
    report = build_method_evidence_coverage_report(current)

    path = write_method_evidence_coverage_markdown(report, tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")

    assert "does not imply an implemented adapter" in text
    assert "Insufficient information" in text
